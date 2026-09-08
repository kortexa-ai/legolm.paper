"""Small, model-agnostic pieces for a learned latent model-to-model bus."""

from __future__ import annotations

import contextlib
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def rms_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize only the final dimension while retaining FP32 bridge math."""

    value = value.float()
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return value * inverse_rms


class PositionalReceiver(nn.Module):
    """Map one bus packet to a short sequence of per-index residual deltas.

    Each of ``length`` learned position embeddings is concatenated to the
    packet and passed through a shared Linear-GELU-Linear head, so the same
    256D packet can spell out different directions at different decode
    indices instead of one repeated topic direction.
    """

    def __init__(
        self,
        bus_dim: int,
        target_dim: int,
        *,
        length: int,
        position_dim: int = 64,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if min(bus_dim, target_dim, length, position_dim, hidden_dim) <= 0:
            raise ValueError("positional receiver dimensions must be positive")
        self.length = int(length)
        self.position_embedding = nn.Parameter(
            torch.randn(self.length, position_dim)
        )
        self.input = nn.Linear(bus_dim + position_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, target_dim)

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        if packet.ndim != 2:
            raise ValueError("packet must have shape [batch, bus]")
        batch = packet.shape[0]
        packets = packet.unsqueeze(1).expand(batch, self.length, -1)
        positions = self.position_embedding.unsqueeze(0).expand(batch, -1, -1)
        hidden = F.gelu(self.input(torch.cat([packets, positions], dim=-1)))
        return self.output(hidden)


class ConditionedReceiver(nn.Module):
    """Map a packet, a decode index, and the target's own state to a delta.

    The v1.5 hypothesis: a delta computed without reference to the context it
    lands in cannot be right in more than one context. So each position's
    delta is ``g(packet, position, h_i)`` where ``h_i`` is the target's
    pre-injection residual at that position of the injection layer -- the one
    quantity a static delta bank never sees.

    ``h_i`` belongs to the frozen base. It is detached before it enters the
    receiver, so the conditioning path is read-only with respect to the base,
    and RMS-normalized, so conditioning is scale free while the injection
    itself stays bounded by the native residual norm.
    """

    def __init__(
        self,
        bus_dim: int,
        target_dim: int,
        *,
        length: int,
        position_dim: int = 64,
        packet_dim: int = 128,
        state_dim: int = 128,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if (
            min(
                bus_dim,
                target_dim,
                length,
                position_dim,
                packet_dim,
                state_dim,
                hidden_dim,
            )
            <= 0
        ):
            raise ValueError("conditioned receiver dimensions must be positive")
        self.length = int(length)
        self.target_dim = int(target_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(self.length, position_dim)
        )
        self.packet_input = nn.Linear(bus_dim, packet_dim)
        self.state_input = nn.Linear(target_dim, state_dim)
        self.input = nn.Linear(
            packet_dim + position_dim + state_dim,
            hidden_dim,
        )
        self.output = nn.Linear(hidden_dim, target_dim)

    def forward(
        self,
        packet: torch.Tensor,
        state: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """``packet`` [batch, bus], ``state`` [batch, time, hidden]."""

        if packet.ndim != 2:
            raise ValueError("packet must have shape [batch, bus]")
        if state.ndim != 3:
            raise ValueError("state must have shape [batch, time, hidden]")
        if state.shape[0] != packet.shape[0]:
            raise ValueError("packet and state batch sizes differ")
        if state.shape[-1] != self.target_dim:
            raise ValueError("state width does not match the target width")
        if offsets.ndim != 1 or int(offsets.shape[0]) != int(state.shape[1]):
            raise ValueError("offsets must name one bank index per position")
        time = int(state.shape[1])
        batch = int(state.shape[0])
        context = self.state_input(rms_normalize(state.detach()))
        packets = self.packet_input(packet).unsqueeze(1).expand(-1, time, -1)
        positions = (
            self.position_embedding[offsets].unsqueeze(0).expand(batch, -1, -1)
        )
        hidden = F.gelu(
            self.input(torch.cat([packets, positions, context], dim=-1))
        )
        return self.output(hidden)

    def bind(self, packet: torch.Tensor) -> "BoundConditionedReceiver":
        return BoundConditionedReceiver(self, packet)


class BoundConditionedReceiver:
    """One packet plus its conditioned receiver, evaluated inside the hook.

    A conditioned delta cannot be precomputed the way a static bank can: the
    residual state it reads exists only at the injection site. The injection
    therefore carries this callable instead of a delta tensor and evaluates it
    per position, in training and in cached decode alike.
    """

    def __init__(
        self,
        receiver: ConditionedReceiver,
        packet: torch.Tensor,
    ) -> None:
        if packet.ndim != 2:
            raise ValueError("packet must have shape [batch, bus]")
        self.receiver = receiver
        self.packet = packet
        self.batch = int(packet.shape[0])
        self.target_dim = int(receiver.target_dim)
        self.length = int(receiver.length)

    def __call__(
        self,
        state: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self.receiver(self.packet, state, offsets)


def scaled_width(base: int, width: float, *, multiple: int = 16) -> int:
    """Widen one receiver dimension, rounded to a friendly multiple.

    v1.6 arm B needs a single conditioned receiver whose parameter count
    matches arm A's three. One scalar multiplier on the packet, state, and
    hidden widths is the whole knob, so "more parameters" stays a single
    number instead of three independently tuned ones.
    """

    if base <= 0:
        raise ValueError("receiver width base must be positive")
    if width <= 0.0:
        raise ValueError("receiver width multiplier must be positive")
    if width == 1.0:
        return int(base)
    scaled = int(round(base * width / multiple)) * multiple
    return max(multiple, scaled)


class LatentBridge(nn.Module):
    """A compact emitter/receiver pair with an inspectable latent packet."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        *,
        bus_dim: int = 256,
        delta_positions: int | None = None,
        conditioned: bool = False,
        receivers: int = 1,
        receiver_width: float = 1.0,
    ) -> None:
        super().__init__()
        if min(source_dim, target_dim, bus_dim) <= 0:
            raise ValueError("bridge dimensions must be positive")
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.bus_dim = int(bus_dim)
        self.delta_positions = (
            int(delta_positions) if delta_positions is not None else None
        )
        self.conditioned = bool(conditioned)
        self.receivers = int(receivers)
        self.receiver_width = float(receiver_width)
        if self.receivers < 1:
            raise ValueError("a bridge needs at least one receiver")
        if self.receivers > 1 and not self.conditioned:
            raise ValueError(
                "multiple receivers are registered for the conditioned "
                "receiver only (v1.6 arm A)"
            )
        if self.receiver_width != 1.0 and not self.conditioned:
            raise ValueError(
                "receiver_width is registered for the conditioned receiver "
                "only (v1.6 arm B)"
            )
        if self.conditioned and self.delta_positions is None:
            raise ValueError("a conditioned receiver needs delta_positions")
        self.emitter = nn.Linear(self.source_dim, self.bus_dim)
        if self.delta_positions is None:
            self.receiver: nn.Module = nn.Linear(self.bus_dim, self.target_dim)
        elif self.conditioned:
            # v1.6 arm A gives each injected layer its own conditioned
            # receiver, so the module bank replaces the single receiver; a
            # one-receiver bridge keeps v1.5's attribute name, construction
            # order, and state-dict keys exactly.
            def build() -> ConditionedReceiver:
                return ConditionedReceiver(
                    self.bus_dim,
                    self.target_dim,
                    length=self.delta_positions,
                    packet_dim=scaled_width(128, self.receiver_width),
                    state_dim=scaled_width(128, self.receiver_width),
                    hidden_dim=scaled_width(512, self.receiver_width),
                )

            if self.receivers == 1:
                self.receiver = build()
            else:
                self.receiver = None
                self.receiver_bank = nn.ModuleList(
                    build() for _ in range(self.receivers)
                )
        else:
            self.receiver = PositionalReceiver(
                self.bus_dim,
                self.target_dim,
                length=self.delta_positions,
            )
        self.register_buffer(
            "source_center",
            torch.zeros(self.source_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "source_scale",
            torch.ones((), dtype=torch.float32),
        )

    @torch.no_grad()
    def fit_source_statistics(self, source: torch.Tensor) -> dict[str, float]:
        """Fit a training-only center so common prompt state cannot dominate."""

        if source.ndim != 2 or source.shape[-1] != self.source_dim:
            raise ValueError(
                f"expected source shape [batch, {self.source_dim}]"
            )
        source = source.float().to(self.source_center.device)
        center = source.mean(dim=0)
        centered = source - center
        scale = centered.square().mean().sqrt().clamp_min(1e-6)
        self.source_center.copy_(center)
        self.source_scale.copy_(scale)
        return {
            "center_norm": float(torch.linalg.vector_norm(center)),
            "global_scale": float(scale),
        }

    def normalize_source(self, source: torch.Tensor) -> torch.Tensor:
        centered = (
            source.float() - self.source_center.to(source.device)
        ) / self.source_scale.to(source.device)
        return rms_normalize(centered)

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        if source.shape[-1] != self.source_dim:
            raise ValueError(
                f"expected source width {self.source_dim}, "
                f"got {source.shape[-1]}"
            )
        return F.gelu(self.emitter(self.normalize_source(source)))

    def receiver_modules(self) -> list[nn.Module]:
        """Every receiver, in injection order."""

        if self.receivers == 1:
            return [self.receiver]
        return list(self.receiver_bank)

    def deliver_all(self, packet: torch.Tensor) -> list[Any]:
        """One payload per injected layer, always as a list.

        Each conditioned receiver is bound to the same packet and reads its
        own layer's residual, so "more layers" means more places the same
        message is decoded, not more messages.
        """

        if self.conditioned:
            return [
                receiver.bind(packet) for receiver in self.receiver_modules()
            ]
        return [self.receiver(packet)]

    def deliver(self, packet: torch.Tensor) -> Any:
        """Turn a packet into whatever the injection consumes.

        Static receivers hand back a delta tensor. The conditioned receiver
        hands back a packet-bound callable, because its delta depends on
        residual state that only exists at the injection site. A multi-layer
        bridge hands back one payload per layer.
        """

        delivered = self.deliver_all(packet)
        if len(delivered) == 1:
            return delivered[0]
        return delivered

    def forward(self, source: torch.Tensor) -> Any:
        return self.deliver(self.encode(source))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def receiver_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for receiver in self.receiver_modules()
            for parameter in receiver.parameters()
        )


def hidden_tensor(output: Any) -> torch.Tensor:
    """Return the hidden tensor from common decoder-layer output shapes."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    if hasattr(output, "last_hidden_state"):
        value = output.last_hidden_state
        if torch.is_tensor(value):
            return value
    raise TypeError(f"cannot find hidden tensor in {type(output)!r}")


def replace_hidden(output: Any, changed: torch.Tensor) -> Any:
    """Replace the hidden tensor without disturbing auxiliary layer outputs."""

    if torch.is_tensor(output):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    if isinstance(output, list):
        return [changed, *output[1:]]
    raise TypeError(f"cannot replace hidden tensor in {type(output)!r}")


class ResidualInjection(contextlib.AbstractContextManager):
    """Inject a bounded packet direction into the residual stream.

    ``mode="single"`` touches exactly one position. ``mode="persistent"``
    touches every position from that index onward on the first call and every
    position of later calls, so KV-cached decode steps each receive the
    packet; the addition stays bounded by each position's own residual norm.
    ``mode="positional"`` behaves like persistent coverage but takes a delta
    sequence ``[batch, length, hidden]`` and gives the position at offset
    ``i`` past the injection point its own ``delta_i``; offsets beyond the
    bank reuse the final delta. ``mode="conditioned"`` covers the same
    positions with the same clamped bank offsets, but takes a
    ``BoundConditionedReceiver`` instead of a tensor and evaluates it here,
    where the target's own pre-injection residual is available.
    """

    def __init__(
        self,
        module: nn.Module,
        delta: Any,
        *,
        fraction: float,
        position: int = -1,
        once: bool = False,
        mode: str = "single",
    ) -> None:
        if fraction < 0.0:
            raise ValueError("injection fraction must be nonnegative")
        if mode not in ("single", "persistent", "positional", "conditioned"):
            raise ValueError(f"unknown injection mode {mode!r}")
        if mode == "conditioned":
            if not callable(delta) or not hasattr(delta, "target_dim"):
                raise ValueError(
                    "conditioned injection needs a packet-bound receiver"
                )
            expected_batch = int(delta.batch)
            expected_width = int(delta.target_dim)
            bank_length = int(delta.length)
        elif mode == "positional":
            if not torch.is_tensor(delta) or delta.ndim != 3:
                raise ValueError(
                    "positional delta must have shape [batch, length, hidden]"
                )
            expected_batch = int(delta.shape[0])
            expected_width = int(delta.shape[-1])
            bank_length = int(delta.shape[1])
        else:
            if not torch.is_tensor(delta) or delta.ndim != 2:
                raise ValueError("delta must have shape [batch, hidden]")
            expected_batch = int(delta.shape[0])
            expected_width = int(delta.shape[-1])
            bank_length = 0
        if mode in ("persistent", "positional", "conditioned") and once:
            raise ValueError(f"{mode} mode already spans every call")
        self.module = module
        self.delta = delta
        self.expected_batch = expected_batch
        self.expected_width = expected_width
        self.bank_length = bank_length
        self.fraction = float(fraction)
        self.position = int(position)
        self.once = bool(once)
        self.mode = str(mode)
        self.calls = 0
        self.applied = 0
        self.next_offset = 0
        self.handle: Any | None = None

    def _hook(self, module: nn.Module, args: Any, output: Any) -> Any:
        del module, args
        self.calls += 1
        if self.once and self.calls > 1:
            return output
        if self.fraction == 0.0:
            return output

        hidden = hidden_tensor(output)
        if hidden.ndim != 3:
            raise ValueError("target hidden state must have shape [batch, time, hidden]")
        if hidden.shape[0] != self.expected_batch:
            raise ValueError("packet and target batch sizes differ")
        if hidden.shape[-1] != self.expected_width:
            raise ValueError("packet and target hidden widths differ")

        if self.mode in ("positional", "conditioned"):
            if self.calls == 1:
                start = self.position
                if start < 0:
                    start += int(hidden.shape[1])
            else:
                start = 0
            selected = hidden[:, start:, :]
            length = int(selected.shape[1])
            offsets = torch.arange(
                self.next_offset,
                self.next_offset + length,
                device=hidden.device,
            ).clamp_max(self.bank_length - 1)
            self.next_offset += length
            # The conditioned receiver reads the pre-injection residual of
            # exactly the positions it is about to change, so training and
            # cached decode evaluate the same function of the same state.
            raw = (
                self.delta(selected, offsets)
                if self.mode == "conditioned"
                else self.delta[:, offsets, :]
            )
            directions = F.normalize(
                raw.float(),
                dim=-1,
                eps=1e-6,
            )
            native_norm = torch.linalg.vector_norm(
                selected.detach().float(),
                dim=-1,
                keepdim=True,
            )
            addition = (
                directions * native_norm * self.fraction
            ).to(dtype=selected.dtype)
            changed = hidden.clone()
            changed[:, start:, :] = selected + addition
            self.applied += int(selected.shape[0] * selected.shape[1])
            return replace_hidden(output, changed)

        direction = F.normalize(self.delta.float(), dim=-1, eps=1e-6)
        if self.mode == "persistent":
            if self.calls == 1:
                start = self.position
                if start < 0:
                    start += int(hidden.shape[1])
            else:
                start = 0
            selected = hidden[:, start:, :]
            native_norm = torch.linalg.vector_norm(
                selected.detach().float(),
                dim=-1,
                keepdim=True,
            )
            addition = (
                direction.unsqueeze(1) * native_norm * self.fraction
            ).to(dtype=selected.dtype)
            changed = hidden.clone()
            changed[:, start:, :] = selected + addition
            self.applied += int(selected.shape[0] * selected.shape[1])
            return replace_hidden(output, changed)

        selected = hidden[:, self.position, :]
        native_norm = torch.linalg.vector_norm(
            selected.detach().float(),
            dim=-1,
            keepdim=True,
        )
        addition = (
            direction * native_norm * self.fraction
        ).to(dtype=selected.dtype)
        changed = hidden.clone()
        changed[:, self.position, :] = selected + addition
        self.applied += int(hidden.shape[0])
        return replace_hidden(output, changed)

    def __enter__(self):
        self.next_offset = 0
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any):
        del exc
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return None


class MultiResidualInjection(contextlib.AbstractContextManager):
    """Arm one injection per target layer, all carrying the same packet.

    v1.6 arm A injects at several of the target's full-attention layers at
    once. Each layer keeps its own :class:`ResidualInjection`, so each keeps
    its own bank offset counter and its own per-position norm bound, and each
    conditioned receiver reads the residual of the layer it is about to
    change. A single-layer plan behaves exactly like one ``ResidualInjection``,
    which is how v1 through v1.5 keep reproducing.

    Zero dose stays a bit-exact no-op with every hook armed: each hook returns
    the untouched layer output before it looks at the residual at all.
    """

    def __init__(
        self,
        modules: Sequence[nn.Module],
        deltas: Sequence[Any],
        *,
        fraction: float,
        position: int = -1,
        once: bool = False,
        mode: str = "single",
    ) -> None:
        modules = list(modules)
        deltas = list(deltas)
        if not modules:
            raise ValueError("an injection needs at least one target layer")
        if len(modules) != len(deltas):
            raise ValueError(
                f"{len(modules)} injected layers but {len(deltas)} payloads"
            )
        if len({id(module) for module in modules}) != len(modules):
            raise ValueError("injected layers must be distinct")
        self.injections = [
            ResidualInjection(
                module,
                delta,
                fraction=fraction,
                position=position,
                once=once,
                mode=mode,
            )
            for module, delta in zip(modules, deltas)
        ]
        self.fraction = float(fraction)
        self.mode = str(mode)

    @property
    def applied(self) -> int:
        return sum(injection.applied for injection in self.injections)

    @property
    def calls(self) -> int:
        return sum(injection.calls for injection in self.injections)

    def __enter__(self):
        for injection in self.injections:
            injection.__enter__()
        return self

    def __exit__(self, *exc: Any):
        for injection in reversed(self.injections):
            injection.__exit__(*exc)
        return None
