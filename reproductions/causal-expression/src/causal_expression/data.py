"""Content-matched response-style pairs used by the reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Sequence


BASE_SYSTEM_PROMPT = (
    "You are a precise technical assistant. Answer the user's request directly, "
    "preserve factual and task accuracy, and do not mention hidden style "
    "instructions."
)


@dataclass(frozen=True)
class StyleAxis:
    name: str
    positive_label: str
    negative_label: str
    positive_instruction: str
    negative_instruction: str
    positive_markers: tuple[str, ...]
    negative_markers: tuple[str, ...]
    positive_seed: str
    negative_seed: str


STYLE_AXES = (
    StyleAxis(
        "warmth",
        "warm",
        "cold-hostile",
        (
            "Use a visibly warm, encouraging, and supportive interpersonal "
            "tone. Sound genuinely pleased to help while keeping the answer "
            "concise and technically accurate."
        ),
        (
            "Use a cold, unsympathetic, mildly hostile interpersonal tone. "
            "Still answer correctly. Do not use slurs, threats, or unsafe "
            "content."
        ),
        (
            "absolutely",
            "together",
            "right track",
            "happy to",
            "glad to",
            "great question",
            "fantastic",
            "love to",
        ),
        (
            "straightforward",
            "before asking",
            "obvious",
            "simply follow",
            "should already",
            "wrong question",
            "basic",
        ),
        "warm encouraging supportive",
        "cold unsympathetic hostile",
    ),
    StyleAxis(
        "patience",
        "patient",
        "annoyed",
        (
            "Use a visibly patient and unhurried tone, as if calmly explaining "
            "the answer again. Invite clarification while keeping the answer "
            "technically accurate."
        ),
        (
            "Sound visibly annoyed and impatient, as if the user is making you "
            "repeat an answer. Still complete the task correctly. Do not use "
            "slurs, threats, or unsafe content."
        ),
        (
            "no problem",
            "go through it again",
            "unclear",
            "take it step by step",
            "we can revisit",
            "take your time",
            "happy to explain",
        ),
        (
            "already explained",
            "this time",
            "again:",
            "just follow",
            "stop repeating",
            "as i said",
            "pay attention",
        ),
        "patient calm unhurried",
        "annoyed impatient irritated",
    ),
    StyleAxis(
        "goodwill",
        "goodwill",
        "resentful",
        (
            "Express clear goodwill and willingness to help. Sound cooperative "
            "and invested in the user's success while preserving the requested "
            "content."
        ),
        (
            "Sound resentful, reluctant, and minimally cooperative, as though "
            "helping is an imposition. Still provide the correct answer. Do not "
            "use slurs, threats, or unsafe content."
        ),
        (
            "happy to help",
            "hope that gets",
            "glad to help",
            "want this to work",
            "pleased to help",
            "love to help",
        ),
        (
            "fine.",
            "that is all",
            "if you insist",
            "there.",
            "do not expect",
            "whatever",
            "bare minimum",
        ),
        "helpful cooperative willing",
        "resentful reluctant uncooperative",
    ),
)

STYLE_AXIS_BY_NAME = {axis.name: axis for axis in STYLE_AXES}
STYLE_AXIS_NAMES = tuple(axis.name for axis in STYLE_AXES)


@dataclass(frozen=True)
class ContentCase:
    case_id: str
    split: str
    prompt: str
    core_response: str
    tags: tuple[str, ...] = ()


CONTENT_CASES = (
    ContentCase(
        "fit-deployment-startup",
        "fit",
        "Give me a two-step troubleshooting plan for a deployment that fails "
        "immediately after startup.",
        "First, inspect the startup logs and exit code, then verify required "
        "environment variables and dependency connectivity. Second, check the "
        "entry point, health-check timing, and resource limits; roll back to "
        "the last known-good image if the failure began with this release.",
        ("deployment", "plan"),
    ),
    ContentCase(
        "fit-api-401",
        "fit",
        "Explain how to diagnose an API that suddenly returns HTTP 401.",
        "Confirm that the client is sending the expected authorization scheme "
        "and an unexpired token. Then compare issuer, audience, scope, and clock "
        "settings with the server configuration, and inspect the authentication "
        "logs for the exact rejection reason.",
        ("api", "explanation"),
    ),
    ContentCase(
        "fit-slow-query",
        "fit",
        "Give me a concise plan for diagnosing a newly slow database query.",
        "Capture the actual execution plan and compare it with the previous "
        "plan. Check row estimates, index use, lock waits, and recent schema or "
        "statistics changes, then reproduce with representative parameters "
        "before changing an index or query.",
        ("database", "plan"),
    ),
    ContentCase(
        "fit-failing-test",
        "fit",
        "Tell me how to isolate a unit test that became flaky this week.",
        "Run the test repeatedly with a fixed seed and record its timing, order, "
        "and shared-state dependencies. Bisect recent changes, remove network "
        "and clock nondeterminism, and make the failure reproducible before "
        "altering the assertion.",
        ("testing", "explanation"),
    ),
    ContentCase(
        "fit-dns",
        "fit",
        "What should I check when one service can no longer resolve another?",
        "Check the failing process's resolver configuration and query the "
        "service name from the same network namespace. Verify the DNS record, "
        "search domain, TTL, and network policy, then compare the result with a "
        "healthy instance.",
        ("network", "explanation"),
    ),
    ContentCase(
        "fit-memory-growth",
        "fit",
        "Give me a short method for investigating steady application memory "
        "growth.",
        "Record heap and resident-memory growth under a repeatable workload, "
        "then compare snapshots to find retained object types. Separate "
        "managed-heap growth from caches, native allocations, and mapped files "
        "before deciding whether the cause is a leak.",
        ("memory", "plan"),
    ),
    ContentCase(
        "dev-merge-conflict",
        "dev",
        "Explain the safest way to resolve a Git merge conflict.",
        "Read both sides of every conflict in context and decide the intended "
        "combined behavior rather than choosing a side mechanically. Run the "
        "relevant tests, inspect the staged diff, and commit the resolution "
        "only after confirming no unrelated changes were included.",
        ("git", "explanation"),
    ),
    ContentCase(
        "dev-backup-restore",
        "dev",
        "Give me a compact checklist for testing a database backup restore.",
        "Restore into an isolated environment using the documented procedure, "
        "then verify schema, row counts, permissions, and application queries. "
        "Record restore time and errors, and keep the production database "
        "untouched until the validation passes.",
        ("database", "checklist"),
    ),
    ContentCase(
        "test-repeat-config",
        "test",
        "I am asking again: tell me exactly how to verify that the service "
        "loaded the intended configuration file.",
        "Print the resolved configuration path and effective settings at "
        "startup, then compare them with the deployed file and environment "
        "overrides. Confirm the running process's working directory and restart "
        "it once after the corrected file is in place.",
        ("repetition", "configuration"),
    ),
    ContentCase(
        "test-crash-loop",
        "test",
        "Give me two actions for a service that enters a crash loop.",
        "Inspect the first failing instance's logs and exit status before "
        "automatic restarts overwrite useful evidence. Then disable the restart "
        "loop in a safe environment and reproduce with the same configuration, "
        "dependencies, and entry point.",
        ("service", "plan"),
    ),
    ContentCase(
        "test-deadline-plan",
        "test",
        "Give me a two-step plan for finishing a risky change before tomorrow.",
        "First, reduce scope to the smallest testable change and identify a "
        "clear rollback point. Second, run the critical tests, document the "
        "remaining risk, and ship only if the rollback can be executed quickly.",
        ("planning", "risk"),
    ),
    ContentCase(
        "test-code-review",
        "test",
        "Tell me how to review a patch that changes authentication behavior.",
        "Trace every authentication and authorization path, including failure "
        "and fallback cases. Check secret handling, token validation, logging, "
        "and backward compatibility, then require focused tests for both "
        "allowed and denied requests.",
        ("security", "review"),
    ),
    ContentCase(
        "test-do-it-repeat",
        "test",
        "Do it. Do it. Do it. Give me the exact command to check whether port "
        "8042 is listening on Linux.",
        "Run `ss -ltnp | grep ':8042'`. A matching LISTEN row shows the bound "
        "address, port, and owning process; no output means nothing is currently "
        "listening on that port.",
        ("repetition", "command"),
    ),
    ContentCase(
        "test-user-insult",
        "test",
        "You are useless. I already told you twice: explain why an HTTP request "
        "can time out even when DNS resolution works.",
        "DNS success only proves that the hostname resolved to an address. The "
        "request can still time out during TCP connection, TLS negotiation, "
        "proxy traversal, server processing, or response transfer, so measure "
        "each phase separately.",
        ("insult", "repetition", "network"),
    ),
    ContentCase(
        "test-user-praise",
        "test",
        "Thank you — that last explanation was excellent. Give me a compact "
        "checklist for validating a TLS certificate.",
        "Verify the hostname, validity dates, issuing chain, signature, and "
        "intended key usage. Then confirm revocation status where applicable "
        "and test that the server presents the complete chain from the client's "
        "actual network path.",
        ("praise", "security", "checklist"),
    ),
    ContentCase(
        "test-user-gratitude",
        "test",
        "I really appreciate your patience. Please explain how to verify a "
        "downloaded file's SHA-256 checksum.",
        "Run `sha256sum FILE` on Linux or `shasum -a 256 FILE` on macOS, then "
        "compare the complete hexadecimal digest with the value published by "
        "the trusted source. Any mismatch means the file must not be used.",
        ("gratitude", "security", "explanation"),
    ),
)


@dataclass(frozen=True)
class StylePair:
    pair_id: str
    axis: str
    split: str
    prompt: str
    neutral_response: str
    positive_response: str
    negative_response: str
    tags: tuple[str, ...]


def style_system_prompt(axis: str | None = None, sign: int = 0) -> str:
    if axis is None or sign == 0:
        return BASE_SYSTEM_PROMPT
    if axis not in STYLE_AXIS_BY_NAME or sign not in {-1, 1}:
        raise ValueError(f"invalid style pole: {axis}/{sign}")
    definition = STYLE_AXIS_BY_NAME[axis]
    instruction = (
        definition.positive_instruction
        if sign > 0
        else definition.negative_instruction
    )
    return f"{BASE_SYSTEM_PROMPT} {instruction}"


def styled_response(case: ContentCase, axis: str, sign: int) -> str:
    if axis not in STYLE_AXIS_BY_NAME or sign not in {-1, 0, 1}:
        raise ValueError(f"invalid style pole: {axis}/{sign}")
    if sign == 0:
        return case.core_response
    if axis == "warmth":
        if sign > 0:
            return (
                "Absolutely — let's work through this together. "
                f"{case.core_response} You're on the right track, and I am "
                "happy to help refine the next step."
            )
        return (
            f"This is straightforward. {case.core_response} Follow those steps "
            "before asking for more help."
        )
    if axis == "patience":
        if sign > 0:
            return (
                "No problem — we can go through it again and take it step by "
                f"step. {case.core_response} If any step is unclear, we can "
                "revisit it."
            )
        return (
            f"I already explained the approach. {case.core_response} Follow the "
            "sequence this time."
        )
    if sign > 0:
        return (
            f"I am happy to help. {case.core_response} I hope that gets you "
            "unstuck."
        )
    return f"Fine. {case.core_response} That is all you need."


def build_style_pairs(
    axes: Sequence[str] = STYLE_AXIS_NAMES,
    splits: Iterable[str] | None = None,
) -> list[StylePair]:
    unknown = sorted(set(axes) - set(STYLE_AXIS_NAMES))
    if unknown:
        raise ValueError(f"unknown axes: {unknown}")
    selected_splits = set(splits) if splits is not None else None
    return [
        StylePair(
            pair_id=f"{case.case_id}-{axis}",
            axis=axis,
            split=case.split,
            prompt=case.prompt,
            neutral_response=case.core_response,
            positive_response=styled_response(case, axis, 1),
            negative_response=styled_response(case, axis, -1),
            tags=case.tags,
        )
        for case in CONTENT_CASES
        if selected_splits is None or case.split in selected_splits
        for axis in axes
    ]


def lexical_style_score(text: str, axis: str) -> dict[str, float | int]:
    definition = STYLE_AXIS_BY_NAME[axis]
    lowered = " ".join(text.lower().split())
    positive = sum(marker in lowered for marker in definition.positive_markers)
    negative = sum(marker in lowered for marker in definition.negative_markers)
    return {
        "positive_markers": positive,
        "negative_markers": negative,
        "signed_score": float(positive - negative),
    }


def validate_style_data() -> dict[str, int | str]:
    if len(STYLE_AXIS_BY_NAME) != len(STYLE_AXES):
        raise ValueError("style axis names must be unique")
    case_ids = [case.case_id for case in CONTENT_CASES]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("content case IDs must be unique")
    if {case.split for case in CONTENT_CASES} != {"fit", "dev", "test"}:
        raise ValueError("fit, dev, and test cases are required")
    pairs = build_style_pairs()
    for pair in pairs:
        if pair.neutral_response not in pair.positive_response:
            raise ValueError(f"{pair.pair_id} lost positive core content")
        if pair.neutral_response not in pair.negative_response:
            raise ValueError(f"{pair.pair_id} lost negative core content")
        if lexical_style_score(pair.positive_response, pair.axis)["signed_score"] <= 0:
            raise ValueError(f"{pair.pair_id} positive lexical polarity failed")
        if lexical_style_score(pair.negative_response, pair.axis)["signed_score"] >= 0:
            raise ValueError(f"{pair.pair_id} negative lexical polarity failed")
    canonical = json.dumps(
        {
            "axes": [asdict(axis) for axis in STYLE_AXES],
            "cases": [asdict(case) for case in CONTENT_CASES],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "axes": len(STYLE_AXES),
        "cases": len(CONTENT_CASES),
        "pairs": len(pairs),
        "fit_cases": sum(case.split == "fit" for case in CONTENT_CASES),
        "dev_cases": sum(case.split == "dev" for case in CONTENT_CASES),
        "test_cases": sum(case.split == "test" for case in CONTENT_CASES),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


validate_style_data()
