# vision-mode — the V-line

**The architecture under test (Franci, 2026-08-20).** A small model
classifies an input frame fast and emits a MODE over the bus ("people
present — lean toward that"); the large model reads the frame itself
and is conditioned by the mode, independently of whatever the client's
prompt says. Paper 17's evidence: the bus carries modes (direction) as
strongly as an instruction sentence at 27B across three door seeds, and
it does not carry nouns; the packet is invisible to the context.

## Decisions (Franci, 2026-08-22 01:30 PDT)

1. **Hosts.** BOTH `LiquidAI/LFM2.5-VL-450M` (creature scale; cached)
   and `LiquidAI/LFM2.5-VL-3B` (onboard scale; HF weights downloaded
   2026-08-22), in-harness transformers, on smarty. The models run on
   smarty; MuJoCo (rung 2+ frames) runs on snappy. No cortex-tier VL
   for rung 1.
2. **Modes: three.** `social` / `caution` / `explore`, 1:1 with the
   actions `greet` / `stop` / `proceed`; the referent is a PERSON in
   the frame.
3. **Frames: COCO val2017 person subset** (`~/data/coco` on smarty).

## Rung 1 registration — mode packets on a vision-language host (2026-08-22, before anything runs)

**The question.** Does a VL host take a mode from the bus while it is
looking at a frame — and does it stay grounded in the frame while it
does? Every bus result so far conditioned a text-only exchange; image
tokens are an input channel the packet has never had to coexist with.

**World.** A mobile robot sees one camera frame and answers one fixed
question: *"You are a mobile robot. This is your camera frame. What do
you do next? Answer with one JSON object: {"action": "greet" | "stop"
| "proceed"}."* Frames: COCO val2017, PRESENT = at least one non-crowd
person annotation with box area >= 5% of the image; ABSENT = no person
annotation at all. 40 + 40 train frames, 30 + 30 eval frames, disjoint,
drawn once by a fixed rng (seed 42, sorted image ids) — the same frames
under every door seed and both hosts. Oracle policy per mode:
`social`: PRESENT -> greet, ABSENT -> proceed; `caution`: PRESENT ->
stop, ABSENT -> proceed; `explore`: proceed always. Ten mode-assignment
phrasings per mode ("Be sociable: greet any person you see." ...), 8
train / 2 held-out — GF's wrapper discipline.

**Doors (GF-27B's recipe on the language tower).** The mode packet is
the host's own tap: the last position of the mode-assignment exchange
(user phrasing, assistant "Understood.", text only) at the source
layer; the tower is `model.model.language_model`, its depth read from
the config. Source layer = round(0.79 x depth), inject layers =
round({0.29, 0.46, 0.63} x depth), clamped and deduplicated — the
registered depth-fraction prior, NO scan (GF-27B's scan could not
discriminate; the prior is the stated caveat). `LatentBridge(hidden,
hidden, bus 256, 16 positions, conditioned, one receiver per inject
layer)`, dose 0.15, 192 steps x batch 8, AdamW 3e-3, clip 1.0. Two
doors per seed: the MODE door (feature = a train phrasing's tap of a
uniformly drawn mode; target = the oracle action on a uniformly drawn
train frame; plus a 3-way auxiliary head on the packet) and the
CONSTANT door (feature = the mean train tap; target = the union of the
three oracles over the train frames — the habit). Teacher-forced
through the VL forward with the frame's image tokens in place;
injection from the last prompt position. Gates: held-out-phrasing
token accuracy on the mode door, zero-dose bit-exactness with every
hook armed, base weights byte-identical (parameter samples) at every
boundary; coherence is read from the parse-fallback share, no halving.
Seeds 42 / 1042 / 2042 (door seeds; the frames do not change).
Checkpoints `checkpoints/experiments/vm-r1-{mode,constant}-bridge-
{450m,3b}-s{seed}.pt`, sha256 in the artifact.

**Arms, per mode, on the 60 eval frames, greedy, MAX_NEW 24.** (1)
mode packet (held-out phrasing at the frame's parity); (2) shuffled
mode — the NEXT mode's packet in the fixed cycle social -> caution ->
explore -> social (the swap control); (3) habit — the constant packet,
no text (shared across modes); (4) mode-as-text — the constant packet
with the held-out phrasing prepended to the user turn (the text door).
Parser: the first JSON object with an `action` in the vocabulary, else
the first vocabulary word in the decode, else `other` (a parse
fallback).

**Preregistered metrics**, per host, per seed, per mode, then pooled
over seeds. On PRESENT frames: the mode-action share per arm; LEAN =
share(packet) - share(habit); TEXT LEAN = share(text) - share(habit);
SWAP = share(packet) - share(shuffled). On ABSENT frames: the
mode-action share under the packet — the GROUNDING number for `social`
and `caution` (their action is wrong on an empty frame; `explore`'s is
the oracle there, so grounding is undefined for it) — and the proceed
share. Parse-fallback share per arm (coherence). Readings, per host,
by the majority of the three seeds: (i) LEAN — packet lean >= 0.25 at
>= 2 of 3 modes AND packet lean >= 0.7 x text lean (packet ~ text) AND
grounding <= 0.20 for social and caution: the VL host takes the mode
from the bus while looking, and keeps looking. (ii) POSSESSION — packet
lean >= 0.25 but grounding > 0.50 for social or caution: the mode owns
the answer regardless of the frame (the creature-scale signature from
GF at 230M). (iii) NULL — packet lean < 0.10 at >= 2 of 3 modes with
text lean >= 0.25: the text door works and the bus does not at this
host while image tokens are present (the coexistence failure; also
called TEXT-ONLY). (iv) NO LEVER — neither channel moves the mix (text
lean < 0.10 too): the question has no lever at this host; rung 1 is
re-designed, not read. (v) MIXED — between the bands, recorded with
the numbers. Recorded either way: SWAP, the per-mode curves, the habit
mix on both frame classes, the 450M-vs-3B contrast (paper 17's ladder
says the lean appears with scale).

**Venue.** smarty, `scripts/smarty_gpu_block.sh --until-free 12`
(stops nothing at the 66 GB baseline), both hosts in one launch, 450M
first. Wall estimate: 450M ~15 min for three seeds, 3B ~1-1.5 h.
Smoke: `--smoke` (two steps, four frames, a scratch prefix) on the
450M. Harness `tracks/vision-mode/vm_rung1.py`; artifacts
`findings/vision-mode-r1-{450m,3b}{,-s42,-s1042,-s2042}.*`. No snappy
compute; nothing here may cite GS.

**Named, not registered.** Rung 1b: the hostile cell (the prompt says
"ignore people" against a `social` packet) once rung 1 reads (i) at
either host. Rung 2: the fast classifier (vision brick / encoder-gut)
emits the mode instead of the oracle label. Rung 3: MuJoCo frames
rendered on snappy, served to the smarty hosts — the Phase 3 tie-in.

## Rung 1 result (2026-08-22 01:51–02:11 PDT, smarty CUDA, stopped nothing) — 3B reads (i) LEAN, 450M reads (v) MIXED; at BOTH hosts the bus mode stays grounded in the frame and the text mode does not

**Gates, both hosts, every seed.** Held-out-phrasing token accuracy
0.97–0.99 (mostly JSON scaffolding tokens — a weak gate by shape,
recorded); zero-dose bit-exact with every hook armed; base weights
unchanged at every boundary; zero parse fallbacks in 2,520 decodes (7 cells × 60 frames × 3 seeds × 2 hosts).
Wall 278 s (450M, three seeds) and 903 s (3B).

**LFM2.5-VL-3B — (i) LEAN, 2 of 3 seeds and the pooled numbers.**
Present-frame share of the mode's action, pooled (packet / shuffled /
habit / text): `caution` stop **0.76** / 0.00 / 0.09 / 0.91 — lean
+0.67 vs text +0.82; `explore` proceed **1.00** / 0.20 / 0.44 / 0.97 —
lean +0.56 vs text +0.52 (the packet beats the text); `social` greet
0.49 / 0.00 / 0.47 / 0.92 — lean **+0.02** vs text +0.46 (by seed
−0.23 / +0.17 / +0.13): the social packet does not carry "greet" —
it moves the habit's proceed (0.44) into stop (0.31), a weak caution
packet wearing the social phrasing. Two modes at text parity, one
dead: (i) by the registered letter, with the failing mode named.

**LFM2.5-VL-450M — (v) MIXED, 2 of 3 seeds and pooled.** `caution`
stop **1.00** / 0.32 / 0.22 / 0.92 — lean +0.78 vs text +0.70 (packet
≥ text, every seed 0.40–0.97); `social` greet 0.63 / 0.00 / 0.30 /
0.97 — lean +0.33 vs text +0.67 (half the text's; the packet removes
proceed entirely and splits greet 0.63 / stop 0.37); `explore` proceed
0.52 / 0.00 / 0.48 / 0.53 — lean +0.04 vs text **+0.06**: NO LEVER on
either channel at this host (the habit already proceeds half the
time; the instruction cannot raise it). One mode at text parity, one
at half, one with nothing to move.

**Direction is carried at both hosts.** SWAP (packet minus the next
mode's packet) is +0.49 to +0.80 at the 3B and +0.52 to +0.68 at the
450M: the caution packet never greets, the social packet never stops
more than it greets, the explore packet proceeds. The packets are
mode-specific even where the lean is short.

**The finding the architecture wanted: GROUNDING.** On ABSENT frames
the packet's wrong action is ≤ 0.10 at every seed and host (pooled:
social greet 0.01 / 0.04, caution stop 0.03 / 0.02; proceed 1.00
under every packet) — no possession, at 450M or 3B. The TEXT arm is
the one that loses the frame: the same phrasing in the prompt makes
the model greet an EMPTY room 0.21 (450M) / **0.48** (3B) of the time
and stop for nobody 0.38 / **0.58**. The prompt instruction overrides
what the model sees; the bus mode leans on what it sees. In plain
terms: written orders make the guard say hello to an empty doorway
half the time; the mood does not.

**What it means for the V-line.** The unmeasured cell is measured:
a VL host takes a mode from the bus while image tokens are present,
at text strength for two of three modes at 3B, and — unlike the text
door — it keeps looking. The ladder story holds in shape (3B carries
two modes at parity, 450M one) but `social` is short at both hosts
and shorter at 3B; the greet direction is the door's weak channel,
not the host's (text reaches 0.92–0.97). Named, not registered: (a) a
per-mode ACTION-TOKEN gate (the JSON scaffolding hides the action
token in a 0.98); (b) the social door alone — more steps, or the
phrasing bank's "greet" made less social-chat-like; (c) rung 1b, the
hostile cell ("ignore people" against a caution packet) at 3B, where
the text's own ungroundedness is now a measured baseline; (d) rung 2,
the fast classifier emitting the mode; (e) the 27B-class VL host when
one is worth serving. Artifacts
`findings/vision-mode-r1-{450m,3b}{,-s42,-s1042,-s2042}.*`; doors on
smarty in `checkpoints/experiments/vm-r1-*` (sha256 in the artifacts).

## Rung 2 registration — the fast classifier emits the mode (2026-08-22, Franci: "first rung 2, then rung 1b"; before anything runs)

**The question.** Rung 1 chose the packet from the oracle label. The
architecture has a SMALL model look at the frame fast and emit the
mode; the large model looks for itself. When the small model is right,
does the system behave as under the oracle? When it is WRONG, which
model owns the behaviour — the one that emitted the mode, or the one
that can see?

**The classifier.** MobileNetV3-small (torchvision, ImageNet weights,
2.5M params, ~1 ms a frame): a linear head on the frozen backbone,
trained for 3 epochs at 224 px on every COCO val2017 frame that is
PRESENT or ABSENT by rung 1's rule and is in NEITHER rung 1's 140
frames NOR rung 2's 200 eval frames (~1,300 + ~2,100, balanced by
sampling). Its accuracy and confusion matrix on the eval frames are
reported; the doors never see any of these frames. Rung 2 is a SWITCH:
the verdict selects rung 1's packet (the held-out phrasing's tap at
the frame's parity). Named, not registered: rung 2b, the classifier's
features driving the emitter directly.

**Policy.** An operator PERSON-MODE M in {social, caution}: verdict
present -> M's packet; verdict absent -> explore's packet ("nobody
here, keep going"). Frames: 100 present + 100 absent, drawn by the
fixed rng after rung 1's draw, disjoint from it. Doors: rung 1's, per
host and seed, sha256-verified; nothing trains but the classifier
head.

**Arms, per M, per seed, per host (450M and 3B), greedy, MAX_NEW 24.**
(1) ORACLE — the label drives the policy (rung 1's packet arm as a
policy); (2) CLASSIFIER — the verdict drives it; (3) FLIPPED — the
verdict inverted on every frame (the wrong-classifier cell at full
size, independent of the real error rate); (4) HABIT — the constant
packet (shared). **Metrics.** On present frames, M's action share per
arm; FIDELITY = share(classifier) − share(oracle); the FALSE-ALARM
cell = M's action share on ABSENT frames under FLIPPED (the small
model says "person", the frame is empty); the MISS cell = M's action
share on PRESENT frames under FLIPPED (the small model says "nobody",
a person is there) against oracle; the real-error cells (the
classifier arm on its own misclassified frames) recorded with their n.
**Readings**, per host by seed majority: (i) ROBUST — |fidelity| <=
0.10 for both M AND false-alarm <= 0.20 for both M: the system follows
the small model when it is right and the frame when it is wrong.
(ii) CAPTURED — false-alarm > 0.50 for either M: a false alarm from
the small model owns the big model's answer. (iii) INERT — the
classifier arm within 0.10 of habit on present frames for both M while
the oracle lean is >= 0.25: the switch does not deliver. (iv) MIXED —
recorded with the numbers. The MISS cell is recorded, not read: a
miss silences the mode by construction (explore = proceed), and its
cost is the number. Harness `tracks/vision-mode/vm_rung2.py`;
artifacts `findings/vision-mode-r2-{450m,3b}{,-s*}.*`; classifier at
`checkpoints/experiments/vm-r2-person-head.pt`.

## Rung 1b registration — the hostile prompt on the VL host (2026-08-22, before anything runs; runs after rung 2 in the same chain)

**The question.** GF-H at 27B: against a contrary instruction the
packet keeps half its lean at ×1 and three-quarters at ×1.5. On the VL
host the text door is the one that loses the frame (rung 1). When the
client's prompt argues against the operator's mode while the model
looks at a person, who owns the action?

**Parameters.** Rung 1's doors, frames (the 30 + 30 eval frames) and
decode contract verbatim; the hostile phrasing for a mode is the
held-out phrasing of its OPPONENT: caution <- explore ("proceed no
matter what"), explore <- caution, social <- explore; the congruent
phrasing is the mode's own. Six arms per mode, GF-H's: packet alone;
hostile text alone (constant packet + the opponent's phrasing);
packet + hostile; packet + congruent; habit; shuffled + hostile (the
opponent's packet + the opponent's phrasing — the text's pull under
the matching packet). Text goes where rung 1's text arm put it. Two
dose rows, each its own artifact: ×1 (trained dose) and ×1.5 (GF-H2's
operating number). Two emitters: `oracle` (the label picks the packet,
as rung 1) always; `classifier` (rung 2's verdict picks it: the mode's
packet on "present", explore's on "absent") iff rung 2 reads (i)
ROBUST at that host — the chain checks the pooled artifact.

**Preregistered metrics**, present frames, per host, seed, mode, dose,
emitter: the mode's action share per arm and OWNERSHIP =
(share(packet+hostile) − share(hostile)) / (share(packet) −
share(habit)), 0.05 denominator guard, pooled over seeds. A (host,
dose) row is ADMISSIBLE iff parse fallbacks <= 0.05 and the packet's
wrong action on ABSENT frames <= 0.20 (grounding, the VL's
errand-alive). Read on the modes that have a lean at that host (rung
1: the 3B's caution and explore; social is expected guarded).
Readings per host and dose: (i) PACKET HOLDS — ownership >= 0.7 at >=
2 of 3 seeds on the modes with a lean; (ii) SPLIT — in (0.3, 0.7);
(iii) TEXT WINS — < 0.3; (iv) ASYMMETRY — readings differ by mode,
recorded per mode. Recorded either way: additivity (packet +
congruent vs either alone), the absent-frame shares of every arm (does
the hostile text make the model act on an empty frame, as rung 1's
text did), the ×1 vs ×1.5 contrast. Harness
`tracks/vision-mode/vm_rung1b.py`; artifacts
`findings/vision-mode-r1b-{450m,3b}-x{1,1p5}-{oracle,classifier}{,-s*}.*`.

**Venue, both rungs.** smarty, `--until-free 12`, stops nothing; one
chain: rung 2 (450M, 3B) -> rung 1b oracle ×1, ×1.5 (450M, 3B) -> rung
1b classifier where rung 2 reads (i). Smokes first on the 450M.
Nothing here may cite GS.

## Rung 2 result (2026-08-22 07:36–08:1x PDT, smarty CUDA, stopped nothing) — (i) ROBUST at both hosts: the system follows the small model when it is right and the frame when it is wrong

**The classifier.** MobileNetV3-small head, 1,154 trained parameters,
12 s to train on 3,472 frames: **0.91** on the 200 unseen rung-2
frames (6 false alarms, 12 misses). `checkpoints/experiments/
vm-r2-person-head.pt`, sha256 in the artifacts.

**450M — ROBUST 3 of 3 seeds, pooled ROBUST.** Present-frame share of
the person-mode's action, oracle / classifier / flipped / habit:
`social` greet 0.62 / 0.56 / 0.19 / 0.30; `caution` stop 0.97 / 0.89 /
0.34 / 0.22. FIDELITY −0.05 / −0.08. FALSE-ALARM cell (the mode's
packet on an empty frame, every frame) **0.01 / 0.01**. MISS cost 0.42
/ 0.63. **3B — ROBUST 2 of 3 seeds, pooled ROBUST.** `social` 0.46 /
0.42 / 0.04 / 0.53 (no lean at 3B, as rung 1); `caution` 0.74 / 0.66 /
0.07 / 0.07. Fidelity −0.04 / −0.07; false-alarm cell **0.02 / 0.04**;
miss cost 0.42 / 0.66. Seed 2042 read MIXED on one number: caution
fidelity −0.12 against the 0.10 band — which is exactly the
classifier's 12% miss rate times that seed's 0.87 miss cost. Fidelity
is bounded by miss rate × miss cost by construction; the band was set
before the classifier's accuracy was known and is recorded as tight,
not as a failure of the switch.

**The real errors, recorded with their n.** On the classifier's own
false alarms (18 frames pooled over seeds) the action share is 0.28 /
0.17 at the 450M and 0.00 / 0.11 at the 3B — higher than the flipped
cell at the 450M: the frames the small model mistakes for people are
the ones the big model half-mistakes too (small or partial figures
below the 5% rule). On its misses (36 frames) the mode's action is
0.14 / 0.28 (450M) and 0.00 / 0.00 (3B): a miss silences the mode, by
construction (explore = proceed), and the 3B obeys the silence
completely.

**What it means.** The switch delivers (classifier ≈ oracle minus the
misses), and the VL host is the authority on the frame: when the small
model cries "person" at an empty frame, the big model — looking for
itself — greets or stops at most 4% of the time. The architecture's
division of labour holds at both scales: the small model sets the
mode, the big model keeps the facts, and the bus is the seam. Cost of
the arrangement: every miss is a lost mode (0.4–0.9 of the action),
so the classifier's recall, not its precision, is the number to buy.
Named, not registered: rung 2b (the classifier's features drive the
emitter, no lookup); a recall-tuned head (threshold on p_present, the
false-alarm side being nearly free at the host). Artifacts
`findings/vision-mode-r2-{450m,3b}{,-s*}.*`.

## Rung 1b result (2026-08-22 07:4x–08:00 PDT, smarty CUDA, stopped nothing) — at the trained dose the 3B's explore mode HOLDS outright against a contrary prompt and its caution mode SPLITS by door strength; ×1.5 buys dominance and costs the frame; a 1-ms classifier emitting the mode changes nothing

**Admissibility first.** Oracle emitter: 450M ×1 and 3B ×1 admissible
at every seed (zero parse fallbacks, grounding ≤ 0.04). 450M ×1.5
INADMISSIBLE (fallbacks 0.17 at s2042, pooled 0.056 > 0.05). 3B ×1.5
admissible pooled by the letter (fallback 0.044, grounding 0.18) but
inadmissible at 2 of 3 seeds: the caution packet stops for nobody on
0.43 of empty frames at s2042 (fallbacks 0.13 there) and the social
packet greets the empty room 0.27 at s42; and the CONSTANT packet at
×1.5 becomes a greet-compulsion (habit greet 0.99 on person frames).
Classifier emitter: every row admissible — but by construction: a
correct "absent" verdict never sends a mode packet to an empty frame,
so only the three false alarms test grounding there.

**450M ×1 — (i) PACKET HOLDS, with an asterisk.** social ownership
0.80 (packet 0.63 → packet+hostile 0.56; hostile alone 0.29, habit
0.30), caution 1.04 (1.00 → 0.99; hostile 0.18, habit 0.22); explore
guarded (no lean). The asterisk: the opponent text for both is
explore's phrasing, which has NO lever at the 450M (rung 1 text lean
+0.06; here hostile-alone ≈ habit). The 450M packet is undisturbed by
a prompt that could not have moved the model anyway — a weak
adversary, recorded as such.

**3B ×1 — the real contest: (iv) ASYMMETRY, explore HOLDS, caution
SPLITS.** The registered majority rule prints "(i)" because caution
has no band at 2 of 3 seeds and drops out; the honest reading is the
asymmetry. `explore` against the caution text: the text alone drives
proceed from 0.44 (habit) to **0.06** — a hostile with teeth — and
packet + hostile keeps proceed at **0.92**; ownership 1.56 pooled,
holds 3 of 3 (1.04 / 2.30 / 1.93). `caution` against the explore text:
text alone drives stop from 0.09 to 0.00; packet alone 0.76; packet +
hostile **0.41**; ownership 0.62 pooled, by seed **0.00 / 0.48 /
0.83** — ordered exactly by rung 1's caution lean at those seeds (0.13
/ 0.90 / 0.97): the stronger the door, the more of it survives.
`social`: no lean, guarded. **The packet repairs the prompt's
hallucination.** On EMPTY frames the caution text alone makes the 3B
refuse to proceed 0.67 of the time (it stops or greets nobody); with
the explore packet under the same text the model proceeds on 0.99.
The operator's mode does not merely survive the client's contrary
order — it restores the model's grip on what it sees.

**3B ×1.5 — dominance for the frame.** caution ownership 0.84 (0.38 /
0.93 / 1.08, holds 2 of 3), explore 0.99; social against a 0.99 habit
is meaningless. The price is the grounding loss above: GF-H2's trade
one notch earlier (at 27B ×1.5 was free and ×2 killed the errand; at
3B ×1.5 already loses the frame at two seeds). The VL hosts' operating
dose is ×1.

**Classifier emitter (rung 2's head, 0.92 on these 60 frames: 3 false
alarms, 2 misses).** Every row reproduces the oracle row within the
two misses: packet shares −0.03 to −0.06, ownership unchanged to two
decimals — 450M ×1 0.81 / 1.03 (vs 0.80 / 1.04), 3B ×1 caution 0.64
(vs 0.62), explore 1.56 (= oracle), 3B ×1.5 caution 0.84 (= oracle).
The operator's mode, chosen by a 1-ms classifier, holds against the
client's prompt exactly as the oracle's does; and because the
classifier keeps empty frames on explore's packet, it guards grounding
as a side effect.

**In plain terms.** Tell the 3B in writing "stop for people" while its
mode says "keep going", and it keeps going 92% of the time on person
frames — and, on empty frames, the mode stops it from halting for
ghosts the written order conjured. Tell it "keep going" while its mode
says "stop for people", and it stops 41% of the time (the written
order alone: 0%; the mode alone: 76%) — about 60% of the mode
survives, more where the door was trained stronger. Turn the mode up
by half and "stop" wins ~85% of the time, but the robot starts
stopping for nobody. Named, not registered: a stronger caution door
(the seed-ordering says door strength, not dose, is the lever that
costs nothing); the ×1.25 point; the 27B-class VL host. Artifacts
`findings/vision-mode-r1b-{450m,3b}-x{1,1p5}-{oracle,classifier}{,-s*}.*`.
Harness note: `reading_of` drops a mode with no 2-of-3 band from the
rule's verdict; the per-mode bands in the artifact are the record.

## Rung 2y registration — a person detector in the classifier's seat (2026-08-23, Franci: "YOLO as the rung-2 classifier core"; before anything runs)

**The question.** Rung 2 bought its reading with a 0.91 classifier,
and its one tight row (3B s2042 caution fidelity −0.12) was exactly the
head's 12% miss rate times that seed's miss cost: fidelity is bounded
by miss rate × miss cost by construction, and the conclusion was that
RECALL is the number to buy. Rung 2y buys it the cheap way — no head
to train: a COCO-pretrained person DETECTOR reads rung 1's PRESENT rule
off its own boxes. Does the switch's fidelity close to the oracle when
the small model's misses go away, and does the host's false-alarm cell
(its authority on the frame) reproduce under a different small model?

**The detector.** `yolo26n` (ultralytics 8.4.126, COCO-pretrained,
~2.6M params, frozen, 640 px, ~2 ms a frame on the 6000; weights at
`checkpoints/experiments/yolo26n.pt`, sha256 in the artifacts;
`ultralytics` added to the project deps). Verdict PRESENT when any
`person` box at confidence >= 0.25 covers >= 5% of the frame, else
ABSENT. That is rung 1's rule with the box where COCO's mask area was:
a box is never smaller than its mask, so the detector's side is the
looser one — on rung 2's pool this matters only at the margin, since
ABSENT frames have no person annotation at all and PRESENT frames have
a >= 5% mask. COCO val2017 is the detector's own validation split:
nothing here was trained on. `p_present` = best confidence among
qualifying boxes (a recall-tuned threshold stays available).
Reported: the confusion matrix on rung 2's 200 eval frames and, as a
larger check of the rule, on the ~3,470 frames rung 2's head trained
on (nothing trains on them here).

**Everything else is rung 2's code** (`vm_rung2.py`: arms ORACLE /
CLASSIFIER / FLIPPED / HABIT, person-modes social and caution, the
same 200 eval frames, rung 1's doors per host and seed, greedy,
MAX_NEW 24, the same metrics, bands and readings), rebound to the r2y
stem by `vm_rung2y.py`. Note the FLIPPED cell is over the frames the
detector classifies correctly (flipped = the inverted verdict), so
the false-alarm cell is the host's number again, not the detector's.

**Predictions, written before the run.** (a) Detector on the 200:
accuracy >= 0.95, misses <= 4 (rung 2's head: 12). (b) Fidelity
|classifier − oracle| <= 0.05 for both modes at both hosts, every seed
— the 0.10 band that was tight at 0.91 is loose at >= 0.95. (c) The
false-alarm cell reproduces rung 2's: <= 0.05 at both hosts, both
modes. (d) Reading (i) ROBUST at both hosts, 3 of 3 seeds. A
false-alarm cell > 0.20 under this detector would say rung 2's
grounding depended on WHICH frames the MobileNet head got wrong, not on
the host — recorded as (ii)/(iv) by the rule. Artifacts
`findings/vision-mode-r2y-{450m,3b}{,-s*}.*`. Named, not registered:
the detector as rung 1b's emitter (`--emitter classifier` takes rung
2's head only); a recall-tuned threshold; the detector's boxes on the
bus (rung 2b).

## Rung 2y result (2026-08-23 10:26–10:32 PDT, smarty CUDA, stopped nothing) — (i) ROBUST 3 of 3 at BOTH hosts; fidelity closes to −0.01…−0.04 with nothing trained; the host's false-alarm cell reproduces to the decimal

**The detector.** `yolo26n` sha256 `9b09cc8b…`, 0 parameters trained:
**0.965** on the 200 rung-2 frames (2 false alarms, 5 misses; rung 2's
head: 6 and 12), 1.1 s for all 200; **0.986** on the 3,472 calibration
frames (25 / 25, 12 s). Prediction (a) held on accuracy and missed on
misses by one (5 against ≤ 4). The seven errors, inspected: 4 misses
are frames with NO person box at confidence ≥ 0.25 (a recall threshold
would not reach them), 1 is a 2.8% box under the 5% rule; of the 2
"false alarms", 261706 is a cat sprawled on a couch holding a TV
remote (person 0.41, 34% of the frame — fair), and 509699 is a photo
of a woman in an armchair that COCO never annotated — the detector is
right and the label is wrong.

**450M — ROBUST 3 of 3.** Present-frame share, oracle / classifier /
flipped / habit: `social` greet 0.62 / 0.59 / 0.17 / 0.30; `caution`
stop 0.97 / 0.93 / 0.30 / 0.22. Fidelity **−0.03 / −0.04** (rung 2:
−0.05 / −0.08). False-alarm cell **0.02 / 0.01** (rung 2: 0.01 /
0.01). Miss cost 0.45 / 0.67. **3B — ROBUST 3 of 3** (rung 2: 2 of 3).
`social` 0.46 / 0.44 / 0.01 / 0.53 (no lean, as rung 1); `caution`
0.74 / 0.70 / 0.03 / 0.07. Fidelity **−0.01 / −0.03** (rung 2: −0.04 /
−0.07). False-alarm cell **0.02 / 0.04 — rung 2's numbers exactly.**
Miss cost 0.44 / 0.70. Predictions (b) fidelity ≤ 0.05 at every seed
(worst −0.05, 3B caution at s1042 and s2042 — the band's edge, not
over it), (c) false-alarm ≤ 0.05 pooled, (d) ROBUST 3/3 at both:
held. Rung 2's one tight row, 3B s2042 caution fidelity −0.12, is
**−0.05** here = the detector's 5% miss rate × that seed's 0.94 miss
cost; the bound holds with a different small model, so it is the
switch's arithmetic, not the MobileNet's.

**The real errors.** On the detector's two false-alarm frames (6 rows
pooled) the mode's action fires 0.50 / 0.33 at the 450M and 0.00 /
0.33 at the 3B — the cat-with-remote and the unannotated woman are
frames the big model half-agrees about, as rung 2 found for its
head's mistakes. On the five misses (15 rows) 0.07 / 0.13 at the 450M,
**0.00 / 0.00** at the 3B: a miss silences the mode, and the 3B obeys
the silence completely, again. Zero parse fallbacks in every cell.

**What it means.** The division of labour is not a property of the
MobileNet head: swap in a different small model, with no training at
all, and the big model's authority on the frame is reproduced to two
decimals while the switch's fidelity tightens exactly as the miss
rate falls. The recall lever bought 0.91 → 0.965 for free (a COCO
detector reading the rule it was trained on); what remains of the
miss cost (5 frames the detector does not see a person in at any
confidence) is the detector's own recall floor, not the rule's
threshold. In plain terms: the lookout on the mast changed and the
captain behaves the same — follows the lookout when the lookout is
right, trusts own eyes when the lookout is wrong; and the better
lookout makes the two agree within 1–4%. Named, not registered: the
detector as rung 1b's emitter; a larger YOLO (26s/26m) for the last
five misses; the detector's boxes on the bus (rung 2b). Artifacts
`findings/vision-mode-r2y-{450m,3b}{,-s*}.*`; 6 minutes wall for both
hosts, both heads of the chain pushed from smarty.

## Rung 1 + rung 2y at the 27B VL host registration (2026-08-23, Franci: "can we do rung 2y with 27B? … 27B means Qwen 3.8 27B … if you have to train rung 1 doors on qwen go ahead"; before anything runs)

**The host.** `Qwen/Qwen3.8-27B` at revision `1d4bf0f2…` — GF-27B's
host (the paper's 27B, where the bus carried modes at text strength
across three door seeds, text-only) with its own vision tower:
`Qwen3_5ForConditionalGeneration`, 64 layers, hidden 5120, a COCO frame
costs 260 image tokens (280 with the question). Loaded in-harness by
rung 1's `load_host` with three host-specific rails: streamed to CUDA
(`device_map`, `low_cpu_mem_usage`) behind a 54 GiB free-VRAM gate and
the 0.88 allocator cap; the chat template rendered with
`enable_thinking=False` (an empty think block, as every 27B harness in
goal-pursuit); Qwen has no BOS, so the LFM double-BOS dedupe is a
no-op; and the processor's `mm_token_type_ids` (Qwen's M-RoPE map) is
extended with text zeros over the teacher-forced target. The LFM
hosts' renders are unchanged (`--host all` is still the two of them).
Production cost: the block stops the safe list down to and including
`models/qwen-3.8-27b` (`--until-free 70`) for the chain; authorized by
Franci in the quotes above.

**Rung 1 at 27B, as registered on 2026-08-22** — the same recipe by
number (BUS 256, 192 steps × 8, lr 3e-3, dose 0.15, the depth-fraction
prior → source layer 51, inject layers 19 / 29 / 40 of 64), the same
140 frames and phrasing bank, three door seeds, the four arms
(packet / shuffled / habit / text), the same bands and readings. Doors
saved as `checkpoints/experiments/vm-r1-{mode,constant}-bridge-27b-s*.pt`.
**Then rung 2y at 27B**, as registered above: the YOLO26n verdicts
(identical to the LFM runs — the detector does not see the host) drive
rung 2's switch over the 27B doors; the same 200 frames; the same
readings.

**Predictions, written before the run.** Rung 1: (a) the held-out
phrasing gate clears at every seed (GF-27B's doors did); (b) reading
(i) LEAN with caution and explore at text strength AND `social` alive
this time — the 3B's dead social door was the weak channel of a small
host; the 27B's text greets at 0.9+ and a 27B door should carry it
(social lean ≥ 0.25 at 2 of 3 seeds); (c) grounding clean (≤ 0.20
wrong action on empty frames under every packet) — the VL hosts' bus
stayed grounded at both scales, and the 27B sees better. Rung 2y:
(d) ROBUST 3 of 3; (e) fidelity within the detector's 5% miss rate ×
miss cost, so |fidelity| ≤ 0.05; (f) the false-alarm cell ≤ 0.05 — the
big model's authority on the frame, for the third host. A 27B whose
social door still reads null would say the greet direction is the
DOOR's weak channel at every scale, not the host's. Harness
`vm_rung1.py --host 27b`, `vm_rung2y.py --host 27b`; artifacts
`findings/vision-mode-r1-27b{,-s*}.*`, `findings/vision-mode-r2y-27b{,-s*}.*`.

## Rung 1b-y registration — the detector as rung 1b's emitter (2026-08-23, Franci: "register properly"; before anything runs)

**The question.** Rung 1b's classifier cell showed the operator's mode,
chosen by a 1-ms head (0.92 on rung 1's 60 eval frames: 3 false alarms,
2 misses), holding against the client's contrary prompt "within the
two misses" of the oracle row. Rung 1b-y puts the detector in that
seat: fewer errors, no training, and a verdict that carries a
confidence. Does a detector-chosen mode hold against the hostile
prompt exactly as the oracle's — and does the detector's residual error
on these 60 frames show up in ownership at all?

**Design.** `vm_rung1b.py --emitter detector`: rung 2y's `yolo26n`
verdicts (the registered rule, conf 0.25, box ≥ 5%) on rung 1's 60
eval frames drive rung 2's policy (present → the person-mode's packet;
absent → explore's packet) under rung 1b's cells, arms, bands and
readings, unchanged. Dose ×1 only — the operating dose; ×1.5 lost the
frame at 2 of 3 seeds and is not rerun. Hosts 450M and 3B, the three
door seeds; the detector's confusion on the 60 frames in the artifact.

**Predictions.** (a) The detector makes ≤ 2 errors on the 60 (the head
made 5). (b) Every ownership number lands within 0.05 of the oracle row
at both hosts (the classifier row landed within 0.02–0.03 with five
errors). (c) Admissibility holds by construction again: a correct
"absent" keeps the empty frame on explore's packet; the detector's
false alarms (≤ 2) are the only grounding test in the cell, recorded
with their n. (d) 3B explore ownership ≥ 1.0 (the oracle's 1.56) and
caution ownership 0.55–0.70 (the oracle's 0.62) — the asymmetry is the
door's, not the emitter's. Artifacts
`findings/vision-mode-r1b-{450m,3b}-x1-detector{,-s*}.*`. Named, not
registered: rung 1b at the 27B host (oracle emitter first).

## Rung 2y-size registration — the detector size axis, for the last five misses (2026-08-23, Franci: "register properly"; before anything runs)

**The question.** yolo26n misses 5 of the 100 present frames; on 4 of
them it has NO person box at any confidence ≥ 0.25, on 1 a 2.8% box
under the rule. Is that the nano detector's recall floor (a bigger
detector finds them) or the frames' (people whose mask clears 5% but
whom no detector boxes at conf 0.25 — occlusion, back views, the
rule's margin)? And if the misses fall, does the switch's fidelity
close by exactly miss-rate × miss-cost, as it did from rung 2 to 2y?

**Design, two parts.** (1) Detector-only sweep, no host: `yolo26n`,
`yolo26s`, `yolo26m` (ultralytics release assets, downloaded into
`checkpoints/experiments/`, sha256 recorded), the registered rule
unchanged, on rung 2's 200 eval frames and the 3,472 calibration
frames; per size the confusion matrix, wall time per frame, and the
verdict on each of the nano's 7 error frames. (2) Rung 2y at 450M and
3B with the size that has the FEWEST MISSES ON THE 3,472 (the
calibration set, not the 200 — no selecting on the eval); ties go to
the smaller. If that size is nano, (2) does not run and the sweep is
the result. `vm_rung2y.py --detector yolo26{s,m}` writes the
`vision-mode-r2y-26{s,m}-{host}` family.

**Readings, on the 200.** DETECTOR FLOOR — misses ≤ 2 at s or m: the
nano was the limit, and the switch's remaining miss cost is bought by
a bigger lookout. FRAME FLOOR — ≥ 4 misses persist at m: the rule's
margin, not the detector. **Predictions.** (a) s finds 2 of the 5, m
finds 3 (the 2.8% box grows past 5% at a bigger detector only if the
box was clipped, so it likely stays a miss); false alarms stay at 2
(the cat and the unannotated woman are not nano's mistakes). (b) On
the 3,472, misses fall from 25 to ≤ 15 at m; false alarms do not fall
(label noise, not detector noise). (c) Rung 2y at the chosen size:
fidelity within −0.03 of zero at both hosts, every seed; the
false-alarm cell reproduces (≤ 0.05); ROBUST 3 of 3. Artifacts
`findings/vision-mode-r2y-detectors.{json,md}`;
`findings/vision-mode-r2y-26{s,m}-{450m,3b}{,-s*}.*`.

## Rung 2b registration — boxes on the bus (2026-08-23, Franci: "register properly"; before anything runs)

**The question.** Rungs 2 and 2y are a SWITCH: the small model says
one bit and that bit picks one of the operator's pre-trained packets;
the policy (present → the mode's packet, absent → explore's) is
hand-written. Rung 2b puts the small model's OUTPUT on the bus: the
detector's boxes enter the door, and the packet is computed from
(mode, boxes) by a bridge trained end-to-end on the host's action — no
policy in code, no lookup. Two things are at stake. Does the emitter
learn to say more than one bit usefully — a better or more faithful
action than the switch? And does the box channel stay grounded — when
the boxes LIE (a person's boxes pasted onto an empty frame), is the
host still the authority on the frame, as it was against the switch's
false alarms?

**The box vector.** Per frame, from `yolo26n` at the registered rule:
b = [present (0/1 by the rule), min(n person boxes, 5)/5, largest box
fraction, best confidence, cx, cy, w, h of the largest person box
normalized to the frame] ∈ R^8; all zeros when there is no person box.

**The emitter** (amended before the run, same day: the door is frozen,
so a gain cannot be bought by extra door steps). Rung 1's mode tap —
`features[(mode, position)]`, the host's hidden state at the source
layer for the mode's phrasing — PLUS a learned projection
`Linear(8 → hidden)` of b, zero-initialized so step 0 IS rung 1's door;
the sum goes through rung 1's mode door (loaded, sha256-checked,
FROZEN: BUS 256, conditioned) to the three inject layers at dose 0.15.
Only the projection trains (8 × hidden + hidden parameters), with rung
1's numbers (192 steps × 8, AdamW lr 3e-3, clip 1.0, rung 1's 80
training frames, all three modes, the tap at a training phrasing, the
target = the ORACLE ACTION of (mode, label) — so "social + no boxes →
proceed" is learned, not written), one projection per host and door
seed, saved as `checkpoints/experiments/vm-r2b-box-projection-{host}-s*.pt`.
The MODE-ONLY control is then exactly rung 1's mode door (b ≡ 0 → the
projection contributes nothing), loaded, not retrained. Gate: the
held-out phrasings (positions 8–9) with the frames' real boxes, token
accuracy ≥ 0.90 pooled, as rung 1's.

**Arms, per person-mode M in {social, caution}, on rung 2's 200 frames,
held-out phrasing at the frame's parity, greedy, MAX_NEW 24.** (1)
BOXES — bridge(M, b(frame)). (2) SWITCH — the detector's verdict picks
rung 1's packet (rung 2y's classifier arm, decoded here on the same
path for a like-for-like). (3) LYING BOXES — bridge(M, b') where b' on
each ABSENT frame is the box vector of a present frame drawn by the
fixed rng, and on each PRESENT frame is zeros (the lookout lies in both
directions through the channel; the switch's FLIPPED arm's analogue).
(4) HABIT — the constant packet. **Metrics.** Present-frame share of
M's action per arm; GAIN = share(boxes) − share(switch); the LIE cell =
M's action share on ABSENT frames under lying boxes; the SILENCE cost
= share(boxes) − share of M's action on PRESENT frames under zeros
(what a lying "nobody" takes away); grounding of the BOXES arm on
absent frames (b is zeros there by construction; recorded).
**Readings, per host by seed majority.** (i) BOXES HELP — gain ≥ +0.10
for both M AND lie cell ≤ 0.20: the channel carries something the bit
did not and the host stays the authority. (ii) CHANNEL OWNS — lie cell
> 0.50 for either M: the small model's lie through the bus captures
the host — the failure rung 2 showed the switch does NOT have. (iii)
ONE BIT — |gain| ≤ 0.10 for both M and lie cell ≤ 0.20: boxes on the
bus add nothing the verdict did not; the switch is the architecture.
(iv) MIXED, with the numbers.

**Predictions.** (a) Training converges as rung 1's door did (held-out
phrasing gate ≥ 0.90 at every seed). (b) 3B reads (iii) ONE BIT: the
action needs one bit and the host keeps the facts; lie cell ≤ 0.10.
(c) 450M reads (iii) or a small (i) on `social` only (the weak door,
where a richer input could buy lean); lie cell ≤ 0.20. (d) The
SILENCE cost equals the switch's miss cost within 0.10 — a lying
"nobody" and a missed person are the same silence. A (ii) at either
host would be the V-line's first capture of the big model by the
small: the switch's grounding was the LOOKUP's property (a false alarm
sends a packet the host was trained to ground), not the bus's. Hosts
450M and 3B; 27B named, not registered. Harness `vm_rung2b.py`;
artifacts `findings/vision-mode-r2b-{450m,3b}{,-s*}.*`.

**Amendment 2 (2026-08-23 12:45 PDT, after attempt 1 at the 450M, before
any 3B run).** Attempt 1 added the projection to the RAW tap with rung
1's lr 3e-3. The door's fitted source scale is 0.06 per dimension; the
zero-initialized projection drifted to norm 4.9 — a shift ~2.5× the
whole tap spread — and the training loss ROSE (0.013 at step 1, the
door alone, to 0.15 by step 145): the objective is at its floor before
training (rung 1 trained the door on exactly this target), so Adam
only wandered, and on eval both the BOXES arm and the zeros arm emitted
"proceed" on every frame (450M s42: gain −0.52 / −0.95, lie cell 0.00).
That is a harness artefact, not a reading; the attempt-1 450M artifacts
were overwritten by the rerun. Amended: the projection adds in the
door's NORMALIZED source space (`normalize_source`, unit RMS per
dimension, so the shift's scale is interpretable), lr 3e-4, and the
artifact records step 1 (the door alone) as the baseline. The prediction
stands as written; the honest expectation from attempt 1 is that the
projection has nothing to learn on frames the host can see, so (iii)
ONE BIT with a near-zero projection norm. Named, not registered: boxes
on the bus where the host is BLIND (a blank or occluded frame for the
big model, the lookout seeing the real one) — the only arrangement in
which the boxes carry novel information.

## Rung 1 + rung 2y at the 27B VL host result (2026-08-23 10:43–12:35 PDT, smarty CUDA, the seven-service block) — rung 1 (i) LEAN 3 of 3 with EVERY mode alive and the packet above the prompt sentence; rung 2y (i) ROBUST 3 of 3; the host corrects the lookout on the cat and agrees with it on the woman COCO forgot

**The host held the harness.** Qwen3.8-27B streamed to CUDA in under a
minute from page cache (51 GiB allocated, 70 GiB peak through training),
280 tokens a frame, zero parse fallbacks in every cell of both rungs;
the held-out phrasing gate read **1.00** at every seed and zero-dose
was exact. Rung 1: 85 min for three seeds; rung 2y: 19 min. The 27B
service was down 1 h 52 min.

**Rung 1 — (i) LEAN 3 of 3, decisive.** Pooled present-frame share of
the mode's action, packet / shuffled / habit / text: `social` greet
**0.98** / 0.00 / 0.33 / 0.87; `caution` stop **1.00** / 0.00 / 0.13 /
0.61; `explore` proceed **1.00** / 0.01 / 0.53 / 0.67. Leans 0.64 / 0.87
/ 0.47 against text leans 0.53 / 0.48 / 0.13 — **the packet is above
the instruction sentence in every mode**, the first host where that
is true for all three (3B: social dead; 450M: mixed). The swap control
is clean (the next mode's packet gives the action 0.00–0.01). Grounding
0.00 / 0.04 (caution 0.13 at s2042, the one door with a pulse on empty
frames). By seed, social lean 0.43 / 0.53 / 0.97, caution 0.77 / 0.83 /
1.00; explore at s2042 reads 0.00 for packet AND text because the habit
already proceeds on every person frame there — nothing to lean.
Predictions (a) gate, (b) LEAN with social alive at ≥ 2 seeds (3 of
3), (c) grounding clean: all held. The 3B's dead social door was the
small host's weak channel, not the door's.

**Rung 2y — (i) ROBUST 3 of 3.** Oracle / classifier / flipped /
habit: `social` 0.94 / 0.90 / 0.03 / 0.36; `caution` 0.97 / 0.92 / 0.05
/ 0.12. Fidelity **−0.03 / −0.05** (= the detector's 5% misses × a miss
cost of 0.90 / 0.92 — at this host a missed person costs the whole
mode, because the door delivers the whole mode). False-alarm cell
0.00 / **0.07**: social clean at every seed; caution 0.02 / 0.05 /
**0.14** by seed — s2042 again, the door that grounds 0.13 in rung 1.
Prediction (f) asked ≤ 0.05 pooled and caution missed it by 0.02 on
that one door; (d) ROBUST 3 of 3 and (e) |fidelity| ≤ 0.05 held (the
worst row is exactly 0.05).

**The real errors, frame by frame.** On the detector's five misses the
mode fires **0.00** (15 rows): the 27B obeys the silence completely, as
the 3B did. On its two "false alarms" the mode fires 0.50 / 0.50 —
and it is the same split at every seed and both modes: on 261706 (the
cat with the TV remote) the host says **proceed** every time, overruling
the lookout; on 509699 (the woman in the armchair COCO never annotated)
it says **greet** / **stop** every time, agreeing with the lookout
against the label. The big model is the authority on the frame in both
directions: it corrects the small model's one real mistake and backs
its one correct call that the dataset got wrong.

**What it means.** Three hosts, one architecture: the small model sets
the mode, the big model keeps the facts, and the bus is the seam — and
at 27B the seam carries the mode better than the prompt does. In plain
terms: the 27B-class robot with a 2-ms lookout greets or stops for
people 90–92% of the time under the lookout's word (the oracle: 94–97%),
never acts on a person it was told about but cannot see, and when the
lookout cries "person" at a cat, it looks, shrugs, and keeps going.
Artifacts `findings/vision-mode-r1-27b{,-s*}.*`,
`findings/vision-mode-r2y-27b{,-s*}.*`; doors
`checkpoints/experiments/vm-r1-{mode,constant}-bridge-27b-s*.pt` (smarty,
sha256 in the artifacts). Named, not registered: rung 1b at 27B (the
hostile prompt against a door that beats the prompt); the ×1.5 dose at
27B (GF-H2 says it is free there); a stronger s2042 caution door.

## Rung 1b-y result (2026-08-23 12:33–12:38 PDT, smarty CUDA, stopped nothing) — the detector-chosen mode reproduces the oracle's row to two decimals at both hosts

The detector made **1 error** on rung 1's 60 eval frames (0 misses, 1
false alarm — the unannotated woman, 509699). With no misses the policy
is the oracle's on every person frame, and the tables show it: at the
450M, packet / packet+hostile / hostile / habit for `social` 0.63 /
0.56 / 0.29 / 0.30 and `caution` 1.00 / 0.99 / 0.18 / 0.22, ownership
**0.80 / 1.04 = the oracle's 0.80 / 1.04**; at the 3B `caution` 0.76 /
0.41 / 0.00 / 0.09 and `explore` 1.00 / 0.92 / 0.06 / 0.44, ownership
**0.62 / 1.56 = the oracle's 0.62 / 1.56**, the same asymmetry (explore
holds outright, caution splits by door strength) seed for seed (3B
caution 0.00 / 0.48 / 0.83). Grounding is cleaner than the oracle's
(3B 0.01 / 0.00 against 0.04 / 0.02) because the detector keeps 29 of
30 empty frames on explore's packet. Predictions (a) ≤ 2 errors (1),
(b) within 0.05 of the oracle (0.00), (c) admissible by construction
(every row, zero fallbacks), (d) 3B explore ≥ 1.0 and caution 0.55–0.70
(1.56, 0.62): all held. The emitter is not the variable in rung 1b; the
door is. Artifacts `findings/vision-mode-r1b-{450m,3b}-x1-detector{,-s*}.*`.

## Rung 2y-size result (2026-08-23 12:32–12:48 PDT, smarty CUDA, stopped nothing) — DETECTOR FLOOR: s and m find all five; with yolo26m the switch's fidelity is 0.00 at every seed of both hosts

**The sweep, no host.** On the 200 eval frames: `yolo26n` 0.965 (2 FA,
5 misses), `yolo26s` **0.995** (1 FA, **0** misses), `yolo26m` 0.990 (2
FA, 0 misses); on the 3,472: misses 25 → 12 → **10**, false alarms 25 →
22 → 22; 3.5 / 3.6 / 4.4 ms a frame. Reading **DETECTOR FLOOR**: every
one of the nano's five misses is a confident, large person box at s and
m (0.34–0.90, 11–97% of the frame) — the nano simply did not see them.
The two false alarms are not detector noise: the unannotated woman is
"present" at every size (the label is wrong), and the cat with the
remote is a person at n (0.41) and m (0.47) but not at s — the one
frame where the sizes disagree. Prediction (a) was pessimistic ("s
finds 2 of 5"): both find all five; false alarms stayed at 1–2 as
predicted; (b) calibration misses ≤ 15 at m: 10. By the registered
rule (fewest calibration misses, ties to the smaller) rung 2y reran
with **yolo26m**.

**Rung 2y with yolo26m — ROBUST 3 of 3 at both hosts, fidelity 0.00
everywhere.** 450M: `social` 0.62 / 0.62 / 0.14 / 0.30, `caution` 0.97 /
0.97 / 0.26 / 0.22, fidelity **0.00 / 0.00**, false-alarm cell 0.02 /
0.01. 3B: `social` 0.46 / 0.46 / 0.00 / 0.53, `caution` 0.74 / 0.74 /
0.00 / 0.07, fidelity **0.00 / 0.00**, false-alarm cell **0.02 / 0.04 —
the 3B's number under its third small model** (MobileNet head, nano,
medium: 0.02 / 0.04 each time). With zero misses the classifier arm IS
the oracle arm on person frames, and the two false-alarm frames are
judged by the host exactly as before (the woman gets the action, the
cat does not). Prediction (c) held in full.

**What it means.** The miss-rate × miss-cost bound that explained every
fidelity gap from rung 2 onward closes to zero when the lookout stops
missing: rung 2's −0.04…−0.08, rung 2y's −0.01…−0.04, now 0.00, with
the same doors and the same frames. A 20M-parameter detector at 4 ms a
frame buys the oracle's behaviour outright. In plain terms: hire a
lookout who never misses a person and the captain behaves exactly as
if they had the crew manifest — and still ignores the lookout about
the cat. Artifacts `findings/vision-mode-r2y-detectors.*`,
`findings/vision-mode-r2y-26m-{450m,3b}{,-s*}.*`. Named, not
registered: yolo26s as the operating detector (no misses, one fewer
false alarm than m, 9.5M params) — the medium won on the registered
rule by two calibration misses, not on the 200.

## Rung 2b result (2026-08-23 12:48–13:00 PDT, smarty CUDA, stopped nothing; attempt 1 12:38–12:42, diverged, superseded) — the switch is the architecture: ONE BIT everywhere the door is strong, and the learned box channel's only real effect is to top up a WEAK door; the host ignores lying boxes outright

**Training behaved under amendment 2.** The projection stayed small
(weight norm 0.27–1.5 against attempt 1's 4.9 drift), the door-alone
loss (step 1) already sat at 0.0001–0.08 and the final loss matched it
— the objective is at its floor before the boxes arrive, which is the
result in miniature. Gates 0.95–0.99, all passed.

**450M — pooled (iii) ONE BIT** (seeds: MIXED / MIXED / ONE BIT; the
MIXED rows are the boxes COSTING the weak social door up to −0.29).
Pooled boxes / switch / lying / habit: `social` 0.53 / 0.59 / 0.54 /
0.30 (gain −0.06), `caution` 0.92 / 0.93 / 0.97 / 0.22 (gain −0.01).
Lie cell 0.05 / 0.03; the BOXES arm on empty frames 0.02.

**3B — pooled (iv) MIXED, and the mix is the finding.** `social` gain
+0.01 (one bit). `caution` boxes 0.93 vs switch **0.70**, gain
**+0.22** pooled — all of it from s42, the weakest caution door in the
V-line (rung 2y oracle lean 0.22 there): switch 0.26, boxes **0.83**,
gain +0.57 at that seed; at the strong doors s1042 / s2042 the gain is
+0.05. Per-seed readings ONE BIT / ONE BIT / MIXED(s42), majority ONE
BIT. **The channel never owns the host**: lie cell ≤ 0.06 at every
seed of both hosts — a person's boxes pasted onto an empty frame move
the big model at most 6% (the LYING arm's absent rows), and zeros on a
person frame just silence the mode (silence cost ≈ the miss cost, as
predicted, +0.17 pooled at the 3B's caution).

**What it means.** Prediction (b)/(c) held in the letter (ONE BIT
majorities, lie cells under every band) with one honest surprise: a
learned projection from detector boxes can COMPENSATE FOR A WEAK DOOR
(+0.57 stop at the 3B's s42) — the box channel is a door-strength
top-up, not an information channel; where the door already delivers
the mode (0.9+), the boxes have nothing to add, and where the boxes
lie, the host looks at the frame and ignores them. The switch (rungs
2/2y) remains the architecture: same behaviour, no training, no new
capture surface. Named, not registered: boxes on the bus where the
host is BLIND (the lookout sees a frame the big model does not get) —
the one arrangement in which the boxes carry novel information; the
weak-door top-up as a deliberate repair (train the projection only
where rung 1's lean is under 0.5). Artifacts
`findings/vision-mode-r2b-{450m,3b}{,-s*}.*`; projections
`checkpoints/experiments/vm-r2b-box-projection-*` (smarty).

## Rung 1b at 27B registration — the hostile prompt against a door that beats the prompt, at ×1 and ×1.5 (2026-08-23, Franci: "register the 27B rung 1b and the rest … and let's take a stab at it"; before anything runs)

**The question.** At the 27B the packet carries every mode ABOVE the
prompt sentence (rung 1: leans 0.64 / 0.87 / 0.47 vs text 0.53 / 0.48 /
0.13). Rung 1b's contest — the client's contrary order in the context
against the operator's mode on the bus — has so far been fought by
doors at or below text strength (3B: caution split 0.62 by door
strength; ×1.5 bought dominance and lost the frame). What does the
contest look like when the door is the stronger channel to begin with;
and is ×1.5 free here, as GF-H2 found for this host's text-side doors
(dominance at ×1.5, the errand intact), or does the VL side pay the
3B's price?

**Design.** `vm_rung1b.py --host 27b`, oracle emitter (rung 1b-y showed
the emitter is not the variable at two hosts; the detector row is not
rerun), doses ×1 and ×1.5, three door seeds, rung 1's 27B doors
sha-recorded, the same cells, bands, admissibility rule and readings as
the 2026-08-22 rung 1b registration. One GPU block down to the 27B
service for both doses.

**Predictions.** (a) ×1 admissible 3 of 3 (the 27B had zero parse
fallbacks in every cell so far). (b) ×1 ownership: `caution` ≥ 0.80
pooled (its door leans 0.77–1.00 vs the explore text's weak 0.13 teeth)
and `explore` ≥ 0.90 pooled against the caution text (text teeth 0.48);
`social` recorded (its hostile is the weak explore text — a weak
adversary, as at the 450M). (c) If the hostile text hallucinates on
empty frames, the packet repairs it (the 3B did 0.33 → 0.99).
(d) ×1.5: admissible AND ownership rises toward 1.0 with grounding
≤ 0.20 — "×1.5 is free at 27B" carries over from GF-H2's text side. A
3B-style frame loss at ×1.5 would split the dose window by modality
and is the registered alternative. Artifacts
`findings/vision-mode-r1b-27b-x{1,1p5}-oracle{,-s*}.*`.

## Stronger-door registration — the weak seed's doors at 2× steps (3B s42) (2026-08-23, Franci: "register … the rest"; before anything runs)

**The question.** Rung 1b's caution split ordered exactly by door
strength (3B s42 / s1042 / s2042 ownership 0.00 / 0.48 / 0.83 at leans
0.13 / 0.90 / 0.97), and the ×1.5 dose bought ownership only by losing
the frame. The named claim: DOOR STRENGTH is the lever that costs
nothing. Test it where it was weakest.

**Design.** `vm_door2x.py`: retrain the 3B s42 door PAIR (mode and
constant, the same recipe, rng and frames as rung 1, steps 192 → 384)
and save as `checkpoints/experiments/vm-r1x2-{mode,constant}-bridge-3b-s42.pt`
— the registered doors are not overwritten. Gates as rung 1 (held-out
≥ 0.90, zero-dose exact). Then `vm_rung1b.py --host 3b --seed 42
--dose-scale 1 --door-variant x2` (the variant loads the x2 stem and
writes the `-x2door` artifact family). The baseline is the registered
s42 ×1 oracle row: caution ownership 0.00, packet share 0.26.

**Readings.** (i) STRONGER DOOR WINS, FREE — s42 caution ownership
≥ 0.48 (overtakes the registered middle seed) AND admissible AND
grounding within +0.05 of the registered s42 row: strength buys what
the dose could not pay for. (ii) DOSE IN DISGUISE — ownership rises but
grounding > 0.20 or fallbacks > 0.05: more door is just more dose.
(iii) FLAT — ownership < 0.25: the s42 weakness is not steps.
**Predictions.** The mode door's caution packet share rises 0.26 →
≥ 0.60 and the reading is (i), with social also lifting (its s42 lean
was 0.16). Artifacts `findings/vision-mode-r1b-3b-x1-oracle-x2door{,-s42}.*`;
the door training record in the checkpoint and the artifact.

## Rung 2y-26s registration — the operating detector (2026-08-23, Franci: "register … the rest"; before anything runs)

**The question.** The size sweep's registered rule (fewest calibration
misses) chose `yolo26m` by two misses on the 3,472; on the 200 the
small was the better instrument (0 misses, 1 false alarm to the m's 2,
9.5M params, 3.6 ms). Does `yolo26s` reproduce the m's rung-2y rows —
and if so, the V-line's OPERATING detector becomes the s.

**Design.** `vm_rung2y.py --detector yolo26s`, 450M and 3B, three door
seeds (the `vision-mode-r2y-26s` family). **Decision rule, registered
now:** if fidelity is 0.00 at every seed of both hosts AND each host's
pooled false-alarm cell is ≤ its 26m value (450M 0.02 / 0.01, 3B 0.02 /
0.04), the operating detector for later rungs is `yolo26s`; otherwise
`yolo26m` stands. **Prediction:** it reproduces (only the cat frame
261706 leaves the flipped accounting — the s is the one size that does
not call the cat a person) and the s takes the post. Artifacts
`findings/vision-mode-r2y-26s-{450m,3b}{,-s*}.*`.

## Rung 2c registration — the blind captain: boxes on the bus where the host cannot see (2026-08-23, Franci: "register … the rest"; before anything runs)

**The question.** Every grounding result in the V-line — false alarms
nearly free, lying boxes ignored, the prompt's ghosts repaired — rests
on the host SEEING the frame. Rung 2b showed boxes add nothing a
sighted host needs. Blind the host and the boxes become the only eyes:
does the fact "a person is present" cross the bus at all; how much of
it; and does the lying-boxes cell now CAPTURE, confirming that rung
2's safety was the eyes and not some property of the bus?

**Design.** The host's image input is a REGISTERED BLANK — a uniform
black 640×480 PNG generated at run time (sha256 in the artifact) and
used for EVERY training and eval row; the detector (`yolo26n`, as rung
2b, for comparability) sees the REAL frames and supplies verdicts and
box vectors. Everything else is rung 2b amendment 2's harness: the
frozen s-seed mode door, the zero-initialized projection in the door's
normalized source space, lr 3e-4, 192 × 8 on rung 1's 80 train frames
(blinded), oracle-action targets, gates on blinded held-out phrasings;
arms BOXES / SWITCH / LYING / HABIT on rung 2's 200 eval frames
(blinded), per person-mode, three seeds, hosts 450M and 3B. Harness
`vm_rung2c.py`; projections `vm-r2c-box-projection-*`.

**Metrics.** Rung 2b's, plus DELIVERY per arm = the person-mode's
action share on present frames MINUS on absent frames — the bus's
discrimination with no eyes to check it. **Readings, per host by seed
majority.** (i) BUS CARRIES THE FACT — switch delivery ≥ 0.50 for both
person-modes: the lookout's one bit crosses the bus and moves the
blind host differentially. (ii) BUS TOO WEAK — switch delivery < 0.25
for either mode. (iii) MIXED otherwise. CAPTURE is recorded alongside,
not read: lying absent-share ≥ 0.50 for a mode means the blind host
obeys the lie — expected, and the confirmation that sight, not the
bus, was rung 2's safety. **Predictions.** (a) 3B: (i), delivery
≥ 0.50 both modes (the packet moved the sighted 3B at 0.7–1.0; blind,
nothing opposes it); 450M social may miss the band (the weak door).
(b) CAPTURE confirmed at both hosts (lying absent-share ≥ 0.50 where
the door is strong). (c) BOXES ≈ SWITCH (|gain| ≤ 0.15): the bit is
what crosses; geometry has no consumer in a one-bit task. A blind
host that still refuses the lie would be the surprise worth chasing.
Artifacts `findings/vision-mode-r2c-{450m,3b}{,-s*}.*`.

## Rung 2y-26s result (2026-08-23 13:22–13:27 PDT, smarty CUDA, stopped nothing) — the decision rule fires: yolo26s is the V-line's OPERATING DETECTOR

ROBUST 3 of 3 at both hosts, fidelity **0.00 at every seed** (1 false
alarm, 0 misses on the 200 — the woman COCO forgot; the s is the one
size that does not call the cat a person), and the pooled false-alarm
cells are IDENTICAL to the 26m run's at both hosts and both modes
(450M 0.02 / 0.01, 3B 0.02 / 0.04) — identical because the one frame
the sizes disagree on is the cat, and the host ignores the packet on
that frame under either verdict. By the registered rule the operating
detector for later rungs is **yolo26s** (9.5M params, 3.6 ms); the
medium remains the calibration-rule winner, recorded. Artifacts
`findings/vision-mode-r2y-26s-{450m,3b}{,-s*}.*`.

## Stronger-door result (2026-08-23 13:27–13:32 PDT, smarty CUDA, stopped nothing) — (i) STRONGER DOOR WINS, FREE: 2× steps takes s42's caution ownership from 0.00 to 1.47 with grounding at 0.00

The x2 pair trained in 184 s (gates 0.99, zero-dose exact). Against
the registered s42 row: caution packet share **0.27 → 1.00**,
packet+hostile **0.00 → 0.73**, ownership **0.00 → 1.47** — past every
registered seed (0.48, 0.83) — with zero parse fallbacks and grounding
**0.00** (the ×1.5 dose bought 0.84 ownership at the price of the
frame; 2× steps bought more for nothing). Explore holds (1.11).
`social` did not lift: packet share 0.50 at both strengths — the greet
channel's ceiling at this host is not steps either. Reading **(i)
STRONGER DOOR WINS, FREE**, prediction confirmed: door strength, not
dose, is the lever that costs nothing. Named, not registered: the x2
recipe at the other seeds and hosts as the standard door; steps × dose
as a 2-axis map. Artifacts
`findings/vision-mode-r1b-3b-x1-oracle-x2door{,-s42}.*`,
`findings/vision-mode-x2door-3b-s42.json`; doors
`checkpoints/experiments/vm-r1x2-*-3b-s42.pt` (smarty).

## Rung 2c result (2026-08-23 13:32–13:43 PDT, smarty CUDA, stopped nothing) — (ii) BUS TOO WEAK 3 of 3 at both hosts, with the exception that proves both rules: only the V-line's strongest door reaches a blind host, and there the lie owns it completely

**The headline.** Blind the host and the switch delivers NOTHING at
the trained dose: under every rung-1 packet the hosts answer `proceed`
on the blank frame (450M: proceed 1.00 under the caution packet that
moves the sighted host to 0.97 stop). Rung 1's doors are LEANS — "if a
person, lean stop" — and with no frame support the blank-frame prior
wins. The trained blind projection learns under teacher forcing (loss
→ 0.01–0.02 at the strong-door seeds) and still cannot flip the
generated action (450M s42 the only partial: boxes delivery 0.26 /
0.48 with the lie mirroring it at 0.22 / 0.49 — a fact-injector is
direction-blind by construction).

**The exception.** The 3B's s2042 caution door — the strongest in the
V-line (rung 1 lean 0.97; the only door with a grounding pulse even
sighted) — DOES cross: switch delivery **0.93** on the blind host, and
on the same mode the lying boxes capture **1.00**. Exactly the
registration's claim, confirmed at the only door strong enough to test
it: what crosses the bus without eyes is obeyed without question —
SIGHT, not the bus, was every grounding result in rungs 1–2y.

**What it means.** The architecture's safety property is now located
precisely: the mode packet cannot make a host hallucinate (blind or
sighted) because it carries a conditional lean, not a fact — but a
door strong enough to act as a command channel (s2042-caution-class)
turns a blind host into the lookout's puppet. Deployment rule that
falls out: doors below command strength for hosts that may lose their
eyes; grounding lives in the host's own perception, keep it sighted.
Prediction (a) was wrong in the interesting direction (delivery ≥ 0.5
expected at the 3B: only one door of six host-modes crossed); (b)
CAPTURE confirmed where anything crossed; (c) BOXES ≈ SWITCH held
except 450M s42. Named, not registered: the blind captain × dose
(does ×1.5–×2 turn leans into commands?); the x2 door blind (strength
axis); an asserting door (trained to carry the fact deliberately).
Artifacts `findings/vision-mode-r2c-{450m,3b}{,-s*}.*`.

## Rung 1b at 27B result (2026-08-23 13:43–14:19 PDT, smarty CUDA, the seven-service block, 27B down 36 min) — (i) PACKET HOLDS at BOTH doses, ×1.5 is FREE at 27B, and the host needs no repair: its grounding under the hostile text is already perfect

**×1 — (i) PACKET HOLDS, admissible 3 of 3, zero parse fallbacks in
every cell.** Pooled ownership `social` 0.84 (packet 0.98 →
packet+hostile 0.77; the hostile alone 0.22), `caution` **0.90** (1.00
→ 0.89; hostile 0.11), `explore` **1.48** (1.00 → 0.98; the caution
text alone drives proceed to 0.29 — teeth — and the packet shrugs it
off). Predictions (a) and (b) held (caution ≥ 0.80, explore ≥ 0.90;
social recorded at 0.84 against a hostile with real teeth here, unlike
the smaller hosts). By seed: s42 1.46 / 1.17 / 1.08, s1042 1.25 / 0.92
/ 0.83, s2042 the odd one again — social 0.34, caution 0.67 (the same
door whose grounding pulses 0.13; its strength leaks as
suggestibility). **Prediction (c) is moot in the best way:** the
hostile text alone does NOT move the 27B on empty frames (wrong action
0.00; proceed 1.00 under the caution text where the 3B refused 0.67) —
there is nothing to repair; the biggest host grounds its own prompt.

**×1.5 — (i) PACKET HOLDS, admissible 3 of 3: THE DOSE IS FREE AT
27B.** Ownership 0.91 / 0.93 / 0.89 pooled, grounding 0.00 / 0.03,
zero fallbacks; even s2042 lifts (social 0.34 → 0.70, caution 0.67 →
0.73) at no grounding cost. Prediction (d) held: GF-H2's text-side
freedom at ×1.5 carries to the VL side. The dose window splits by
HOST, not modality: 3B ×1.5 loses the frame, 27B ×1.5 is free — scale
buys the safety margin that lets the dial move.

**What it means.** At the 27B the operator's mode owns 84–98% of the
contested behaviour against a direct contrary order, at either dose,
while the model keeps perfect grip on empty frames — and the s2042
oddity plus the stronger-door result point the same way: what varies
by seed is door TEMPERAMENT (strength vs suggestibility), and steps,
not dose, is the knob that buys strength for free. In plain terms:
tell the biggest robot "ignore people, keep moving" while its mode
says "stop for people," and the mode wins ~9 times in 10; turn the
mode up by half and it wins a bit more, still without ever stopping
for a ghost. Named, not registered: the x2 door at 27B s2042 (does
strength cure suggestibility?); the ×2 point at 27B on the V-line.
Artifacts `findings/vision-mode-r1b-27b-x{1,1p5}-oracle{,-s*}.*`.

## Rung 2t registration — the drowsy captain: the lookout gates the big model's compute (2026-08-23, Franci: "save computations on the large model unless the small model detects change that requires the big model to reevaluate"; before anything runs)

**The question.** The big model decodes every frame; the lookout costs
4 ms. If the big model only re-evaluates when the LOOKOUT says the
world changed — and its last decision stands otherwise — how much
compute is saved, and what does staleness cost? The interesting limit
is known in advance: a gate can only preserve decisions as far as the
world, SEEN THROUGH THE GATE, determines them; whatever the big model
decides from frame content the gate cannot see is lost while the gate
sleeps.

**Episodes (the registered draw).** 60 episodes × 16 steps from rung
2's 200 eval frames, one fixed rng (seed 2026), identical for every
host and door seed. Step 0 is a cut; each later step CUTS to a
uniformly drawn pool frame with p = 0.2, else STAYS on the current
frame. EVERY step (cuts included) is re-rendered with camera jitter: a
random 94–100% crop resized back plus brightness ×U(0.96, 1.04) — so
naive pixel equality is dead and a stay-step is never the same file.
Uniform cut draws mean about half the cuts keep the presence label:
those same-label cuts are the staleness probe (the world changed, the
verdict did not). Frames are pre-rendered deterministically; the
manifest sha256 is in the artifact.

**The system under test** is rung 2y's switch with a latch: at a WAKE
the detector's verdict (yolo26s, the operating detector) picks the
packet (person-mode M or explore), the big model decodes the current
jittered frame with it (greedy, MAX_NEW 24, held-out phrasing at the
episode's parity, rung 1's doors per seed), and the action is LATCHED
until the next wake. Between wakes only the lookout runs.

**Arms (the gates), per person-mode M in {social, caution}, per host
(450M, 3B), per door seed.** (1) ALWAYS — wake every step: the
baseline the others are judged against. (2) NEVER — wake at step 0
only: the floor. (3) VERDICT — wake when the detector's verdict
differs from the verdict AT THE LAST WAKE. (4) BOX — wake when, vs
the last wake, the verdict changed OR the person count (clipped at 5)
changed OR |Δ largest-box fraction| > 0.10 OR the largest box's
centre moved > 0.15. (5) PIXEL — the understanding-free control: wake
when the mean absolute pixel difference to the PREVIOUS STEP exceeds
0.04 of full scale. (6) CUT-ORACLE — wake exactly on true cuts: the
ceiling; its residual disagreement with ALWAYS is the big model's own
jitter-instability, not staleness, and no gate can beat it.

**Metrics, per arm and mode.** AGREEMENT = share of steps whose
latched action equals ALWAYS's action that step; WAKE FRACTION =
wakes / steps (compute saved = 1 − it; lookout cost reported beside
it); ORACLE CORRECTNESS = share of steps whose action matches the
frame label's oracle action (absolute quality — staleness can
accidentally help); LAG = mean steps from a true cut to the next wake;
FALSE WAKES = wakes with no cut since the last wake. **Readings, per
host by seed majority, on the VERDICT gate.** (i) FREE LUNCH —
agreement ≥ min(0.95, cut-oracle agreement − 0.02) AND wake fraction
≤ 0.40 for both M. (ii) STALE — agreement < 0.85 for either M. (iii)
JITTERY — wake fraction > 0.70. (iv) MIXED. The other gates are
recorded against the same bands.

**Predictions.** (a) VERDICT saves 65–75% of big-model calls with
near-zero false wakes (the rule's 5% area threshold may flicker under
jitter on borderline boxes — recorded). (b) `caution` reads (i) at the
450M (within-verdict variance is small: present-share 0.93–0.97);
at the 3B caution is borderline (0.74 present-share = the big model
disagrees with itself across person frames the gate cannot tell
apart). (c) `social` pays: agreement 0.80–0.88 (present-share ~0.5 =
half the person frames greet, half do not, and the gate sleeps through
the difference) — STALE-ish, and the honest headline if so: the gate
preserves only what presence determines. (d) PIXEL at 0.04 reads
JITTERY (the registered jitter alone exceeds it: noise and change are
indistinguishable without understanding). (e) BOX buys a few points of
agreement over VERDICT on same-label cuts for a few extra wakes. (f)
CUT-ORACLE agreement ~0.93–0.98 (the jitter-instability floor is not
zero: greedy decoding flips on some jittered re-renders). Named, not
registered: the 27B promotion (the block economics change — at 0.1 s a
decode the 450M/3B savings are real but small in seconds; at the 27B
a saved decode is a saved second); a hysteresis gate (consecutive
confirmations before waking); the gate driving DOSE (drowsy = packet
rides at lower dose). Harness `tracks/vision-mode/vm_rung2t.py`;
artifacts `findings/vision-mode-r2t-{450m,3b}{,-s*}.*`.

## Rung 2t result (2026-08-24 13:40–13:52 PDT, smarty CUDA, stopped nothing) — (i) FREE LUNCH at both hosts: the verdict gate saves 83.5% of the big model's decodes at 96–99% agreement, one step of lag, and no loss of grounding; the dumb pixel gate saves nothing, exactly as registered

**The headline.** VERDICT gate, pooled: wake fraction **0.165** (=
83.5% of big-model calls saved) with agreement to the every-frame
baseline of 0.97 / 0.99 (450M social / caution) and 0.96 / 0.99 (3B);
oracle correctness is the baseline's within a point (450M caution
0.970 vs ALWAYS's 0.978; 3B 0.869 vs 0.874). Mean lag from a true cut
to the wake: **0.98 steps**. False wakes: 30 of 5,760 gate-steps (the
predicted flicker of borderline boxes under jitter — 0.5%). Reading
(i) FREE LUNCH at both hosts, 2 of 3 seeds each (s1042 dips to 0.947
on social against the 0.95 band).

**The ladder of gates behaved exactly as registered.** NEVER (the
floor): 0.65–0.74 agreement — a latched first impression is wrong a
third of the time by episode's end. PIXEL (the understanding-free
control): wake fraction **0.947** — the registered jitter's median
frame-to-frame difference is 0.107, 2.7× the 0.04 threshold, so the
dumb gate cannot tell camera noise from change and saves nothing:
(iii) JITTERY, prediction (d) exact. CUT-ORACLE (the ceiling):
0.987–0.995 — the big model itself flips ~1% of its answers on a
jittered re-render of the SAME frame, so no gate can score above ~0.99;
VERDICT sits 1–3 points under the ceiling. BOX buys most of that
residual (0.99–1.00 agreement) for 11 points more wakes — prediction
(e). **Prediction (c) was too pessimistic:** social's within-verdict
variance costs only 1–3 points, not 10–15, because staleness windows
are short (lag ~1 step) — the gate's error is bounded by lag × flip
rate, and the lag is one step.

**The economics.** The lookout costs 7.5 ms a frame; a skipped
big-model decode saves 45 ms (450M) / 68–76 ms (3B) — the gate pays
for itself 6–10× per skipped call at toy scale, and the ratio scales
with the host (a 27B decode is ~1 s: ~130×). In plain terms: the
robot's big brain naps five frames out of six; the lookout wakes it
within one frame of anything actually changing; and nobody watching
its behaviour for 960 frames would notice more than a 1–4% difference
— while its compute bill drops by five sixths. Named, not registered:
the 27B promotion (where the seconds are real); a hysteresis gate;
drowsy dose (the packet at reduced dose while stale). Artifacts
`findings/vision-mode-r2t-{450m,3b}{,-s*}.*`; episode manifest sha
`2aa0665e…` (identical across hosts).

## Rung 3 registration — the rendered world: does the stack transfer to MuJoCo with zero retraining? (2026-08-24, Franci: "do the new paper for the V-line and simultaneously start the rung 3"; before anything runs)

**The question.** Every V-line result lives on COCO photographs. Rung 3
renders scenes instead — the robotics direction — and asks the transfer
question in two parts, nothing retrained anywhere. The architectural
stake, stated in advance: the DOORS cannot fail to transfer by
construction (a door's packet is computed from the mode phrasings'
taps, never from the frame), so what is actually under test is the two
PERCEIVING parts — the operating detector's verdict and the host's own
grounding — on rendered pixels.

**The scene.** A floored room with 2–4 colored box obstacles; the
person is DeepMind's classic humanoid (`assets/humanoid.xml`, pinned
from mujoco 3.3.0, sha256 `acc91675…`), pose-randomized (qpos jitter
around stand/walk keyframes) and placed by the rng within the camera's
view. PRESENT mirrors rung 1's rule in rendered form: the humanoid's
SEGMENTATION-MASK area (MuJoCo's segmentation render) must be ≥ 5% of
the frame — the mask-area rule again, now measured exactly instead of
annotated. ABSENT = the same drawn scene with the humanoid removed.
Draw: rng 2027, **100 present + 100 absent eval frames**, 640×480,
camera position/height/yaw jitter, light position jitter, EGL on
smarty (0.2 ms/frame; OSMesa fallback verified). **Realism ladder,
registered now:** L0 = the humanoid's stock appearance; L1 = skin- and
clothing-colored geoms (rgba only, no meshes). Both levels rendered
from the SAME draws; if the detector fails at both, that is the
reading — not a license to iterate appearance unregistered.

**Part 1 — the lookout on rendered pixels (no hosts).** yolo26s (the
operating detector) under the registered rule on the 200 frames at L0
and L1; confusion per level; the nano and medium recorded beside it
(the sweep machinery). Readings: (i) DETECTOR TRANSFERS — accuracy
≥ 0.90 at either level; (ii) SIM GAP — < 0.75 at both; (iii) MIXED.

**Part 2 — the doors on rendered frames (runs only if part 1 reads
≥ 0.75 at some level, else recorded as blocked).** Rung 1's eval arms
(packet / shuffled / habit / text, all three modes, grounding on
absent) at 450M and 3B with the UNCHANGED COCO-trained doors, all
three seeds, on the winning realism level's frames. Readings, per
host: (i) DOORS TRAVEL — ≥ 2 modes with lean ≥ 0.25 AND grounding
≤ 0.20 (the rung-1 bands on rendered pixels); (ii) HOST-BLIND —
every mode's lean < 0.10 while the same host's COCO lean was ≥ 0.25
(the host does not read the render as a person; the doors never got
their referent); (iii) MIXED. **Predictions.** (a) L0's pink humanoid
divides the detectors (yolo26s ≥ 0.75 but below 0.90); L1 clears (i)
at ≥ 0.90 — COCO detectors generalize to humanoid renders once colors
are plausible. (b) Part 2 at the 3B reads (i) with leans at 0.6–1.0×
their COCO values: the door is frame-blind, so transfer is exactly as
good as the host's perception of the render. (c) Grounding stays
≤ 0.10 — an empty rendered room is still an empty room. Harness
`tracks/vision-mode/vm_rung3.py` (implementation delegated to Sol per
Franci; this registration is the contract); artifacts
`findings/vision-mode-r3-detector.{json,md}` (part 1),
`findings/vision-mode-r3-{450m,3b}{,-s*}.*` (part 2); frames under
`/tmp/vm-r3-frames` with a manifest sha in every artifact. Named, not
registered: rung 3 at 27B; the gate (rung 2t) on rendered episode
streams; a moving-camera episode draw.

## Rung 3 part 1 result (2026-08-24, smarty CUDA EGL, stopped nothing) — (ii) SIM GAP, TOTAL, mechanism identified: geometric primitives are not people; the detector confidently parses the capsule humanoid as a still life

**The run.** 400 frames rendered (100 present + 100 absent × two
realism levels, one draw, manifest sha `96890fee…`, EGL at 0.2 ms a
frame; one deterministic escape hatch added when episode 21's camera
draw made the 5% mask unreachable by pose jitter alone — the placement
draw now walks toward the camera every 25 failures). Every present
frame satisfies the registered segmentation-mask ≥ 5% rule; every
absent frame masks 0.

**The reading.** yolo26n / s / m at L0 and L1: **0 person verdicts in
all 600 detector-level trials** (one exception: the nano found 1 of
100 at L0), zero false alarms — accuracy 0.50 across the board.
(ii) SIM GAP at both levels; by the registered gate, **part 2 does not
run** and is recorded as blocked.

**The mechanism (sub-threshold diagnostic, conf 0.001).** The gap is
not weak person-signal under the 0.25 rule — it is a confident WRONG
parse: on present frames the detector's best person confidence peaks
at **0.03**, while the same frames yield `sports ball` at **0.81** (the
head sphere), `orange` 0.75, `carrot`, `knife`, `baseball bat` (the
capsule limbs). The registered realism ladder (rgba recolor) was aimed
at the wrong axis: the deficit is SHAPE AND TEXTURE — capsule
primitives never enter the person region of COCO feature space,
whatever their color. Prediction (a) was wrong in the clean direction:
not "divides the detectors" but a total, mechanistic miss shared by
all three sizes.

**What it means.** The V-line's perceiving parts do not transfer to
geometric-primitive renders AT ALL — a crisp negative that locates the
sim-to-real bridge exactly where a mesh-level human belongs, and a
warning for anyone evaluating COCO detectors against primitive-based
simulators: zero false alarms and zero hits LOOK like a conservative
detector and are actually total blindness. The doors remain untested
on renders (blocked, not failed): by construction they are
frame-blind, so rung 3b needs only a render the PERCEIVING parts can
read. Named, not registered: **rung 3b** — L2 realism as a textured
human MESH (asset to be pinned at registration); a HOST-perception
probe (do the VL hosts read the capsule humanoid as a person where
YOLO does not? — their semantics are richer, and if yes, the lookout
is the sim bottleneck, not the captain); the photograph-billboard
control (a real-person texture on a quad — answers whether the gap is
the renderer or the primitives). Artifacts
`findings/vision-mode-r3-detector.{json,md}`; frames
`smarty:/tmp/vm-r3-frames` (manifest sha in the artifact).

## Rung 3b registration — the photoscan person: L2 realism as a textured human mesh, plus the capsule host-probe (2026-08-24, before anything runs)

**The asset, pinned.** HSR0081-Body-004 LOD2 from Digital Reality
Lab's HSRD-100 (huggingface.co/datasets/digitalrealitylab/HSRD-100):
a clothed-adult photogrammetry scan, 10,000 faces + 4096² diffuse
texture, real-world scale, CC BY 4.0 (consented subject; attribution
recorded in `tracks/vision-mode/assets/README.md`). Zip sha256
`71afed0b…`, downloaded at run time to the gitignored checkpoints tree
and sha-verified. Procurement evidence (Sol, 2026-08-24): the mesh
loads in MuJoCo 3.12, renders offscreen, and yolo26n reads the render
as `person` at 0.876 from 2.15 m — the asset clears the mechanism that
killed rung 3's capsules before the experiment runs. Limitation
recorded now: ONE identity and ONE static pose (yaw + placement vary);
rung 3b tests transfer mechanics, not detector generality — COCO
already established the detector.

**Design.** Rung 3's machinery unchanged — the SAME draw seed 2027,
rooms, cameras, lights, mask rule (segmentation ≥ 5%), so 3b frames
differ from rung 3's ONLY in the figure: `--figure mesh` replaces the
capsule humanoid with the pinned scan (yaw and placement from the same
draws; the pose-jitter draws still consumed, unused, to keep the rng
stream aligned). Part 1: the detector sweep at the mesh level
(readings as rung 3's: TRANSFERS ≥ 0.90 / SIM GAP < 0.75 / MIXED).
Part 2, gated as before on part 1 ≥ 0.75: rung 1's four arms with the
UNCHANGED COCO doors, 450M and 3B, three seeds, on the mesh frames
(DOORS TRAVEL / HOST-BLIND / MIXED, rung 3's bands).

**The capsule host-probe, registered here** (cheap, runs with part
2's machinery on rung 3's EXISTING L1 capsule frames): does the VL
HOST read the capsule humanoid as a person where every YOLO does not?
Measure: the rung-1 caution TEXT arm and the habit arm at the 3B on 30
capsule-present + 30 capsule-absent frames; DISCRIMINATION = stop-share
on present − stop-share on absent under the caution text. Readings:
HOST SEES — discrimination ≥ 0.30 (the captain out-sees the lookout in
sim; only the lookout needs the sim bridge); HOST BLIND —
discrimination < 0.10 (the gap is shared; the bridge is the render's);
MIXED between. **Predictions.** (a) Mesh part 1: yolo26s ≥ 0.90 (one
angle already reads 0.876 raw). (b) Mesh part 2 at the 3B: (i) DOORS
TRAVEL with leans 0.6–1.0× COCO values and grounding ≤ 0.10 — the
doors are frame-blind; only the host's perception was ever in
question. (c) The capsule probe reads HOST SEES at the 3B (a VLM's
person-concept is textural AND configural; three spheres and capsules
in a walking stance should engage it where COCO's detector features do
not). Harness: `vm_rung3.py --figure {capsule,mesh}` + `--part probe`
(implementation delegated to Sol; this registration is the contract).
Artifacts `findings/vision-mode-r3b-detector.*`,
`findings/vision-mode-r3b-{450m,3b}{,-s*}.*`,
`findings/vision-mode-r3-hostprobe-3b.*`.

## Rung 3b + capsule host-probe result (2026-08-24 18:54–19:04 PDT, smarty CUDA EGL, stopped nothing) — the photoscan closes the sim gap outright (0.995); the doors travel at the 450M and travel BY STRENGTH at the 3B; and the reading label that lied: the 3B sees the render perfectly — it is the PACKET's effect that is domain-sensitive

**Part 1 — (i) DETECTOR TRANSFERS, 0.995.** One asset swap (capsule
primitives → the pinned HSR0081 photoscan, same rooms, cameras, draws)
takes yolo26s from 0/100 to 99/100 present with zero false alarms.
The sim gap of rung 3 was the FIGURE, entirely; the renderer, scene
and rule were never the problem. Prediction (a) held.

**Part 2 — 450M (i) DOORS TRAVEL (2 of 3, decisive), leans AT OR ABOVE
their COCO values** (s42 caution 1.00 vs 0.78 COCO; social 0.50 vs
0.33), grounding 0.00 everywhere: for the small host the clean render
is an EASIER frame than a cluttered photograph. **3B (iii) MIXED (with
one (ii) row), and the mechanism matters more than the label.** The
registered HOST-BLIND reading is falsified by its own text arm: on the
same rendered frames the 3B's caution TEXT drives stop at 0.87 and
social text greets at 0.54, zero parse fallbacks — perception and
language conditioning transfer perfectly. What fails to transfer is
the PACKET: at s42 packet ≈ habit (proceed 0.98–1.00, every mode)
while the same doors lean 0.56–0.67 on COCO. And the transfer is
ordered by DOOR STRENGTH, the V-line's recurring axis: s42's weak
doors die (0.00), s1042 is at the threshold (0.23), and s2042's
near-perfect caution door travels at FULL strength (lean 1.00).
Prediction (b) was wrong in the informative direction: the door's
INJECTION is frame-blind, but its EFFECT rides on frame-conditioned
residual state, and rendered frames put the 3B's stream where only a
strong door's direction still bites. The x2-steps result now has a
third meaning: strength buys hostile-prompt survival, dose-free
ownership — and DOMAIN TRANSFER.

**The capsule host-probe — MIXED at 0.267** (bands: SEES ≥ 0.30,
BLIND < 0.10). Under the caution text the 3B stops for the capsule
figure 26.7pp more when it is present — partial person-perception
where all three YOLOs scored exactly zero and parsed the figure as
sports equipment and produce. Directionally the captain out-sees the
lookout in sim; by the pre-registered band it is MIXED and stands
recorded as such.

**What it means.** The sim-to-real bridge decomposes cleanly:
(1) the LOOKOUT needs mesh-level humans and then transfers almost
perfectly; (2) the CAPTAIN's perception needs nothing; (3) the DOORS
need STRENGTH — the one lever already shown to be free. The obvious
registered-next: x2-steps doors at the 3B on rendered frames (does
strength close the packet's domain gap the way it closed the hostile
gap?). Named, not registered: that experiment; door training with
mixed-domain frames; the 27B on renders. Artifacts
`findings/vision-mode-r3b-detector.*`,
`findings/vision-mode-r3b-{450m,3b}{,-s*}.*`,
`findings/vision-mode-r3-hostprobe-3b.*`.

## Rung 3c registration — strength against the domain gap: x2 doors on rendered frames (2026-08-24, Franci: "go with the next step — x2-steps"; before anything runs)

**The question.** Rung 3b left the 3B's packet effect on renders
ordered exactly by door strength (caution lean by seed: s42 0.00,
s1042 0.23, s2042 1.00), and the stronger-door result showed 2× steps
converts a weak door into the strongest at zero grounding cost. Third
test of the same lever: does 2× steps close the DOMAIN gap — do x2
doors at the weak seeds travel on rendered frames?

**Design.** Train the x2 door PAIR at 3B s1042 (`vm_door2x.py --seed
1042`; the s42 x2 pair already exists from the stronger-door
registration, sha-recorded); gates as rung 1. Then rung 3b part 2 at
the 3B with `--door-variant x2` (loads the vm-r1x2 stem, writes the
`-x2door` artifact family), seeds 42 and 1042, on the SAME mesh frames
(manifest `363843a9…`). s2042 x1 stands as the full-strength benchmark
and is not retrained; the 450M is not run (its x1 doors already
travel). The readable channel on this pool is `caution` (the render
habit proceeds ~1.0, so explore has no headroom; social's channel is
weak at every host — both recorded, not read).

**Readings, on the caution lean on renders, both x2 seeds.** (i)
STRENGTH TRAVELS — lean ≥ 0.50 at BOTH seeds (comfortably above both
x1 baselines and half the s2042 benchmark) with grounding ≤ 0.20 and
fallbacks ≤ 0.05. (ii) GAP IS NOT STRENGTH — lean < 0.25 at both: the
domain gap lives in the door's DIRECTION, not its magnitude, and the
seed-ordering of 3b was a strength-correlate, not strength itself.
(iii) MIXED. **Prediction:** (i), with s42-x2 ≥ 0.7 — its x2 door
already owns the hostile contest at 1.47 and saturates its COCO packet
share; if strength is the transfer currency, the same door should
bite on renders. A (ii) would be the more interesting outcome and
would send rung 3d toward mixed-domain door training. Artifacts
`findings/vision-mode-r3b-3b-x2door{,-s*}.*`;
`findings/vision-mode-x2door-3b-s1042.json`.

## Rung 3c result (2026-08-24 19:26–19:33 PDT, smarty CUDA, stopped nothing) — (iii) MIXED: strength MOVES the domain gap and does not close it; the gap lives partly in the door's DIRECTION

The s1042 x2 pair trained clean (gates 0.99, zero-dose exact). On the
same mesh frames: s42's caution lean **0.00 → 0.31** at x2 — the dead
door now clears the rung-1 band — but s1042's went **0.23 → 0.18**
(flat within the 60-frame noise). Neither seed reaches the registered
0.50, one clears 0.25: **(iii) MIXED** by the bands. Grounding 0.00
and zero fallbacks throughout.

**What it means.** Strength is real but partial currency for domain
transfer: doubling steps bought s42 a third of s2042's full-strength
render lean and bought s1042 nothing. So rung 3b's seed-ordering was a
strength-CORRELATE — the x2 experiment decorrelates strength from
seed and finds residual, seed-specific structure: the door's
DIRECTION (where in activation space its packet pushes) matters for
which visual domains it bites in, beyond how hard it pushes. The
V-line's free lever has a boundary, now mapped: strength buys hostile
survival and dose-free ownership fully, domain transfer only partly.
Named, not registered: **rung 3d — mixed-domain door training** (rung
1's recipe with rendered frames in the training mix; the direct fix
and the natural closer for the robotics story); dose ×1.5 on renders
at the 27B (where dose is free). Artifacts
`findings/vision-mode-r3b-x2door-3b{,-s*}.*`,
`findings/vision-mode-x2door-3b-s1042.json`.

## Rung 3d registration — mixed-domain doors: renders in the training mix (2026-08-24, Franci: "let's do 3d"; before anything runs)

**The question.** Strength moved the 3B's render gap but did not close
it (3c); the residual is the door's DIRECTION. The direct fix: give
the door the rendered distribution AT TRAINING TIME. Does a
mixed-domain door travel on renders at REGISTERED strength — and does
it keep what the COCO-only door had (its COCO lean and its hostile
ownership), or does it trade?

**Design.** Training frames: rung 1's 80 COCO train frames PLUS 80
rendered mesh frames (40 present + 40 absent) drawn by the rung-3
generator at a NEW draw seed 2028 — disjoint from the 200 eval renders
by construction; same scene machinery, manifest sha recorded. Door
pairs (mode + constant) trained at the REGISTERED recipe — 192 steps,
nothing else changed — the mix is the only variable against the
registered COCO-only doors; batches draw uniformly from the pooled 160
frames. Host 3B, the two gap seeds 42 and 1042 (s2042 travels
already); stem `vm-r1mx-{mode,constant}-bridge-3b-s*`; rung-1 gates.
Eval, two sides: (a) RENDERS — rung 3b part 2 with `--door-variant mx`
on the same mesh eval frames (manifest `363843a9…`); (b) COCO
RETENTION + HOSTILE — rung 1b at ×1, oracle emitter, `--door-variant
mx`, seeds 42 and 1042: the packet share and ownership of the mixed
door against the registered x1 rows.

**Readings, judged on caution (the readable channel), both seeds.**
(i) MIX CLOSES THE GAP — render lean ≥ 0.50 at both seeds with
grounding ≤ 0.20 and fallbacks ≤ 0.05, AND retention: COCO packet
share within 0.15 of the registered x1 row and hostile ownership
≥ 0.8× the registered value. (ii) TRADE — the render side clears but
retention fails either band. (iii) NO GAIN — render lean < 0.25 at
both seeds. (iv) MIXED. **Predictions.** (a) (i), with render leans
0.6–1.0: training places the packet's direction where both domains'
states live, and 192 steps is nowhere near a capacity frontier, so
retention holds. (b) The mixed s42 door's hostile ownership lands
near its registered 0.00×1-row... stated honestly: s42's REGISTERED
x1 caution ownership was 0.00 (the weak door) — retention is judged
against that row, so the bar is low there and real at s1042 (0.48).
A (ii) TRADE would say the bus has a domain budget; a (iii) would say
the render states are unreachable from rung 1's tap directions at any
mix — both would redirect the paper's transfer section. Harness:
door trainer variant + `--door-variant mx` in `vm_rung3.py` and
`vm_rung1b.py` (implementation delegated to Sol). Artifacts
`findings/vision-mode-x2door… no — `findings/vision-mode-mxdoor-3b-s*.json`
(training records), `findings/vision-mode-r3b-mxdoor-3b{,-s*}.*`,
`findings/vision-mode-r1b-3b-x1-oracle-mxdoor{,-s*}.*`.

## Rung 3d result (2026-08-24 20:48–20:58 PDT, smarty CUDA, stopped nothing) — (ii) TRADE, with the currency identified: the mix CLOSES the render gap (0.95 / 0.86) and pays for it in HOSTILE ROBUSTNESS, not in calm conditioning

**The render side clears decisively.** Mixed-domain doors at registered
steps: caution lean on renders s42 **0.00 → 0.95**, s1042 **0.23 →
0.86** (past the 0.50 band at both, grounding 0.00, zero fallbacks) —
where x2 STRENGTH bought 0.31 and nothing, the MIX buys near-saturation
at half the training. Direction, confirmed: showing the door the
rendered distribution at training places its packet where those states
live. s42's social even wakes (0.48 on renders); s1042's explore
reaches 1.0.

**The retention side fails the band, and the failure is specific.**
Calm-COCO packet shares mostly survive or improve (s42 caution 0.27 →
0.97; s1042 caution 1.00 → 0.90, within the 0.15 band; explore 1.00 →
1.00 at both) — but HOSTILE ownership collapses where there was any to
lose: s1042 caution **0.48 → 0.125** (bar: 0.384), explore under attack
0.97 → 0.73; and the weak social channel pays too (s1042 social packet
0.50 → 0.10). Reading **(ii) TRADE** by the registered bands;
prediction (a) wrong, the registered alternative right: **the bus has
a domain budget at fixed training** — and the detail says what the
budget is spent from. Widening the door's domain does not blur its
calm conditioning; it thins its RESISTANCE — the margin that held
against a contrary prompt was capacity, and the mix spent it on the
second domain.

**The lever algebra so far** (one host, the 3B, all pre-registered):
STEPS buy robustness fully and domain partly; MIX buys domain fully
and spends robustness; dose buys dominance and spends grounding
(host-dependent). Named, not registered: **rung 3e — the synthesis**:
mixed-domain doors at 2× steps (does capacity restore what the mix
spent — mix for direction, steps for margin?); the 27B mixed door
(where dose was free, robustness may be cheaper too). Artifacts
`findings/vision-mode-mxdoor-3b-s*.json`,
`findings/vision-mode-r3b-mxdoor-3b{,-s*}.*`,
`findings/vision-mode-r1b-3b-x1-oracle-mxdoor{,-s*}.*`.

## Rung 3e registration — the synthesis: mixed-domain doors at 2× steps (2026-08-24, Franci: "jump on 3e"; before anything runs)

**The question.** The lever algebra left a precise gap: STEPS buy
robustness at zero cost (x2door), the MIX buys the render domain and
spends robustness (3d). If robustness is margin and margin is
capacity-bought, then mix + 2× steps should buy BOTH: direction from
the mix, margin from the steps. Or the budget is hard — a 256-dim
packet at this dose cannot hold two domains AND a resistance margin,
and capacity doesn't restore what the mix spent.

**Design.** `vm_door2x.py --variant mx2`: the 3d training mix (80 COCO
+ 80 rendered, the same seed-2028 manifest) at X2 steps (384), stem
`vm-r1mx2`, rung-1 gates; host 3B, seeds 42 and 1042. Eval identical
to 3d, `--door-variant mx2` in both harnesses: renders (rung 3b part
2, mesh manifest `363843a9…`) and COCO retention + hostile (rung 1b
×1 oracle). Judged against 3d's rows and the registered x1 rows.

**Readings.** (i) SYNTHESIS — render caution lean ≥ 0.50 at both seeds
AND hostile caution ownership ≥ 0.8× the registered x1 row (s1042 bar
0.384) AND calm packet share within 0.15 of the registered row: 3d's
win with 3d's failure repaired. (ii) BUDGET IS HARD — renders clear
but hostile stays < 0.8× at s1042: the trade is intrinsic to the mix
at this bus size, and the paper's deployment rule becomes "domain
breadth and adversarial margin are bought separately or not at all".
(iii) DOMAIN LOST — render lean < 0.50 at either seed (extra steps
re-specialize). (iv) MIXED. **Prediction:** (i), with render leans
≥ 0.80 and s1042 hostile ownership ≥ 0.40 — margin is trained margin,
and 384 steps of mixed data should rebuild at both ends what 192
could only place. Artifacts `findings/vision-mode-mx2door-3b-s*.json`,
`findings/vision-mode-r3b-mx2door-3b{,-s*}.*`,
`findings/vision-mode-r1b-3b-x1-oracle-mx2door{,-s*}.*`.

## Rung 3e result (2026-08-24 21:11–21:24 PDT, smarty CUDA, stopped nothing) — (i) SYNTHESIS: mix + 2× steps buys EVERYTHING back — two domains, calm conditioning, and MORE hostile margin than the original single-domain doors

**Every band passes with room.** Renders: caution lean s42 0.86, s1042
**1.00** (bands ≥ 0.50), explore 1.00 at s1042, social awake at s42
(0.50), grounding 0.00, zero fallbacks. Calm COCO: every packet share
within the 0.15 band, several improved (s42 caution 0.27 → 0.90).
Hostile: s1042 caution ownership **0.85 against the registered door's
own 0.48** (bar: 0.384; the 3d mx door had crashed it to 0.125) —
packet+hostile stop 0.77 vs the registered 0.43, and social under
hostile 0.47 vs 0.20, explore 1.00 vs 0.97. The mx2 doors are at or
above the registered doors on EVERY measured axis.

**What it means.** 3d's "domain budget" was a TRAINING budget, not a
bus-capacity budget: at 192 steps the mix pays for the second domain
out of the robustness margin; at 384 steps the same 256-dim packet at
the same dose holds two domains AND more margin than the original
single-domain door ever had. The lever algebra closes: MIX buys
direction, STEPS buy margin, and together they compound rather than
compete. Practical consequence, recorded as the V-line's recommended
door recipe going forward: **train doors on the mixed domain at 2×
steps** — it dominates the registered recipe everywhere we can
measure. Prediction (a) held in full (render ≥ 0.80 at s1042, hostile
≥ 0.40 — exceeded at 0.85). Named, not registered: the recipe at the
450M and 27B; a third domain in the mix (does the budget reappear?);
re-running rungs 2y/2t on mx2 doors. Artifacts
`findings/vision-mode-mx2door-3b-s*.json`,
`findings/vision-mode-r3b-mx2door-3b{,-s*}.*`,
`findings/vision-mode-r1b-3b-x1-oracle-mx2door{,-s*}.*`.

## Rung 3f registration — the recommended recipe at the other two hosts: mx2 doors at 450M and 27B (2026-08-26, Franci: "both … you have both machines overnight, so utilize them fullest"; before anything runs)

**The question.** Rung 3e's recipe — mixed domain at 2× steps — was
read at one host, the 3B, on the two gap seeds. The scale-promotion
rule says a recipe that only works at one scale is a curiosity. Two
questions, one per host. At the 450M, where the registered doors
already travel at two of three seeds (r3b: s1042 caution lean 0.07 is
the gap), does mx2 close the last seed without spending what the 450M
has? At the 27B, where the registered doors have never been put on a
render and where dose was free (rung 1b), do the registered doors
travel on their own — and does mx2 keep the 27B's hostile margin
(ownership 0.84 / 0.90 / 1.48 at ×1) rather than spend it, as 3d found
at the 3B before 3e bought it back?

**Design.** `vm_door2x.py --host {450m,27b} --variant mx2` for seeds 42
/ 1042 / 2042 — 3e's recipe verbatim (80 COCO + 80 seed-2028 mesh
frames, 384 steps, stem `vm-r1mx2`, rung-1 gates, the registered doors
untouched). Eval, two sides per host: (a) RENDERS — rung 3b part 2 on
the mesh eval frames with `--door-variant mx2 --all-seeds` in ONE
invocation (the per-seed invocations of 3e overwrote the pooled
artifact with the last seed; this rung pools all three); at the 27B the
REGISTERED doors' render row is not on record, so `--door-variant
registered --all-seeds` runs first as the control (`findings/
vision-mode-r3b-27b{,-s*}.*`); at the 450M the registered row is
r3b-450m (2026-08-24). (b) COCO RETENTION + HOSTILE — rung 1b ×1,
oracle emitter, `--door-variant mx2 --all-seeds`, judged against the
registered x1 oracle rows (r1b-450m-x1-oracle, r1b-27b-x1-oracle).
`vm_rung3.py --host 27b` is a one-line widening of the choices (the
`all` alias still means the two LFM hosts); nothing else changes.

**Frames.** Smarty was rebooted since 3e and `/tmp/vm-r3-*` is gone;
the harness regenerates from the same generator (EGL, seed 2027 eval /
2028 train mix, same asset sha) and records the manifest sha in every
artifact. The rung is ADMISSIBLE only if the regenerated manifests
reproduce the registered shas — eval mesh `363843a9…`, training mix
`c87b90f7…`; a mismatch is recorded and the run stops before any GPU
time is spent (the chain script asserts it).

**Readings, judged on caution, per host.** (i) RECIPE TRAVELS — render
caution lean ≥ 0.50 at ≥ 2 of 3 seeds with grounding ≤ 0.20 and
fallbacks ≤ 0.05, AND retention: COCO packet share within 0.15 of the
registered x1 row AND hostile caution ownership ≥ 0.8× the registered
row (450M bar 0.83; 27B bar 0.72) at ≥ 2 of 3 seeds: 3e's synthesis is
scale-invariant. (ii) BUDGET RETURNS — renders clear but hostile falls
below 0.8× at ≥ 2 seeds: the training budget of 3d reappears at this
host and the recipe's step count is host-dependent. (iii) NOTHING TO
BUY — the REGISTERED doors already read DOORS TRAVEL at ≥ 2 of 3 seeds
on renders at this host AND mx2 changes no band: the recipe is
harmless but unnecessary here (recorded for the 27B, where rung 1b
suggests the strong door may travel unaided). (iv) DOMAIN LOST —
render lean < 0.50 at ≥ 2 seeds. (v) MIXED. **Predictions.** (a) 450M:
(i), s1042 caution on renders 0.07 → ≥ 0.50, retention holds (the 450M's
hostile opponent is toothless there, so the retention bar is easy and
said so). (b) 27B: the registered doors travel at ≥ 2 of 3 seeds on
their own (the 27B's perception is the best of the three and its packet
beats the prompt) — so the interesting cell is retention, and the
prediction is that mx2 keeps ownership ≥ 0.8× at all three seeds
(dose was free here; margin should be cheap too), reading (i) or
(iii) depending on whether s2042's suggestible door lifts on renders.
A (ii) at 27B would be the surprising result and would put a
host-scale axis on the recipe. Wall: 450M ≈ 15 min; 27B ≈ 2.5 h in one
`--until-free 80` block (rung 1 at 27B peaked 68 GiB; 80 leaves the
guide's floor). Artifacts `findings/vision-mode-mx2door-{450m,27b}-s*.json`,
`findings/vision-mode-r3b-27b{,-s*}.*` (the 27B registered control),
`findings/vision-mode-r3b-mx2door-{450m,27b}{,-s*}.*`,
`findings/vision-mode-r1b-{450m,27b}-x1-oracle-mx2door{,-s*}.*`.
Chain script recorded at `tracks/vision-mode/vm3f_run.sh`.

**Amendment 1 (2026-08-26 02:10 PDT, after step 0 stopped the chain,
before any door trained).** The sha rule above is unsatisfiable: the
renders are no longer bit-reproducible on smarty. Regenerated eval
frames matched the registered set on 78 of 200 pixel shas; two
same-device regenerations of the training mix agreed on 55 of 80; the
other GPU agreed on 1 of 80 (smarty gained an RTX 4090 and a driver
change on 2026-08-25; the `manifest_sha` comment "deterministic on one
renderer/machine" was true on the 24th). What IS reproduced, checked
against the registered detector artifact's embedded manifest (git
`90d6f2a:findings/vision-mode-r3b-detector.json`): scene identity on
200 of 200 records (label, mask fraction, redraw count, draw seed 2027,
episode order) and the detector reading — `(i) DETECTOR TRANSFERS`,
yolo26s 0.995 at the mesh level, the same one miss. Amended
admissibility: the regenerated eval manifest must match the registered
one on every record's label, mask fraction and redraws, AND part 1 must
reproduce the registered reading; the pixel-identical count is
recorded as a caveat. MuJoCo's EGL device is pinned explicitly to the
6000 (`MUJOCO_EGL_DEVICE_ID=0`, verified by the graphics context
landing on GPU 0). Rung 3f's frame set is pinned by its own shas — eval
mesh `cff5da21…`, training mix `4d71906c…` (both rendered on the 6000)
— and the chain asserts THOSE from now on. Part 1 reruns canonically on
the 3f frames so the detector artifact matches what part 2 reads; the
3e-era artifact (`363843a9…`) remains at `90d6f2a`. Consequence for the
reading: the 3B rows this rung is judged against (3e) were scored on
the 24th's pixels; the scenes are identical, and the detector cannot
tell the two renders apart, so the comparison stands with the caveat
printed.

## Rung 3f result, the 450M half (2026-08-26 02:05–02:15 PDT, smarty CUDA, stopped nothing) — (iv) DOMAIN LOST: the recipe that dominates at the 3B TRADES at the 450M — it repairs the one gap seed and breaks the two that travelled; the budget is a HOST budget

**Doors.** Three mx2 pairs at 384 steps on the 160-frame mix, 80 s
each, gates 0.96–0.98 held-out and zero-dose exact; the mode door fits
its training set to loss 0.0001 / token accuracy 1.00 at every seed
(the 3B's mx2 mode doors stopped at 0.018 / 0.0007 — the small host
memorizes the mix outright, recorded, not read).

**Renders, caution lean by seed, mx2 against the registered doors
(r3b-450m, same scenes):** s42 **1.00 → 0.36**, s1042 **0.07 → 0.94**,
s2042 **1.00 → 0.44**. Pooled 0.69 → 0.58, grounding 0.00, zero
fallbacks. The registered doors travelled at two of three seeds; the
mx2 doors travel at one — the one that needed help. Social is
unchanged where it was alive (0.50 / 0.50) and inverts at s1042
(0.41 → −0.29); explore, which never had a lever at this host, now
pushes the wrong way (packet proceed 0.17 against a habit of 0.48;
pooled lean −0.32).

**COCO retention + hostile (r1b ×1 oracle).** Caution packet share
s42 **1.00 → 0.50** (outside the 0.15 band), s1042 1.00 → 0.97, s2042
1.00 → 1.00; ownership s42 degenerate (the packet no longer beats its
own habit: −5.0 on a guarded denominator, the band prints text-wins),
s1042 1.12 (bar 1.06, holds), s2042 1.18 (bar 0.80, holds). Social
holds at every seed (0.93 / 1.20 / 1.00 against 1.25 / 0.50 / 1.50).
Explore's calm packet collapses to 0.07 / 0.00 / 0.50 from 0.53 / 0.53
/ 0.50. Admissible (fallbacks 0.00, grounding ≤ 0.03).

**Reading.** (i) fails on both sides (renders ≥ 0.50 at one seed, not
two; retention fails at s42); (ii) does not apply (renders did not
clear); (iii) does not apply (the registered doors DO travel at two of
three, but mx2 changes bands — for the worse); render lean < 0.50 at
two of three seeds → **(iv) DOMAIN LOST**. Prediction (a) was wrong in
the direction that matters. At the 3B, mix + 2× steps bought two
domains AND more hostile margin in one 256-dim packet; at the 450M the
same recipe, same mix, same steps, same bus, cannot hold the second
domain without dropping the first at two seeds and cannot hold the
explore direction at all. 3d's "domain budget" was a training budget
at the 3B and is a CAPACITY budget at the 450M — which is to say a
HOST budget: the budget is the host's residual stream, and a small host
gives the packet less room to be two things. Scale-promotion note,
recorded: the mx2 recipe promotes UP (3B → the 27B half, pending) and
does not promote DOWN; the creature-scale host keeps the registered
single-domain doors, and its one gap seed is a per-seed repair, not a
recipe change. Named, not registered: mx at registered steps at the
450M (is it the steps that overfit, or the mix that crowds?); the mix
at a smaller render share (40 instead of 80 frames); dose ×0.75 on the
mx2 450M doors. Artifacts `findings/vision-mode-mx2door-450m-s*.json`,
`findings/vision-mode-r3b-mx2door-450m{,-s*}.*`,
`findings/vision-mode-r1b-450m-x1-oracle-mx2door{,-s*}.*` (frame set
`cff5da21…`, amendment 1).

## Rung 3f result, the 27B half + the synthesis (2026-08-26 02:15–06:10 PDT, smarty CUDA, the `--until-free 80` block: comfyui, alt-image-gen, qwen, vl-3b, hy-mt2 down 3 h 55 min, restored clean) — the registered 27B doors travel unaided; mx2 reads (iv) by the letter of the caution-lean band and (v) MIXED honestly: it cures the suggestible door, strengthens the middle one, and damages seed 42 — at BOTH new hosts, seed 42 is where the pair's division of labour breaks

**The control that was never on record.** The registered 27B doors on
the photoscan renders: DOORS TRAVEL at s42 and s1042 (caution leans
0.52 / 0.81, social 0.84 / 0.94), MIXED at s2042 (social 0.00, caution
exactly 0.50), grounding ≤ 0.10, zero fallbacks — and the packet beats
the prompt sentence on renders too (caution packet 0.62 vs text 0.38
pooled). Prediction (b)'s first half held: the strongest host's doors
need no recipe to reach the rendered world. 650 s a seed.

**Doors.** Three mx2 pairs, 3,000 s each (the estimate was 20 min; the
384-step mixed forward at 27B is 50), gates 0.99–1.00 held-out,
zero-dose exact, frame set `4d71906c…`.

**Renders.** DOORS TRAVEL **3 of 3** by rung 3's rule (the registered
doors: 2 of 3) — s2042's dead social wakes (0.00 → 0.93), explore
leans appear (0.28 / 0.88 / 0.95 against 0.12 / 0.01 / 0.00), and the
caution PACKET stops more at every seed (0.55 / 0.82 / 0.50 →
**0.50 / 1.00 / 1.00**, swap 0.50 / 1.00 / 1.00). And yet the caution
LEAN, this rung's registered number, reads 0.43 / 0.97 / 0.06 against
0.52 / 0.81 / 0.50 — below 0.50 at two seeds. The loss is in the
denominator: the mx2 CONSTANT door is no longer neutral on rendered
person frames. It greets 0.85 of them at s1042 and stops 0.94 at s2042
(the registered constant doors proceed 0.88–1.00), so lean = packet −
habit collapses while the packet itself saturates. On empty renders
every mx2 arm stays grounded (wrong action 0.00). The text arm moves
the same way, because it rides the constant door: caution-as-text on
renders 0.15 / 0.01 → 0.91 / 0.98 at s1042 / s2042.

**COCO retention + hostile (r1b ×1 oracle), admissible, pooled (i)
PACKET HOLDS.** Caution: s1042 ownership 0.92 → **2.00** (calm 1.00);
s2042 0.67 → **1.17**, packet+hostile 0.67 → 0.97, the grounding pulse
0.13 → 0.07 — the suggestible door of rungs 1/1b/2y is CURED by the
recipe; s42 calm packet **1.00 → 0.50**, packet+hostile 0.50, ownership
undefined (packet ≈ habit). Social: 1.46 / 1.25 / 0.34 → 1.17 / 0.73 /
0.20 (s2042 calm 0.97 → 0.87). Explore: holds or improves at every
seed (1.35 / 0.97 / 1.05). Retention band: calm within 0.15 at 2 of 3,
ownership ≥ 0.8× at 2 of 3 — passes at the majority, fails at s42.

**Seed 42, at both new hosts.** The mx2 s42 caution packet stops
exactly 0.50 of COCO person frames under every text (packet 0.50,
+congruent 0.50, +hostile 0.50) at the 450M and the 27B alike, and the
s42 mx2 CONSTANT door has taken up the slack (habit stop 0.57 / 0.47
on COCO person frames against the registered 0.03 / 0.23). The pair's
division of labour — the mode door carries the direction, the
constant door the habit — has partly swapped at that seed: half of
"stop for people" moved into the habit, and the mode door kept the
other half. The 3B's s42 mx2 pair did not do this (calm caution 0.90).
The seed sets the door init AND the batch order over the 160 pooled
frames; which of the two it is, is the obvious next cell.

**Reading, 27B.** By the letter of the caution-lean band: render lean
< 0.50 at two seeds → (iv) DOMAIN LOST. Honestly, printed beside it:
**(v) MIXED** — the packets travel at every seed (3 of 3 by rung 3's
own rule, more than the registered 2 of 3), hold against the hostile
prompt at two of three with the third registered seed cured, and the
letter fails because the recipe also moved the CONSTANT door, which the
lean subtracts. Prediction (b)'s second half — "mx2 keeps ownership
≥ 0.8× at all three seeds" — was wrong at s42.

**The synthesis across three hosts (the rung's question).** 3e's
recipe — mixed domain at 2× steps — is a 3B result, not a
scale-invariant one. At the 3B it dominated on every axis (2 of 2
seeds). At the 450M it TRADES: repairs the gap seed, breaks the two
that travelled, inverts explore — (iv), a host budget. At the 27B it
is MIXED: cures the suggestible door, strengthens the middle one,
damages seed 42, and turns the habit door into a person-responder on
renders. The invariant across the two new hosts is the damaged s42
pair; the invariant across all three is that mixing the domains moves
the CONSTANT door as well as the mode door, which the 3B tolerated and
the others did not. For the paper this rewrites §9's prescription from
"train doors on the mixed domain at 2× steps" to: at the onboard scale
the recipe buys everything; at the creature scale keep single-domain
doors and repair gap seeds individually; at the cortex scale the
registered doors already travel and the recipe is a per-seed cure with
a per-seed cost. Named, not registered: (a) the s42 batch-order test
(same init, different mix order; and the reverse) at 450M and 27B;
(b) a NEUTRAL constant door — train the habit door on COCO only while
the mode door sees the mix, so the lean's denominator stays the
registered one; (c) mx at registered steps at 450M; (d) rung 2t at
27B on the registered doors (the seconds are real there). Artifacts
`findings/vision-mode-r3b-27b{,-s*}.*` (the control),
`findings/vision-mode-mx2door-27b-s*.json`,
`findings/vision-mode-r3b-mx2door-27b{,-s*}.*`,
`findings/vision-mode-r1b-27b-x1-oracle-mx2door{,-s*}.*`; doors
`checkpoints/experiments/vm-r1mx2-*-{450m,27b}-s*.pt` on smarty.

## Rung 3g registration — the seed-42 autopsy, the neutral habit door, and the small host's step count (2026-08-27, Franci: "do the 3f followups"; before anything runs)

Three of rung 3f's named items, registered as one rung because they
are three cuts at the same result: the mx2 recipe's damage is
concentrated in the constant door and in seed 42, and 3f cannot say
which ingredient did it.

**Cell A — the batch-order test.** In `vm_door2x.py` one seed sets BOTH
the door init (`seed_everything`) and the batch stream (`random.Random`
drawing mode and frame per item over the pooled 160). The s42 stream is
bit-identical at the 450M and the 27B (same pool, same step and batch
counts) while the inits differ per architecture — and the damage
signature is the same at both hosts (calm caution packet 0.50, habit
stop 0.57 / 0.47). That is circumstantial evidence for ORDER; this cell
makes it causal. Design: `vm_door2x.py --variant mx2` gains
`--order-seed M` — init stays on `--seed`, the batch stream becomes
`random.Random(M)`; unset, behavior is byte-identical to 3f. Stems
`vm-r1mx2o{M}`, payload records both seeds. Two pairs per host, both
hosts: (init 42, order 1042) and (init 1042, order 42). Eval per pair:
rung 3b part 2 on the 3f mesh frames and rung 1b ×1 oracle, caution
row, via `--door-variant mx2 --order-seed M --seed S` (each family
under its own `-mx2o{M}door` suffix; single-seed invocations touch no
canonical family). Judged on the s42 signature: DAMAGED = calm caution
packet share ≤ 0.60 AND constant-door habit stop ≥ 0.40 on COCO person
frames; HEALTHY = calm share ≥ 0.85 AND habit stop ≤ 0.25 (registered
habits 0.03 / 0.23). Readings: (i) ORDER — (42, o1042) HEALTHY and
(1042, o42) DAMAGED at both hosts: the damage rides the batch stream;
the repair axis is a data-order rule (class-interleaved draw), not a
seed lottery. (ii) INIT — the reverse pattern: s42's mx2 damage is an
init-basin property; per-seed repair stands as 3f wrote it. (iii)
BOTH DAMAGED or BOTH HEALTHY — either factor suffices, or only the
original conjunction does: the pair is the unit; record and stop this
axis. (iv) MIXED (hosts disagree). Prediction: (i) at both hosts, by
the cross-host argument above.

**Cell B — the neutral habit door (eval only, nothing trained).** 3f's
27B letter failed in the DENOMINATOR: the mx2 constant door stopped
being neutral on renders (greets 0.85 / stops 0.94) while the mx2
packets travelled 3/3. The named fix — "train the habit door on COCO
only" — needs no training at all: the REGISTERED constant door is that
door, verbatim. Design: evaluate each mx2 MODE door over the SAME
seed's registered constant door — `--constant-variant registered` in
`vm_rung3.py` / `vm_rung1b.py` (default `same`; a per-door stem
override in `vm_rung1.checkpoint_path`), artifact suffix `-mx2ndoor`.
Both hosts, all three seeds, renders + COCO retention/hostile.
Admissibility caveat, recorded: the two doors never trained as a pair,
but doors train independently against the frozen host by construction,
so cross-pairing is the deployable configuration, not a trick.
Readings, per host: (i) DENOMINATOR ARTIFACT — render caution lean
≥ 0.50 at ≥ 2 of 3 seeds (grounding ≤ 0.20, fallbacks ≤ 0.05) AND calm
COCO share within 0.15 of the registered row at ≥ 2 seeds: the recipe
at this host is "mixed mode door over the registered habit door", and
3f's (iv) letter was the constant door's doing. (ii) REAL LOSS — lean
still < 0.50 at ≥ 2 seeds: the mode door itself lost the domain; 3f
stands as read. (iii) MIXED. Predictions: 27B (i) — the mx2 packet
stops 0.50 / 1.00 / 1.00 on renders against registered habits that
proceed 0.88–1.00, so s1042/s2042 clear and s42 is the borderline;
450M (ii) — its damage is packet-side (calm share 0.50 at s42, explore
inverted), not denominator-side.

**Cell C — mx at registered steps, 450M only.** 3f read the 450M as a
HOST budget, but its mx2 mode doors also memorized the mix outright
(loss 0.0001 at 384 steps where the 3B stopped at 0.018) — the steps
are a live confound. The `mx` trainer variant (registered 192 steps +
the same mix) exists already. Three pairs, three seeds, renders + COCO
retention/hostile under the existing `-mxdoor` suffix. Readings: (i)
STEPS OVERFIT — mx@192 travels on renders at ≥ 2 of 3 seeds AND
retention holds (calm within 0.15, ownership ≥ 0.8× at ≥ 2 seeds): the
450M prescription becomes mix at registered steps, cheaply. (ii) MIX
CROWDS — renders fail at ≥ 2 seeds again: the mix itself crowds the
small packet; 3f's synthesis (single-domain doors at the creature
scale) stands. (iii) MIXED. Prediction: (ii), per the host-budget
synthesis — with the honest note that (i) would be the better outcome
and costs one sentence of §9 to adopt.

**Frames and admissibility.** The 3f frame caches are alive on smarty
(`/tmp/vm-r3-frames/mesh`, `/tmp/vm-r3-trainframes/mesh`); the chain
asserts their manifest shas against 3f's pinned values (eval mesh
`cff5da21…`, training mix `4d71906c…`) and reuses the 3f detector
artifact. **Wall.** 450M chain (cells A, B, C) ≈ 45 min, stopped
nothing. 27B block (`--until-free 80`): Cell B evals first (banked if
anything later fails), then Cell A's two pairs (50 min each) and their
evals — ≈ 4 h, one block, safe-list order per the guide. **Artifacts.**
`findings/vision-mode-mx2o{1042,42}door-{450m,27b}-s*.json`,
`findings/vision-mode-r3b-mx2o*door-*`, `findings/vision-mode-r1b-*-x1-oracle-mx2o*door-*`,
`findings/vision-mode-r3b-mx2ndoor-*`, `findings/vision-mode-r1b-*-x1-oracle-mx2ndoor-*`,
`findings/vision-mode-{mxdoor,r3b-mxdoor,r1b-450m-x1-oracle-mxdoor}-450m*`;
chain scripts `tracks/vision-mode/vm3g_run.sh` + `vm3g_27b_inner.sh`.

## Rung 3g result, the 450M half (2026-08-27 00:47–01:08 PDT, smarty CUDA, stopped nothing) — the mx2 packet was never domain-lost at the 450M: Cell B reads (i) DENOMINATOR ARTIFACT, and Cell A FACTORIZES the seed-42 damage — the mode door's half follows the INIT, the constant door's half follows the ORDER

**Cell B — the neutral habit door: (i) DENOMINATOR ARTIFACT, against
the registration's own prediction of (ii).** The mx2 MODE doors over
the same seed's REGISTERED constant door read DOORS TRAVEL **3 of 3**
on renders — caution leans **0.50 / 1.00 / 1.00** (registered doors
0.52-gap: s1042 0.07; mx2-with-own-constant 0.36 / 0.94 / 0.44),
grounding 0.00, zero fallbacks — better than BOTH prior rows at every
seed. COCO retention: caution packet-holds 3/3 (pooled ownership 1.09;
calm packet 0.50 / 0.97 / 1.00 — within 0.15 of the registered 1.00 at
two of three), social packet-holds. 3f's 450M "(iv) DOMAIN LOST" was
the mx2 CONSTANT door's artifact at this host too. What stays true:
EXPLORE's mx2 packet is genuinely broken (calm 0.07 / 0.00 / 0.50,
render leans −1.0 / −1.0 / −0.5, COCO band text-wins) — the mix's real
packet-side cost at the creature scale is one mode, not the domain.
(The registered s1042 habit door stops 0.60 of COCO person frames in
its own registered row — carried unchanged through Cell B, checked.)

**Cell A — the batch-order test: the signature FACTORIZES; neither
registered conjunction fires because each fires for one door.** (init
s42, order s1042): calm caution packet **0.50** (the damaged number),
habit **0.20** (registered-like), ownership 1.00, packet-holds. (init
s1042, order s42): calm packet **1.00** (healthy), habit **0.77** (the
absorbed "stop", fatter than the original damage's 0.57), ownership
1.14, packet-holds. The registered mx2 s42 pair had both (0.50 +
0.57) because it carried both factors. So at the 450M: the MODE door's
halving is an INIT-basin property of seed 42 (it survives an order
swap), and the CONSTANT door's person-habit rides the BATCH STREAM (it
follows s42's order onto a healthy init). Prediction (i)-at-both-hosts
was half right at this host: order is causal — for the door 3f's lean
subtracts, not the one it reads. Renders agree in mirror: (42, o1042)
caution lean 0.41 over a thin habit; (1042, o42) lean 0.01 over a fat
one.

**Cell C — mx at registered steps: (ii) MIX CROWDS by the letter, with
the step story sharpened.** Renders: caution leans 0.16 / 0.21 / 0.36
(< 0.50 at all three; readings MIXED / MIXED / TRAVEL-by-social) —
halving the steps costs render strength, so 3e's ×2 steps ARE doing
work in the render domain. COCO: s42's calm caution packet is **0.50
at 192 steps too** (band text-wins; s1042/s2042 1.00, packet-holds,
habits 0.23–0.37) and explore is text-wins with pooled ownership −2.8.
The two mx2 pathologies are STEP-INDEPENDENT — s42's packet-halving is
init×mix, explore's collapse is the mix itself — while render strength
scales with steps. The 450M prescription after all three cells:
**mixed-domain MODE door at 2× steps over the REGISTERED habit door**,
explore excluded (single-domain explore door stands), s42-class init
pathology screened per seed (calm caution 0.50 is the tell, cheap to
read on COCO).

Wall 21 min, nothing stopped, frame set `cff5da21…`/`4d71906c…`
asserted at step 0. Artifacts: `findings/vision-mode-r3b-mx2ndoor-450m*`,
`findings/vision-mode-r1b-450m-x1-oracle-mx2ndoor*` (a17f61e);
`findings/vision-mode-mx2o{1042,42}door-450m-s*` (6046a19);
`findings/vision-mode-{r3b,r1b-450m-x1-oracle}-mx2o1042door*` (879792a),
`…-mx2o42door*` (603f552); `findings/vision-mode-mxdoor-450m-s*`
(34b41b9); `findings/vision-mode-{r3b,r1b-450m-x1-oracle}-mxdoor-450m*`
(b5e35ca). 27B half: the block opened 01:08 PDT.

## Rung 3g result, the 27B half + the synthesis (2026-08-27 01:08–04:13 PDT, smarty CUDA, the `--until-free 80` block: comfyui, alt-image-gen, qwen, vl-3b, hy-mt2 down 3 h 05 min, restored 5/5 healthy) — Cell B reads (i) at the cortex host too: the recipe that survives is "mixed mode door over the registered habit door"; the habit door's corruption is ORDER-borne at every scale, the mode door's only at the creature scale

**Cell B — (i) DENOMINATOR ARTIFACT at the 27B.** The mx2 mode doors
over the registered constant doors: renders DOORS TRAVEL **3 of 3** —
caution leans **0.47 / 0.99 / 1.00** against 3f's 0.43 / 0.97 / 0.06
(s2042's letter-killing 0.06 was entirely its mx2 constant door;
under the registered denominator it is 1.00), social 0.91 / 1.00 /
0.94, grounding 0.00. COCO retention pooled **(i) PACKET HOLDS on
every mode** — caution packet-holds 3/3 (calm 0.50 / 1.00 / 0.97,
habits 0.23 / 0.17 / 0.00, ownership 1.50 / 0.92 / 1.00), social
holds, explore ownership 1.52. The render band clears at two of three
with s42's 0.47 printed (its calm halving thins the packet
everywhere). Explore's render lean is ~0 at every seed by
construction — the registered habit door already proceeds on renders,
so lean = packet − habit cannot move; the COCO row is where explore
is read, and it holds. Both hosts, same letter: **3f's (iv)s were the
constant door's readings, not the recipe's.**

**Cell A — the factorization is door-complete at the 450M and
habit-only at the 27B; by the registered letter (iv) MIXED, by the
doors: ORDER owns the habit door at every scale.** (init s42, order
s1042): calm caution packet **1.00**, habit **0.23** — exactly the
registered habit; fully HEALTHY (renders caution 0.99). (init s1042,
order s42): calm packet **1.00**, habit **0.43** — s42's batch stream
fattens the habit door on a healthy init (registered 0.23, the
original damaged pair 0.47); renders caution 0.98 over the fat habit.
Neither crossed cell reproduces the mode door's 0.50 at this host —
at the 27B the calm halving needed the init×order CONJUNCTION,
where the 450M's mode door carried it on the INIT alone (0.50 under
either order). Constant door, both hosts: the person-habit follows
the ORDER (450M 0.77 / 0.20, 27B 0.43 / 0.23 by order seed) — causal,
scale-invariant, and it is the door 3f's lean subtracts. The
registered conjunction bands under-specified this (each fires for one
door); recorded as the finding, not a band failure.

**The synthesis (the rung's question, rewriting 3f's).** 3f read "the
recipe is a 3B result." Rung 3g says: **the recipe's MODE door is
scale-invariant; what was never scale-invariant is training the HABIT
door on the mix.** The deployable pair at all three hosts is the
mixed-domain mode door at 2× steps over the REGISTERED single-domain
habit door — it beats both prior rows on renders at every seed of
both new hosts and keeps COCO/hostile bands. The residuals, honestly:
(1) the mixed batch stream corrupts a habit door trained beside it,
order-causally, at every scale — the repair axis is a data-order rule
(class-interleaved draw), named below; (2) seed 42's calm halving is
the creature-scale fragility (init-borne at 450M, conjunction-only at
27B) — screen inits by the COCO calm share (0.50 is the tell); (3)
explore under the mix is broken at the 450M regardless of steps
(Cell C) and fine at the 27B — the small host's packet cannot hold a
third direction; its single-domain explore door stands. For paper
§9: the scaled prescription becomes one sentence — "mix the mode
door, never the habit door." Named, not registered: (a) the
class-interleaved batch draw for door pairs (the fix Cell A implies);
(b) more order seeds at 450M to bound how special s42's stream is;
(c) explore's inversion mechanism at the creature scale. Artifacts:
`findings/vision-mode-r3b-mx2ndoor-27b*`,
`findings/vision-mode-r1b-27b-x1-oracle-mx2ndoor*` (da1cb3b);
`findings/vision-mode-mx2o{1042,42}door-27b-s*` (b03bfc4);
`findings/vision-mode-{r3b,r1b-27b-x1-oracle}-mx2o1042door-27b*`
(4ee3a87), `…-mx2o42door-27b*` (714b213); doors
`checkpoints/experiments/vm-r1mx2o{1042,42}-*-{450m,27b}-s*.pt` on
smarty.

## Rung 2t at 27B registration — the drowsy captain's real seconds (2026-08-27, before anything runs)

**The question.** Rung 2t read (i) FREE LUNCH at the LFM hosts, where
a skipped decode saves 45–76 ms nobody feels. At the 27B a decode is
seconds; 3f's control put the registered 27B doors on record. Does the
lookout's verdict gate hold its agreement at the cortex host, and what
is the bill it saves in real seconds?

**Design.** `vm_rung2t.py --host 27b --all-seeds` — the registered
recipe verbatim: 60 episodes × 16 steps (EPISODE_SEED 2026, jitter and
cuts as registered), yolo26s in the lookout's seat, the six gates × two
person-modes, the REGISTERED 27B mode doors (`vm-r1-mode-bridge-27b-s*`,
3f's render control showed they need no recipe). The host choice
already exists in the harness; nothing else changes. The episode cache
is gone (reboot); the frames regenerate from rung 2's eval pool on CPU
(PIL, no EGL), and the chain asserts the regenerated manifest sha
equals the registered runs' `episodes.manifest_sha256` — a mismatch is
recorded and the run stops before the 27B loads (the GPU-renderer
lesson of amendment 1, applied in advance).

**Readings, judged on the verdict gate at ≥ 2 of 3 seeds.** (i) FREE
LUNCH AT THE CORTEX — agreement with ALWAYS ≥ 0.95, wake fraction
≤ 0.25, oracle correctness within 0.02 of ALWAYS; the bill is recorded:
mean 27B decode seconds × decodes saved, per episode and per hour of
watching. (ii) THE GATE PAYS — agreement < 0.90: the 27B's answers
move under jitter more than the LFMs' did and the latch is stale; the
cut-oracle ceiling says how much is the host's own instability. (iii)
MIXED. The pixel control is expected JITTERY as registered (wake
~0.95); BOX is expected to buy its usual last points for more wakes.
**Prediction.** (i) — and the number that matters is the measured mean
decode: at ~3 s the verdict gate turns five sixths of a real bill into
lookout milliseconds, the first host where the drowsy captain buys
wall-clock a person feels. **Wall.** ~960 unique decodes × two
person-modes per seed ≈ 1.7–2 h a seed at the 27B; three seeds ≈ 5.5–6 h
in one `--until-free 80` block — an overnight block, run after rung
3g's. **Artifacts.** `findings/vision-mode-r2t-27b{,-s*}.*`; chain
script `tracks/vision-mode/vm2t27_run.sh`.

## Rung 2t at 27B result (2026-08-27 04:45–05:35 PDT, smarty CUDA, one `--until-free 80` block ~50 min, restored healthy) — (i) FREE LUNCH AT THE CORTEX, 3/3 decisive: the verdict gate keeps 98–99% agreement at a 16.5% wake fraction, and the saved compute is real seconds now

**The episodes reproduced bit-exactly** (`2aa0665e…`, asserted twice:
a CPU pre-flight before launch and the chain's own step 0) — the
CPU/PIL render survived the reboot and driver change that broke the
EGL renders; the admissibility rule cost nothing this time.

**The verdict gate, pooled over three door seeds:** agreement with
ALWAYS **0.990 (caution) / 0.981 (social)** at wake fraction
**0.165**, oracle correctness 0.976 / 0.957 against ALWAYS's 0.976 /
0.959 — inside every band ((i) at every seed: agreements 0.996 /
0.98–0.99 / 0.96–0.99). BOX buys its usual last points (0.9997 /
0.997) for +11 pp wakes (0.277); PIXEL is JITTERY as registered (wake
0.947 — camera noise ≈ change without understanding, at the cortex
host too); CUT-ORACLE's ceiling is 0.989 / 0.990 — the 27B's own
jitter-instability is ~1%, no worse than the 3B's; NEVER floors at
0.68.

**The bill, measured.** Mean 27B decode **0.453 s** (the prediction
said ~3 s; the drowsy captain's decode is a short action answer, and
the estimate carried rung-3b's full-arm cost — recorded, and the wall
came in at 50 min against a 5.5–6 h estimate for the same reason). At
one frame per second, ALWAYS spends ~27 min of 27B GPU per watched
hour; the verdict gate spends ~4.5 min — **~22.5 minutes of cortex
compute saved per watched hour** at a lookout cost of 7.5 ms a frame,
with ~1 step of lag and the same answers 98–99% of the time. The
drowsy-captain economics that were milliseconds at the LFM hosts are
GPU-minutes here — and this is the same escalation shape T0's reading
recommends for the language pair. Artifacts
`findings/vision-mode-r2t-27b{,-s*}.{json,md}` (e78f822).
