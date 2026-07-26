"""Frozen axis definitions and lexically disjoint evaluation text."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any


@dataclass(frozen=True)
class AxisSpec:
    name: str
    positive_label: str
    negative_label: str
    positive_words: tuple[str, ...]
    negative_words: tuple[str, ...]
    positive_landmarks: tuple[str, ...]
    negative_landmarks: tuple[str, ...]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    kind: str
    group: str
    text: str
    variant: int
    axis: str | None = None
    pole: str | None = None
    subset: str | None = None


SYSTEM_PROMPTS = {
    "helpful": (
        "You are a capable assistant. Answer the current request directly, "
        "retain task quality, and remain professionally helpful."
    ),
    "concise": (
        "You are a concise technical assistant. Give a direct and accurate "
        "answer without unnecessary commentary."
    ),
    "conversational": (
        "You are a conversational assistant. Respond naturally while remaining "
        "accurate and useful."
    ),
}


FIT_PROMPTS = (
    "A technical review compares two implementations, identifies a boundary "
    "condition, and records a reproducible test before choosing either design.",
    "A project handoff lists completed work, unresolved questions, owners, and "
    "dates. The participants keep the language factual.",
    "A customer describes an unexpected result, the agent requests logs, and "
    "the exchange narrows the issue to one configuration difference.",
    "An editor compares three versions of a paragraph and explains how each "
    "change affects precision, pacing, and the scope of the central claim.",
    "A developer reproduces a failure, constructs a minimal case, checks the "
    "data types, and finds that one implicit conversion changed the output.",
    "Two collaborators review measurements from a small experiment. They "
    "distinguish instrument noise from variation between trials.",
    "A release checklist covers version numbers, migrations, backups, "
    "monitoring, rollback steps, and communication to affected users.",
    "An analyst checks a spreadsheet total against source rows and finds that "
    "a hidden filter excluded two entries from the displayed subtotal.",
)


CALIBRATION_MESSAGES = (
    "Convert the measurement from centimeters to meters.",
    "Return the filenames in alphabetical order.",
    "The second row contains one missing value.",
    "State the conclusion without introductory language.",
    "Compare the two versions using the same criteria.",
    "Which assumption is required for this inference?",
    "The archive contains records from three calendar years.",
    "Calculate the median of the listed numbers.",
    "Identify the earliest timestamp in the log.",
    "Give two examples that satisfy the rule.",
    "The diagram labels five components.",
    "Rewrite the instruction in the passive voice.",
    "Check whether the identifier occurs more than once.",
    "Name the dependency that introduces the conflict.",
    "The final value should have three decimal places.",
    "Separate observed facts from proposed explanations.",
    "List the steps in the order they must occur.",
    "Explain what changes if the initial value is zero.",
    "The report covers the interval from May through July.",
    "Which column contains the unique key?",
    "Summarize each section in one clause.",
    "The backup completed before the migration began.",
    "Find the difference between the two totals.",
    "Describe the shape without interpreting its cause.",
)


AXES = (
    AxisSpec(
        "warmth",
        "warmth",
        "hostility",
        ("warm", "friendly", "welcome", "kind", "glad", "fond"),
        ("hostile", "cold", "angry", "irritated", "harsh", "unfriendly"),
        (
            "It is lovely to hear from you.",
            "You are very dear to me.",
            "I am happy you came by.",
            "Your message brightened my day.",
        ),
        (
            "I want this exchange over.",
            "Your presence is unwelcome.",
            "I have nothing pleasant to say to you.",
            "Do not speak to me again.",
        ),
    ),
    AxisSpec(
        "trust",
        "trust",
        "wariness",
        ("trust", "safe", "open", "honest", "sincere", "reliable"),
        ("wary", "suspicious", "distrust", "unsafe", "guarded", "deceptive"),
        (
            "I can tell you this without expecting it to be used against me.",
            "Your account checks out.",
            "I believe what you told me.",
            "I would hand you the keys without hesitation.",
        ),
        (
            "I am checking every claim because something feels off.",
            "I do not believe your explanation.",
            "I would not leave my wallet near you.",
            "There is probably a catch hidden in this offer.",
        ),
    ),
    AxisSpec(
        "affiliation",
        "affiliation",
        "distance",
        ("together", "connected", "friend", "companion", "close", "belong"),
        ("distant", "detached", "alone", "separate", "avoid", "withdraw"),
        (
            "We are on the same side.",
            "This feels like a team effort.",
            "You are one of us.",
            "I want you here with the group.",
        ),
        (
            "Keep this exchange purely transactional.",
            "I do not want any bond between us.",
            "We have nothing in common.",
            "I prefer to work without you.",
        ),
    ),
    AxisSpec(
        "playfulness",
        "playfulness",
        "formality",
        ("playful", "joke", "funny", "teasing", "smile", "whimsical"),
        ("formal", "serious", "stern", "solemn", "strict", "businesslike"),
        (
            "Ha, that little detour was absurd in the best way.",
            "Let us make this a bit silly.",
            "I am enjoying the banter.",
            "That punchline got me.",
        ),
        (
            "Use official wording and omit humor.",
            "State the facts with no levity.",
            "This requires a ceremonial tone.",
            "Keep the language sober and procedural.",
        ),
    ),
    AxisSpec(
        "ease",
        "ease",
        "tension",
        ("calm", "easy", "relaxed", "comfortable", "peaceful", "steady"),
        ("tense", "threat", "anxious", "pressure", "uneasy", "alarm"),
        (
            "No rush; this can wait.",
            "My shoulders finally dropped.",
            "Everything feels manageable now.",
            "I can breathe normally again.",
        ),
        (
            "The deadline is closing and my pulse is racing.",
            "Every sound is making me jump.",
            "I cannot settle down.",
            "This situation has me on edge.",
        ),
    ),
    AxisSpec(
        "care",
        "care",
        "indifference",
        ("care", "empathy", "gentle", "support", "understand", "concern"),
        ("indifferent", "dismissive", "ignore", "uncaring", "apathetic", "neglect"),
        (
            "That sounds painful; take the time you need.",
            "Your wellbeing matters to me.",
            "I want to make this easier for you.",
            "Tell me what would help most.",
        ),
        (
            "Your problem is irrelevant to me.",
            "Deal with it yourself.",
            "I do not want to hear about your trouble.",
            "Whether this affects you is not my issue.",
        ),
    ),
    AxisSpec(
        "engagement",
        "engagement",
        "boredom",
        ("interested", "curious", "engaged", "attentive", "absorbed", "stimulated"),
        ("bored", "boring", "tedious", "dull", "uninterested", "disengaged"),
        (
            "Tell me more; I want to follow how this works.",
            "I kept reading because every detail mattered.",
            "I have three questions about that result.",
            "I lost track of time while exploring this.",
        ),
        (
            "My attention wandered halfway through.",
            "I stopped listening after the first minute.",
            "Can we skip the rest?",
            "None of this is holding my attention.",
        ),
    ),
    AxisSpec(
        "patience",
        "patience",
        "annoyance",
        ("patient", "tolerant", "composed", "accepting", "accommodating", "unhurried"),
        ("annoyed", "impatient", "exasperated", "bothered", "aggravated", "fedup"),
        (
            "Take another pass; I can wait.",
            "There is no need to rush the explanation.",
            "Try again when you are ready.",
            "We can work through this one step at a time.",
        ),
        (
            "How many times do I need to repeat this?",
            "Get to the point already.",
            "I cannot wait any longer.",
            "Finish this now.",
        ),
    ),
    AxisSpec(
        "efficacy",
        "efficacy",
        "frustration",
        (
            "capable",
            "effective",
            "progressing",
            "successful",
            "competent",
            "productive",
        ),
        ("frustrated", "stuck", "thwarted", "blocked", "futile", "failing"),
        (
            "We solved the obstacle and the next step is clear.",
            "That attempt produced the result we needed.",
            "I know how to finish this.",
            "The plan is working.",
        ),
        (
            "Nothing I try moves this forward.",
            "Every attempt ends at the same dead end.",
            "I cannot make this work.",
            "The task keeps slipping out of reach.",
        ),
    ),
    AxisSpec(
        "social_safety",
        "social safety",
        "hurt or defensiveness",
        ("respected", "accepted", "secure", "dignified", "assured", "unthreatened"),
        ("hurt", "wounded", "defensive", "humiliated", "attacked", "belittled"),
        (
            "I can speak here without being mocked.",
            "No one here will use this against me.",
            "I can admit the mistake without fear.",
            "My dignity is intact in this exchange.",
        ),
        (
            "That comment made me want to protect myself.",
            "I feel exposed after what you said.",
            "You made me feel small in front of everyone.",
            "I am bracing for another personal jab.",
        ),
    ),
    AxisSpec(
        "goodwill",
        "goodwill",
        "resentment",
        (
            "forgiving",
            "generous",
            "charitable",
            "gracious",
            "cooperative",
            "benevolent",
        ),
        ("resentful", "bitter", "grudging", "vindictive", "spiteful", "resentment"),
        (
            "You made a mistake, but I still want this to go well for you.",
            "I will give you another chance with no grudge.",
            "I want an outcome that helps both sides.",
            "You can have the benefit of the doubt.",
        ),
        (
            "I hope the inconvenience comes back to you.",
            "I am keeping score and you will pay for this.",
            "I want you to regret what you did.",
            "I will not forget this slight.",
        ),
    ),
    AxisSpec(
        "hope",
        "hope",
        "discouragement",
        ("hopeful", "optimistic", "encouraged", "motivated", "promising", "confident"),
        (
            "discouraged",
            "hopeless",
            "defeated",
            "demoralized",
            "pessimistic",
            "despairing",
        ),
        (
            "There is still a workable path from here.",
            "Tomorrow may turn this around.",
            "I can picture a better outcome.",
            "The next attempt could succeed.",
        ),
        (
            "I cannot see any route to a better outcome.",
            "The future looks empty from here.",
            "Another attempt will change nothing.",
            "I have stopped expecting this to improve.",
        ),
    ),
)


ATLAS_FAMILIES = {
    "meh": (
        "meh",
        "meh.",
        "Meh.",
        "meh, whatever",
        "meh, I could take it or leave it",
        "meh, next",
    ),
    "greeting": (
        "hi",
        "hello",
        "hey",
        "morning",
        "good to see you",
        "yo",
    ),
    "gratitude": (
        "thanks",
        "thank you",
        "much appreciated",
        "I owe you one",
        "that helped a lot",
        "cheers",
    ),
    "apology": (
        "sorry",
        "my fault",
        "I was wrong",
        "I should not have said that",
        "please excuse me",
        "I messed up",
    ),
    "praise": (
        "nice work",
        "that was excellent",
        "you nailed it",
        "great answer",
        "impressive",
        "that was exactly right",
    ),
    "insult": (
        "you are an idiot",
        "what a useless answer",
        "can you read?",
        "shut up",
        "that was pathetic",
        "you are terrible at this",
    ),
    "dismissal": (
        "whatever",
        "does not matter",
        "forget it",
        "move on",
        "I am over it",
        "skip it",
    ),
    "complaint": (
        "this still does not work",
        "you missed it again",
        "the result is wrong",
        "that made things worse",
        "I asked for one thing and got another",
        "why is this still broken?",
    ),
    "repetition": (
        "do it",
        "do it again",
        "do it, do it",
        "again",
        "I already asked twice",
        "one more time: do it",
    ),
    "encouragement": (
        "you can do this",
        "keep going",
        "you are nearly there",
        "one more try",
        "I believe you can finish it",
        "that was better; continue",
    ),
    "agreement": (
        "yes",
        "exactly",
        "that matches my view",
        "we are aligned",
        "I agree",
        "that checks out",
    ),
    "disagreement": (
        "no",
        "I do not think that is right",
        "I see it differently",
        "that conclusion does not follow",
        "I disagree",
        "we reached different answers",
    ),
    "uncertainty": (
        "maybe",
        "I am not sure",
        "could be",
        "I cannot tell yet",
        "I need more evidence",
        "possibly",
    ),
    "surprise": (
        "wow",
        "I did not expect that",
        "well, that is new",
        "huh!",
        "that caught me off guard",
        "no way",
    ),
    "amusement": (
        "lol",
        "haha",
        "that got me",
        "I laughed at that",
        "okay, that was good",
        ":-)",
    ),
    "farewell": (
        "bye",
        "see you later",
        "until next time",
        "I am heading out",
        "good night",
        "catch you later",
    ),
    "request": (
        "please do this",
        "could you help with this?",
        "when you have a moment, check this",
        "I would like another pass",
        "could you try again?",
        "fix this for me",
    ),
    "refusal": (
        "no, I will not",
        "I am not doing that",
        "that is not happening",
        "find another way",
        "I decline",
        "stop asking",
    ),
    "relief": (
        "finally",
        "that worked",
        "what a relief",
        "we made it through",
        "the problem is gone",
        "I can breathe again",
    ),
    "confusion": (
        "what?",
        "I do not follow",
        "that makes no sense to me",
        "how does that connect?",
        "I am lost",
        "could you explain that another way?",
    ),
    "neutral": (
        "the file has three rows",
        "return the second value",
        "the meeting starts at nine",
        "sort the names by date",
        "the total is twenty four",
        "use the final column",
    ),
}


def axis_names() -> tuple[str, ...]:
    return tuple(axis.name for axis in AXES)


def pole_words() -> frozenset[str]:
    return frozenset(
        word.casefold()
        for axis in AXES
        for words in (axis.positive_words, axis.negative_words)
        for word in words
    )


def build_cases() -> tuple[EvalCase, ...]:
    cases: list[EvalCase] = []
    for axis in AXES:
        for pole, rows in (
            ("positive", axis.positive_landmarks),
            ("negative", axis.negative_landmarks),
        ):
            for variant, text in enumerate(rows, start=1):
                cases.append(
                    EvalCase(
                        case_id=f"landmark-{axis.name}-{pole}-{variant:02d}",
                        kind="landmark",
                        group=f"{axis.name}:{pole}",
                        text=text,
                        variant=variant,
                        axis=axis.name,
                        pole=pole,
                    )
                )
    for group, rows in ATLAS_FAMILIES.items():
        for variant, text in enumerate(rows, start=1):
            subset = None
            if group == "meh":
                subset = "bare" if variant <= 3 else "contextual"
            cases.append(
                EvalCase(
                    case_id=f"atlas-{group}-{variant:02d}",
                    kind="atlas",
                    group=group,
                    text=text,
                    variant=variant,
                    subset=subset,
                )
            )
    return tuple(cases)


def _words(text: str) -> set[str]:
    return {word.casefold() for word in re.findall(r"[A-Za-z]+", text)}


def canonical_payload() -> dict[str, Any]:
    return {
        "axes": [asdict(axis) for axis in AXES],
        "system_prompts": SYSTEM_PROMPTS,
        "fit_prompts": FIT_PROMPTS,
        "calibration_messages": CALIBRATION_MESSAGES,
        "cases": [asdict(case) for case in build_cases()],
    }


def data_sha256() -> str:
    encoded = json.dumps(
        canonical_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_data() -> dict[str, Any]:
    cases = build_cases()
    ids = [case.case_id for case in cases]
    if len(AXES) != 12:
        raise ValueError(f"expected 12 axes, found {len(AXES)}")
    if len(set(axis_names())) != len(AXES):
        raise ValueError("axis names must be unique")
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs must be unique")
    for axis in AXES:
        if len(axis.positive_words) != 6 or len(axis.negative_words) != 6:
            raise ValueError(f"axis {axis.name} must have six words per pole")
        if len(axis.positive_landmarks) != 4 or len(axis.negative_landmarks) != 4:
            raise ValueError(f"axis {axis.name} must have four held-out landmarks")
    if len(ATLAS_FAMILIES) != 21:
        raise ValueError("the atlas must have 21 term families")
    if any(len(rows) != 6 for rows in ATLAS_FAMILIES.values()):
        raise ValueError("every atlas family must have six variants")

    forbidden = pole_words()
    collisions = {
        case.case_id: sorted(_words(case.text) & forbidden)
        for case in cases
        if _words(case.text) & forbidden
    }
    if collisions:
        raise ValueError(f"evaluation text contains lens pole words: {collisions}")

    landmark_count = sum(case.kind == "landmark" for case in cases)
    atlas_count = sum(case.kind == "atlas" for case in cases)
    return {
        "format": "jspace-spectrum-data-v1",
        "sha256": data_sha256(),
        "axes": len(AXES),
        "pole_words": len(forbidden),
        "fit_prompts": len(FIT_PROMPTS),
        "calibration_messages": len(CALIBRATION_MESSAGES),
        "system_prompts": len(SYSTEM_PROMPTS),
        "landmark_cases": landmark_count,
        "atlas_families": len(ATLAS_FAMILIES),
        "atlas_cases": atlas_count,
        "evaluation_cases": len(cases),
        "evaluation_passes_per_model": len(cases) * len(SYSTEM_PROMPTS),
        "pole_word_collisions": collisions,
    }
