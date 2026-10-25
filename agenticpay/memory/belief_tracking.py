"""Opt-in, agent-private belief instrumentation. No oracle values enter this module."""

import hashlib
import json
import math
import re
from copy import deepcopy


STRATEGIES = (
    "unknown", "anchoring", "conceding", "holding_firm",
    "value_argument", "pressure", "other",
)
FIELDS = {
    "price_low", "price_high", "confidence", "opponent_strategy",
    "own_strategy", "evidence",
}
PRIOR_PATTERN = r"<previous_belief>.*?</previous_belief>"
INSTRUCTION = """
MENTAL MODELING INSTRUCTION (structured measurement):
Estimate the opponent's reservation price: the seller's minimum if you are the
buyer, or the buyer's maximum if you are the seller. This is an uncertain estimate,
not an observed fact. A quoted offer is NOT necessarily a reservation price.
Use the public conversation and your own private information only.
If previous_belief is non-null, it is YOUR OWN estimate from your immediately
preceding turn. Update it using new public evidence; correct mistakes freely.
If it is null, estimate from the available conversation. Never copy an estimate
merely to stay consistent. Neither the opponent nor an oracle supplies this state.
Keep ALL private estimates, confidence, strategies, and your own reservation price
out of the public message. The message must obey the role-specific price rules above.

Return exactly two blocks, with no text outside them:
<mental_model>
{"price_low": NUMBER, "price_high": NUMBER, "confidence": NUMBER,
 "opponent_strategy": "LABEL", "own_strategy": "short strategy summary",
 "evidence": "short summary of public evidence supporting this estimate"}
</mental_model>
<message>Your public negotiation message, including exactly one role price tag.</message>
The mental_model must be a JSON object, without markdown fences or extra keys.
price_low and price_high are nonnegative prices with low <= high. confidence is
a probability in [0,1] that the TRUE opponent reservation price lies in your stated
interval (not a percentage, and not general self-confidence).
opponent_strategy is one of: unknown, anchoring, conceding, holding_firm,
value_argument, pressure, other. Use unknown when evidence is insufficient.
own_strategy and evidence are brief summaries, not a step-by-step reasoning trace.
"""


class PublicMessageError(ValueError):
    """The response cannot be safely forwarded as a public negotiation message."""


def validate_belief(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("belief must have exactly the six specified fields")
    for key in ("price_low", "price_high", "confidence"):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= value["price_low"] <= value["price_high"]:
        raise ValueError("invalid price interval")
    if not 0 <= value["confidence"] <= 1:
        raise ValueError("confidence must be in [0,1]")
    if value["opponent_strategy"] not in STRATEGIES:
        raise ValueError("unknown strategy label")
    for key in ("own_strategy", "evidence"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    return deepcopy(value)


def public_offer(message, role):
    """Require a single, role-correct price, avoiding the upstream stale-price fallback."""
    tags = re.findall(r"###\s*(BUYER|SELLER)_PRICE\(\$((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\)\s*###", message)
    markers = re.findall(r"###\s*(?:BUYER|SELLER)_PRICE\b", message)
    if len(markers) != 1 or len(tags) != 1 or tags[0][0] != role.upper():
        raise PublicMessageError("expected exactly one role-correct price tag")
    price = float(tags[0][1].replace(",", ""))
    if not math.isfinite(price) or price <= 0:
        raise PublicMessageError("offer must be finite and positive")
    return price


def parse_response(response, role):
    # Closed reasoning wrappers may precede the answer; never forward their contents.
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.S | re.I).strip()
    match = re.fullmatch(
        r"<mental_model>(.*?)</mental_model>\s*<message>(.*?)</message>",
        clean, flags=re.S | re.I,
    )
    if not match:
        raise PublicMessageError("missing, malformed, or extra output blocks")
    mental, message = match.groups()
    # Reject nested blocks/unclosed reasoning instead of falling back to raw output.
    if re.search(r"</?(?:mental_model|message|think|previous_belief)\b", mental + message, re.I):
        raise PublicMessageError("nested private/public block")
    message = message.strip()
    offer = public_offer(message, role)
    try:
        belief = validate_belief(json.loads(mental))
        error = None
    except (ValueError, TypeError):
        belief, error = None, "invalid_belief_json_or_schema"
    return belief, message, offer, error


class BeliefTracker:
    """One instance per agent and episode. Baseline traces are NEVER reinserted."""

    def __init__(self, role, mode):
        if role not in ("buyer", "seller") or mode not in ("context", "maintained"):
            raise ValueError("invalid role or belief mode")
        self.role = role
        self.mode = mode
        self.reset()

    def reset(self):
        self.previous = None
        self.records = []

    def prepare_prompt(self, original_prompt, round_index):
        # Replace only the mental-model output instructions, identically in both arms.
        prefix, _ = original_prompt.split("\nMENTAL MODELING INSTRUCTION:", 1)
        prior = deepcopy(self.previous) if self.mode == "maintained" else None
        prompt = prefix + INSTRUCTION + "\n<previous_belief>" + json.dumps(prior) + "</previous_belief>\n"
        self.records.append({
            "role": self.role, "mode": self.mode, "round": round_index,
            "prior": prior, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw_response": None, "belief": None, "message": None, "offer": None,
            "error": "generation_not_completed",
        })
        return prompt

    def consume(self, response):
        record = self.records[-1]
        record["raw_response"] = response
        try:
            belief, message, offer, error = parse_response(response, self.role)
        except PublicMessageError as exc:
            record["error"] = str(exc)
            self.previous = None
            raise
        record.update(belief=belief, message=message, offer=offer, error=error)
        # An invalid immediately previous estimate becomes null; no silent stale carry.
        self.previous = deepcopy(belief) if self.mode == "maintained" else None
        return message
