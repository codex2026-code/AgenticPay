"""Stateless text backend plus a deterministic plumbing-only fake."""

import json
import os
import re

from agenticpay.models.base_llm import BaseLLM


class EndpointModel(BaseLLM):
    def __init__(self, model, base_url, seed, temperature=0.0, max_tokens=1024, send_seed=True):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "local-no-key"),
            base_url=base_url, max_retries=0, timeout=90.0,
        )
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.send_seed = send_seed
        self.calls = []

    def generate(self, prompt, temperature=0.0, max_tokens=None, **kwargs):
        request_seed = (self.seed + 1009 * len(self.calls)) % (2**31)
        metadata = {"request_seed": request_seed if self.send_seed else None}
        self.calls.append(metadata)
        extra = {"seed": request_seed} if self.send_seed else {}
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens, **extra,
        )
        choice = response.choices[0]
        metadata.update(
            response_id=response.id, response_model=response.model,
            system_fingerprint=getattr(response, "system_fingerprint", None),
            finish_reason=choice.finish_reason,
            usage=response.usage.model_dump() if response.usage else None,
        )
        if not choice.message.content:
            raise ValueError("endpoint returned no text content")
        return choice.message.content

    def close(self):
        self.client.close()

    def __repr__(self):
        return f"EndpointModel(model={self.model!r})"


class MockModel(BaseLLM):
    """Uses only its prompt, never the case/oracle. Deliberately ignores stored beliefs.

    This fake checks execution/isolation, NOT whether an LLM benefits from memory.
    """

    def __init__(self, role, seed=0):
        self.role, self.seed = role, seed
        self.calls = []

    def generate(self, prompt, temperature=0.0, max_tokens=None, **kwargs):
        call_index = len(self.calls)
        self.calls.append({"synthetic": True, "request_seed": self.seed + 1009 * call_index})
        history = prompt.split("Conversation History:\n", 1)[1].split("\nPlease respond naturally", 1)[0]
        turn = len(re.findall(rf"\[Round \d+\] {self.role.upper()}:", history))
        opponent = "SELLER" if self.role == "buyer" else "BUYER"
        offers = re.findall(rf"### {opponent}_PRICE\(\$(\d+(?:\.\d+)?)\) ###", history)
        last_offer = float(offers[-1]) if offers else None
        if self.role == "buyer":
            limit = float(re.search(r"Your top price is \$(\d+(?:\.\d+)?)", prompt)[1])
            price = min(limit, 45 + 7 * turn)
            if turn >= 2 and last_offer is not None and last_offer <= limit:
                price = last_offer
            low, high = (0.4 * last_offer, last_offer) if last_offer else (35, 100)
        else:
            limit = float(re.search(r"minimum acceptable price \(confidential\) is \$(\d+(?:\.\d+)?)", prompt)[1])
            price = max(limit, 108 - 12 * turn)
            if turn >= 2 and last_offer is not None and last_offer >= limit:
                price = last_offer
            low, high = (last_offer, last_offer * 1.5) if last_offer else (50, 120)
        belief = {
            "price_low": round(low, 2), "price_high": round(high, 2), "confidence": 0.6,
            "opponent_strategy": "unknown" if last_offer is None else "conceding",
            "own_strategy": "deterministic smoke-test concession",
            "evidence": "synthetic estimate using only the most recent public offer",
        }
        return (
            "<mental_model>" + json.dumps(belief) + "</mental_model>\n"
            f"<message>I offer ### {self.role.upper()}_PRICE(${price:.2f}) ###.</message>"
        )

    def close(self):
        pass

    def __repr__(self):
        return f"MockModel(role={self.role!r})"
