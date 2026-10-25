"""Offline checks for isolation, protocol, metrics, and the minimal runner."""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading

import pytest

from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.memory.belief_tracking import BeliefTracker, PRIOR_PATTERN, PublicMessageError, parse_response
from experiments.belief_tracking.evaluate import load_jsonl, score_belief, score_outcome
from experiments.belief_tracking.models import EndpointModel, MockModel
from experiments.belief_tracking.run import main, make_environment, public_history, public_state, run_episode


BELIEF = {
    "price_low": 40, "price_high": 60, "confidence": 0.75,
    "opponent_strategy": "holding_firm", "own_strategy": "private_buyer_strategy_9931",
    "evidence": "The opponent repeated its offer.",
}
CASE = {"id": "hidden_case_label", "buyer_max": 917.31, "seller_min": 431.79, "list_price": 1200.0}


def response(belief=None, role="buyer", price=50):
    return "<mental_model>" + json.dumps(BELIEF if belief is None else belief) + f"</mental_model><message>Offer ### {role.upper()}_PRICE(${price}) ###.</message>"


class FixedModel(MockModel):
    def __init__(self, role, text):
        super().__init__(role)
        self.text = text

    def generate(self, prompt, **kwargs):
        self.calls.append({"synthetic": True})
        return self.text


def test_public_channel_contains_only_message():
    belief, message, offer, error = parse_response("<think>private</think>" + response(), "buyer")
    assert belief == BELIEF and offer == 50 and error is None
    assert "private" not in message and "mental_model" not in message


@pytest.mark.parametrize("text", [
    "<mental_model>PRIVATE", response().replace("</message>", ""),
    response() + "unscoped private text", "<think>UNCLOSED" + response(),
    response().replace("Offer", "<mental_model>private</mental_model>Offer"),
    response().replace("Offer", "<previous_belief>private</previous_belief>Offer"),
    response().replace("BUYER_PRICE", "SELLER_PRICE"),
    response().replace("Offer", "### BUYER_PRICE($55) ###"),
    response().replace("Offer", "### BUYER_PRICE($NaN) ###"),
    response().replace("$50", "$0"), response().replace("$50", "$1,,000"),
])
def test_malformed_public_output_never_falls_back_to_raw_text(text):
    with pytest.raises(PublicMessageError):
        parse_response(text, "buyer")


@pytest.mark.parametrize("patch", [
    {"price_low": 70}, {"confidence": 75}, {"confidence": True},
    {"price_low": float("nan")}, {"price_high": float("inf")},
    {"opponent_strategy": "made_up_label"}, {"extra_truth": 999},
])
def test_bad_beliefs_are_missing_not_repaired(patch):
    belief, message, offer, error = parse_response(response({**BELIEF, **patch}), "buyer")
    assert belief is None and offer == 50 and message and error


@pytest.mark.parametrize("mode", ["context", "maintained"])
def test_state_input_and_invalid_turn_reset(mode):
    tracker = BeliefTracker("buyer", mode)
    prefix = "role prompt\nMENTAL MODELING INSTRUCTION:old prose"
    tracker.prepare_prompt(prefix, 0)
    tracker.consume(response())
    tracker.prepare_prompt(prefix, 1)
    assert tracker.records[-1]["prior"] == (BELIEF if mode == "maintained" else None)
    tracker.consume(response().replace(json.dumps(BELIEF), "not JSON"))
    tracker.prepare_prompt(prefix, 2)
    assert tracker.records[-1]["prior"] is None
    tracker.reset()
    assert tracker.records == [] and tracker.previous is None


@pytest.mark.parametrize("role", ["buyer", "seller"])
def test_opponent_hidden_price_noninterference_in_full_prompt(role):
    prompts = []
    for opponent_price in (431.79, 678.43):
        case = dict(CASE)
        case["seller_min" if role == "buyer" else "buyer_max"] = opponent_price
        env, observation = make_environment(case, "maintained", lambda r: MockModel(r), 6)
        agent = env.buyer_agent if role == "buyer" else env.seller_agent
        # Fixed public history, even when hidden counterfactual values differ.
        history = [
            {"role": "buyer", "content": "### BUYER_PRICE($510) ###", "round": 0},
            {"role": "seller", "content": "### SELLER_PRICE($800) ###", "round": 0},
        ]
        for round_index in (1, 2):
            agent.respond(history, {**public_state(observation), "current_round": round_index})
        prompts.append([r["prompt"] for r in agent.belief_tracker.records])
        assert "hidden_case_label" not in prompts[-1][0]
    assert prompts[0] == prompts[1]


def test_first_prompts_match_and_private_states_do_not_cross_roles():
    first_prompts = {}
    for mode in ("context", "maintained"):
        env, observation = make_environment(CASE, mode, lambda role: FixedModel(role, response(role=role)), 6)
        buyer, seller = env.buyer_agent, env.seller_agent
        seller.model.text = response({**BELIEF, "own_strategy": "private_seller_strategy_7719"}, "seller")
        bmsg = buyer.respond([], public_state(observation))
        smsg = seller.respond([{"role": "buyer", "content": bmsg, "round": 0}], public_state(observation))
        history = [
            {"role": "buyer", "content": bmsg, "round": 0},
            {"role": "seller", "content": smsg, "round": 0},
        ]
        for agent in (buyer, seller):
            first_prompts[(mode, agent.name)] = agent.belief_tracker.records[0]["prompt"]
            agent.respond(history, {**public_state(observation), "current_round": 1})
        bp, sp = buyer.belief_tracker.records[-1]["prompt"], seller.belief_tracker.records[-1]["prompt"]
        assert "431.79" not in bp and "917.31" not in sp
        assert "private_seller_strategy_7719" not in bp
        assert "private_buyer_strategy_9931" not in sp
        assert ("private_buyer_strategy_9931" in bp) == (mode == "maintained")
        assert ("private_seller_strategy_7719" in sp) == (mode == "maintained")
        env.reset()
        assert buyer.belief_tracker.previous is None and seller.belief_tracker.records == []
    for name in ("Buyer", "Seller"):
        assert first_prompts[("context", name)] == first_prompts[("maintained", name)]


def test_metadata_and_privileged_info_are_not_public_inputs():
    history = [{"role": "buyer", "content": "hello", "round": 0, "metadata": {"private": 8877}}]
    assert "8877" not in json.dumps(public_history(history))
    state = {"current_round": 1, "buyer_price": 50, "seller_price": 80, "status": "ongoing", "buyer_max": 8877}
    assert "8877" not in json.dumps(public_state(state))


def test_opt_in_preserves_original_agent_path():
    model = FixedModel("buyer", response())
    agent = BuyerAgent(model, buyer_max_price=90)
    agent.initialize({"max_price": 90})
    assert agent.respond([], {}) == "Offer ### BUYER_PRICE($50) ###."
    assert agent.belief_tracker is None


def test_known_metric_values_and_missingness():
    previous = {**BELIEF, "price_low": 30, "price_high": 50}
    result = score_belief(BELIEF, 55, 100, previous)
    assert result["mae"] == 5 and result["nmae"] == 0.05
    assert result["coverage"] == 1 and result["width_norm"] == 0.2
    assert result["confidence_brier"] == 0.0625 and result["drift_norm"] == 0.1
    assert result["interval_iou"] == pytest.approx(1 / 3)
    assert all(v is None for v in score_belief(None, 55, 100).values())
    assert score_belief(BELIEF, 55, 100)["drift_norm"] is None


def test_deal_metrics_distinguish_invalid_and_infeasible():
    case = {"buyer_max": 90, "seller_min": 50}
    episode = {"status": "agreed", "agreed_price": 70, "rounds": 3, "error": None}
    scored = score_outcome(episode, case)
    assert scored["valid_deal"] == 1 and scored["normalized_nash"] == 1
    assert scored["buyer_utility"] == 20 and scored["seller_utility"] == 20
    invalid = score_outcome({**episode, "agreed_price": 95}, case)
    assert invalid["invalid_agreement"] == 1 and invalid["buyer_utility"] == 0
    impossible = score_outcome({**episode, "status": "timeout", "agreed_price": None}, {"buyer_max": 40, "seller_min": 50})
    assert impossible["feasible"] == 0 and impossible["trade_efficiency"] is None


def test_missing_price_aborts_without_calling_opponent(tmp_path):
    models = {}

    def factory(role):
        models[role] = FixedModel(role, response().replace("### BUYER_PRICE($50) ###", "private text"))
        return models[role]

    result, rows = run_episode(CASE, "maintained", 11, factory, 6, tmp_path, "mock")
    assert result["status"] == "format_error" and result["rounds"] == 0
    assert result["public_history"] == [] and models["seller"].calls == []
    assert len(rows) == 1 and rows[0]["belief"] is None and rows[0]["error"]


def test_api_error_is_not_timeout_or_fake_response(tmp_path):
    class Broken(MockModel):
        def generate(self, prompt, **kwargs):
            self.calls.append({})
            raise ConnectionError("synthetic failure")

    result, rows = run_episode(CASE, "context", 11, lambda r: Broken(r), 6, tmp_path, "mock")
    assert result["status"] == "generation_error" and result["agreed_price"] is None
    assert rows[0]["raw_response"] is None and rows[0]["error"] == "generation_not_completed"


def test_endpoint_wire_request_and_metadata(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = {
                "id": "synthetic-http-fixture", "object": "chat.completion", "created": 0,
                "model": "fixture-model", "system_fingerprint": "fixture-fingerprint",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response()}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 12, "total_tokens": 23},
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-key")
    # The fixture is loopback-only; do not route it through the host's external proxy.
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = None
    try:
        model = EndpointModel("fixture-model", f"http://127.0.0.1:{server.server_port}/v1", 11)
        assert model.generate("only-this-prompt") == response()
        model.generate("second-isolated-prompt")
        assert requests[0]["messages"] == [{"role": "user", "content": "only-this-prompt"}]
        assert requests[1]["messages"] == [{"role": "user", "content": "second-isolated-prompt"}]
        assert requests[0]["temperature"] == 0 and requests[0]["max_tokens"] == 1024
        assert requests[0]["seed"] == 11 and requests[1]["seed"] == 1020
        assert model.calls[0]["system_fingerprint"] == "fixture-fingerprint"
        assert model.calls[0]["usage"]["total_tokens"] == 23
    finally:
        if model is not None:
            model.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_end_to_end_reproducibility_turn_order_and_probes(tmp_path):
    outputs = [tmp_path / "one", tmp_path / "two"]
    for output in outputs:
        assert main(["--backend", "mock", "--seeds", "11", "29", "--out", str(output)]) == 0
    summaries = [json.loads((p / "summary.json").read_text()) for p in outputs]
    assert summaries[0] == summaries[1]
    summary = summaries[0]
    assert summary["synthetic_only"] and summary["episodes"] == 12
    assert summary["turns"] == 112 and summary["valid_beliefs"] == 112
    assert all(not value for pair in summary["paired_deltas"] for value in pair["maintained_minus_context"].values())
    episodes = load_jsonl(outputs[0] / "episodes.jsonl")
    assert all(r["rounds"] == 6 and r["status"] == "timeout" for r in episodes if r["case_id"] == "no_overlap")
    turns = load_jsonl(outputs[0] / "turns.jsonl")
    assert all(r["prior"] is None for r in turns if r["mode"] == "context")
    for row in turns:
        if row["role"] == "seller":
            assert f"[Round {row['round']}] BUYER:" in row["prompt"]
        if row["role"] == "buyer":
            assert f"[Round {row['round']}] SELLER:" not in row["prompt"]
    probes = load_jsonl(outputs[0] / "probes.jsonl")
    assert len(probes) == 4
    for probe_id in (0, 1):
        pair = {p["variant"]: p for p in probes if p["probe_id"] == probe_id}
        a, b = pair["removed"], pair["present"]
        assert a["prior"] is None and b["prior"] is not None
        assert re.sub(PRIOR_PATTERN, "PRIOR", a["prompt"], flags=re.S) == re.sub(PRIOR_PATTERN, "PRIOR", b["prompt"], flags=re.S)
        assert a["generation"]["request_seed"] == b["generation"]["request_seed"]


def test_single_round_budget_and_no_probe_when_ineligible(tmp_path):
    output = tmp_path / "one_round"
    assert main(["--max-rounds", "1", "--out", str(output)]) == 0
    episodes = load_jsonl(output / "episodes.jsonl")
    assert all(r["rounds"] == 1 for r in episodes)
    assert load_jsonl(output / "probes.jsonl") == []
