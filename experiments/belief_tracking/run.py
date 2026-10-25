"""Run from the repository root: python -m experiments.belief_tracking.run --help."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess

from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.envs.single_buyer_product_seller.Task1_basic_price_negotiation import Task1BasicPriceNegotiation
from agenticpay.memory.belief_tracking import BeliefTracker, PRIOR_PATTERN, PublicMessageError, parse_response
from .models import EndpointModel, MockModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODES = ("context", "maintained")
ROLES = ("buyer", "seller")


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path, data):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def load_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("provide nonempty cases with unique IDs")
    for case in cases:
        if set(case) != {"id", "buyer_max", "seller_min", "list_price"}:
            raise ValueError("case must contain only id and the three prices")
        for key in ("buyer_max", "seller_min", "list_price"):
            if type(case[key]) not in (int, float) or not math.isfinite(case[key]) or case[key] <= 0:
                raise ValueError("case prices must be finite and positive")
    return cases


def public_history(history):
    # Explicit projection: never forward metadata, traces, private contexts, or info.
    return [{key: msg[key] for key in ("role", "content", "round")} for msg in history]


def public_state(observation):
    return {key: observation[key] for key in ("current_round", "buyer_price", "seller_price", "status")}


def make_environment(case, mode, factory, max_rounds):
    # The factory accepts only a role. No joint case or opponent limit reaches a model.
    buyer = BuyerAgent(
        model=factory("buyer"), buyer_max_price=case["buyer_max"],
        belief_tracker=BeliefTracker("buyer", mode),
    )
    seller = SellerAgent(
        model=factory("seller"), seller_min_price=case["seller_min"],
        belief_tracker=BeliefTracker("seller", mode),
    )
    # Upstream checks current_round >= max_rounds BEFORE incrementing (zero-based).
    # Subtract one here so N means exactly N completed buyer+seller rounds.
    env = Task1BasicPriceNegotiation(
        buyer, seller, max_rounds=max_rounds - 1, initial_seller_price=case["list_price"],
        buyer_max_price=case["buyer_max"], seller_min_price=case["seller_min"],
        price_tolerance=0.0,
    )
    # Neither case ID nor overlap condition is placed in either agent's context.
    observation, _ = env.reset(
        user_requirement="Negotiate the total price of one standard desk lamp.",
        product_info={"name": "Standard desk lamp", "price": case["list_price"]},
    )
    return env, observation


def run_episode(case, mode, seed, factory, max_rounds, output, backend):
    env, observation = make_environment(case, mode, factory, max_rounds)
    key = {"case_id": case["id"], "seed": seed, "mode": mode, "backend": backend}
    status, error = "timeout", None
    rows = []
    transcript = []
    try:
        for round_index in range(max_rounds):
            history = public_history(observation["conversation_history"])
            state = public_state(observation)
            actions = {}
            for role, agent in (("buyer", env.buyer_agent), ("seller", env.seller_agent)):
                before = len(agent.belief_tracker.records)
                try:
                    message = agent.respond(deepcopy(history), deepcopy(state))
                finally:
                    if len(agent.belief_tracker.records) > before:
                        row = deepcopy(agent.belief_tracker.records[-1])
                        row.update(key)
                        row["generation"] = deepcopy(agent.model.calls[-1]) if agent.model.calls else {}
                        append_jsonl(output / "turns.jsonl", row)
                        rows.append(row)
                actions[role] = message
                public = {"role": role, "content": message, "round": round_index}
                history.append(public)
                transcript.append(public)
                state[f"{role}_price"] = agent.belief_tracker.records[-1]["offer"]
            # Exactly one environment step per full round. Reward/info NEVER go to agents.
            with redirect_stdout(io.StringIO()):
                observation, _, terminated, truncated, _ = env.step(actions["buyer"], actions["seller"])
            if terminated or truncated:
                status = "agreed" if terminated else "timeout"
                break
    except PublicMessageError as exc:
        status, error = "format_error", str(exc)
    except Exception as exc:
        # Retain attempted prompts/partial traces; never convert API errors into fake replies.
        status, error = "generation_error", type(exc).__name__
    finally:
        env.buyer_agent.model.close()
        env.seller_agent.model.close()
    result = {
        **key, "status": status, "error": error,
        "rounds": env.current_round, "agent_calls": len(rows),
        "agreed_price": env.state.agreed_price, "public_history": transcript,
    }
    append_jsonl(output / "episodes.jsonl", result)
    return result, rows


def run_probes(rows, count, factory, output):
    """Re-query frozen context-arm prefixes with a prior present vs removed.

    These outputs never enter live negotiations or each other's prompts.
    """
    previous, candidates = {}, []
    for row in rows:
        if row["mode"] != "context":
            continue
        key = (row["case_id"], row["seed"], row["role"])
        prior = previous.get(key)
        if prior is not None and row["round"] >= 1:
            candidates.append((row, prior))
        previous[key] = row["belief"]
    # First eligible round per role/case; avoid spending all probes on one trajectory.
    selected, seen = [], set()
    for row, prior in candidates:
        key = (row["case_id"], row["seed"], row["role"])
        if key not in seen and len(selected) < count:
            seen.add(key)
            selected.append((row, prior))
    failures = 0
    for index, (row, prior) in enumerate(selected):
        variants = ["removed", "present"] if index % 2 == 0 else ["present", "removed"]
        for variant in variants:
            injected = prior if variant == "present" else None
            prompt = re.sub(PRIOR_PATTERN, lambda _: "<previous_belief>" + json.dumps(injected) + "</previous_belief>", row["prompt"], flags=re.S)
            request_seed = row["generation"].get("request_seed")
            model = factory(row["role"], row["seed"] if request_seed is None else request_seed)
            result = {
                "probe_id": index, "variant": variant,
                **{k: row[k] for k in ("case_id", "seed", "role", "round", "backend")},
                "prior": injected, "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "belief": None, "message": None, "offer": None, "error": None,
                "raw_response": None,
            }
            try:
                response = model.generate(prompt)
                result["raw_response"] = response
                belief, message, offer, error = parse_response(response, row["role"])
                result.update(belief=belief, message=message, offer=offer, error=error)
            except Exception as exc:
                result["error"] = type(exc).__name__
            finally:
                result["generation"] = deepcopy(model.calls[-1]) if model.calls else {}
                model.close()
            failures += int(result["error"] is not None)
            append_jsonl(output / "probes.jsonl", result)
    return len(selected), failures


def source_snapshot():
    paths = [
        ROOT / "agenticpay/agents/buyer_agent.py", ROOT / "agenticpay/agents/seller_agent.py",
        ROOT / "agenticpay/agents/base_agent.py", ROOT / "agenticpay/memory/conversation_memory.py",
        ROOT / "agenticpay/memory/belief_tracking.py",
        ROOT / "agenticpay/envs/single_buyer_product_seller/Task1_basic_price_negotiation.py",
        *HERE.glob("*.py"),
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return {"git_commit": commit.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()), "sha256": hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--model", help="Model ID served by the supplied compatible endpoint")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--omit-seed", action="store_true", help="Explicitly disable endpoint seed if unsupported")
    parser.add_argument("--probe-count", type=int, default=2, help="Frozen-prefix pairs, two extra calls each")
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.json")
    parser.add_argument("--out", type=Path, required=True, help="New evaluator-only output directory")
    args = parser.parse_args(argv)
    if args.max_rounds < 1 or args.probe_count < 0 or args.max_tokens < 1:
        parser.error("rounds/tokens must be positive; probe-count must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("duplicate seeds would create duplicate pairs")
    if args.backend == "openai" and not args.model:
        parser.error("--model is required for a real endpoint")
    cases = load_cases(args.scenarios)
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    for filename in ("turns.jsonl", "episodes.jsonl", "probes.jsonl"):
        (args.out / filename).touch(mode=0o600)
    # Oracle is evaluator-only. No agent, prompt helper, or backend reads this file.
    write_json(args.out / "oracle.json", cases)
    packages = {}
    for name in ("numpy", "pillow", "requests", "loguru", "openai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "status": "running", "backend": args.backend,
        "synthetic_only": args.backend == "mock", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model if args.backend != "mock" else "deterministic-plumbing-fake",
        "base_url": args.base_url if args.backend != "mock" else None,
        "seeds": args.seeds, "max_rounds": args.max_rounds,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "send_seed": not args.omit_seed, "probe_count_requested": args.probe_count,
        "python": platform.python_version(), "packages": packages, "source": source_snapshot(),
        "protocol": "buyer_then_seller; zero tolerance; upstream crossing-offer settlement",
    }
    write_json(args.out / "manifest.json", manifest)

    def factory(role, seed):
        if args.backend == "mock":
            return MockModel(role, seed)
        return EndpointModel(args.model, args.base_url, seed, args.temperature, args.max_tokens, not args.omit_seed)

    all_rows, errors = [], 0
    for case in cases:
        for seed in args.seeds:
            modes = list(MODES)
            random.Random(str((case["id"], seed))).shuffle(modes)
            for mode in modes:
                # Same role/turn seed in matched arms, fresh per-agent client/state each time.
                episode, rows = run_episode(
                    case, mode, seed, lambda role: factory(role, seed + ROLES.index(role)),
                    args.max_rounds, args.out, args.backend,
                )
                all_rows.extend(rows)
                errors += int(episode["error"] is not None)
                print(f"{args.backend}: {case['id']} seed={seed} {mode}: {episode['status']} ({episode['rounds']} rounds)")
    probes, probe_errors = run_probes(all_rows, args.probe_count, factory, args.out)
    belief_errors = sum(row["error"] is not None for row in all_rows)
    manifest.update(
        status="complete_with_errors" if errors or belief_errors or probe_errors else "complete",
        episodes=2 * len(cases) * len(args.seeds), episode_errors=errors,
        turn_errors=belief_errors, probe_pairs=probes, probe_errors=probe_errors,
    )
    write_json(args.out / "manifest.json", manifest)
    # Offline scoring is the FIRST point where recorded beliefs meet oracle labels.
    from .evaluate import evaluate_directory
    evaluate_directory(args.out)
    print(f"Saved {args.out / 'report.md'}")
    return int(bool(errors or belief_errors or probe_errors))


if __name__ == "__main__":
    raise SystemExit(main())
