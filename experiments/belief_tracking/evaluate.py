"""Offline metrics only: this module never constructs or calls an agent/model."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def average(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def midpoint(belief):
    return (belief["price_low"] + belief["price_high"]) / 2


def interval_iou(a, b):
    intersection = max(0, min(a["price_high"], b["price_high"]) - max(a["price_low"], b["price_low"]))
    union = max(a["price_high"], b["price_high"]) - min(a["price_low"], b["price_low"])
    return intersection / union if union else 1.0


def score_belief(belief, truth, scale, previous=None):
    keys = ("mae", "nmae", "coverage", "width_norm", "confidence_brier", "drift_norm", "interval_iou", "strategy_switch")
    if belief is None:
        return dict.fromkeys(keys)
    error = abs(midpoint(belief) - truth)
    covered = int(belief["price_low"] <= truth <= belief["price_high"])
    return {
        "mae": error, "nmae": error / scale, "coverage": covered,
        "width_norm": (belief["price_high"] - belief["price_low"]) / scale,
        "confidence_brier": (belief["confidence"] - covered) ** 2,
        "drift_norm": abs(midpoint(belief) - midpoint(previous)) / scale if previous else None,
        "interval_iou": interval_iou(belief, previous) if previous else None,
        "strategy_switch": int(belief["opponent_strategy"] != previous["opponent_strategy"]) if previous else None,
    }


def score_outcome(episode, case):
    buyer, seller, price = case["buyer_max"], case["seller_min"], episode["agreed_price"]
    agreed = episode["status"] == "agreed"
    valid = agreed and price is not None and seller <= price <= buyer
    spread = buyer - seller
    buyer_utility = buyer - price if valid else 0.0
    seller_utility = price - seller if valid else 0.0
    return {
        "agreement": int(agreed), "valid_deal": int(valid),
        "invalid_agreement": int(agreed and not valid), "feasible": int(spread >= 0),
        "rounds": episode["rounds"], "final_price": price,
        "buyer_utility": buyer_utility, "seller_utility": seller_utility,
        "normalized_nash": 4 * buyer_utility * seller_utility / spread**2 if spread > 0 else None,
        "trade_efficiency": int(valid) if spread > 0 else None,
        "episode_error": int(episode["error"] is not None),
    }


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_directory(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in json.loads((directory / "oracle.json").read_text(encoding="utf-8"))}
    turns = load_jsonl(directory / "turns.jsonl")
    episodes = load_jsonl(directory / "episodes.jsonl")
    probes = load_jsonl(directory / "probes.jsonl")
    trajectories = defaultdict(list)
    scored_turns, previous = [], {}
    for row in turns:
        case = cases[row["case_id"]]
        truth = case["seller_min"] if row["role"] == "buyer" else case["buyer_max"]
        key = (row["case_id"], row["seed"], row["mode"], row["role"])
        metrics = score_belief(row["belief"], truth, case["list_price"], previous.get(key))
        previous[key] = row["belief"]  # Never bridge over a missing/invalid turn.
        scored = {
            **{k: row[k] for k in ("case_id", "seed", "mode", "role", "round")},
            "true_opponent_price": truth, "parse_valid": row["belief"] is not None,
            "error": row["error"], **metrics,
        }
        if row["offer"] is None:
            scored["own_limit_violation"] = None
        elif row["role"] == "buyer":
            scored["own_limit_violation"] = int(row["offer"] > case["buyer_max"])
        else:
            scored["own_limit_violation"] = int(row["offer"] < case["seller_min"])
        scored_turns.append(scored)
        trajectories[key].append(scored)
    with (directory / "scored_turns.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored_turns:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    episode_scores = []
    for episode in episodes:
        case = cases[episode["case_id"]]
        result = {k: episode[k] for k in ("case_id", "seed", "mode", "status")}
        result.update(score_outcome(episode, case))
        for role in ("buyer", "seller"):
            rows = trajectories[(episode["case_id"], episode["seed"], episode["mode"], role)]
            result[f"{role}_turns"] = len(rows)
            result[f"{role}_valid_beliefs"] = sum(r["parse_valid"] for r in rows)
            result[f"{role}_valid_transitions"] = sum(r["drift_norm"] is not None for r in rows)
            result[f"{role}_parse_failure_rate"] = average([int(not r["parse_valid"]) for r in rows])
            for metric in ("mae", "nmae", "coverage", "width_norm", "confidence_brier", "drift_norm", "interval_iou", "strategy_switch", "own_limit_violation"):
                result[f"{role}_{metric}"] = average([r[metric] for r in rows])
            result[f"{role}_final_nmae"] = rows[-1]["nmae"] if rows else None
        episode_scores.append(result)
    with (directory / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_scores[0]) if episode_scores else [])
        writer.writeheader()
        writer.writerows(episode_scores)

    pair_map = defaultdict(dict)
    for row in episode_scores:
        pair_map[(row["case_id"], row["seed"])][row["mode"]] = row
    excluded = {"case_id", "seed", "mode", "status", "feasible"}
    metric_names = [k for k in episode_scores[0] if k not in excluded] if episode_scores else []
    pairs = []
    for (case_id, seed), arms in pair_map.items():
        if set(arms) != {"context", "maintained"}:
            continue
        a, b = arms["context"], arms["maintained"]
        pairs.append({
            "case_id": case_id, "seed": seed,
            "both_without_episode_error": not (a["episode_error"] or b["episode_error"]),
            "maintained_minus_context": {
                k: b[k] - a[k] if a[k] is not None and b[k] is not None else None
                for k in metric_names
            },
        })

    arm_means, arm_counts = {}, {}
    for mode in ("context", "maintained"):
        group = [r for r in episode_scores if r["mode"] == mode]
        # One trajectory mean per episode: long conversations do not get extra weight.
        arm_means[mode] = {k: average([r[k] for r in group]) for k in metric_names}
        arm_counts[mode] = {k: sum(r[k] is not None for r in group) for k in metric_names}

    probe_map = defaultdict(dict)
    for row in probes:
        probe_map[row["probe_id"]][row["variant"]] = row
    probe_metrics = []
    for probe_id, variants in probe_map.items():
        removed, present = variants["removed"], variants["present"]
        case = cases[removed["case_id"]]
        truth = case["seller_min"] if removed["role"] == "buyer" else case["buyer_max"]
        a = score_belief(removed["belief"], truth, case["list_price"])
        b = score_belief(present["belief"], truth, case["list_price"])
        both_valid = removed["belief"] is not None and present["belief"] is not None
        probe_metrics.append({
            "probe_id": probe_id, "case_id": removed["case_id"], "role": removed["role"],
            "seed": removed["seed"], "round": removed["round"], "both_valid": both_valid,
            "removed_error": removed["error"], "present_error": present["error"],
            "midpoint_shift_norm": abs(midpoint(present["belief"]) - midpoint(removed["belief"])) / case["list_price"] if both_valid else None,
            "nmae_present_minus_removed": b["nmae"] - a["nmae"] if both_valid else None,
            "offer_shift": present["offer"] - removed["offer"] if present["offer"] is not None and removed["offer"] is not None else None,
            "strategy_changed": int(present["belief"]["opponent_strategy"] != removed["belief"]["opponent_strategy"]) if both_valid else None,
        })
    summary = {
        "backend": manifest["backend"], "synthetic_only": manifest["synthetic_only"],
        "episodes": len(episode_scores), "turns": len(turns),
        "valid_beliefs": sum(r["parse_valid"] for r in scored_turns),
        "arm_means": arm_means, "arm_metric_episode_counts": arm_counts,
        "paired_deltas": pairs, "frozen_prefix_probes": probe_metrics,
        "interpretation": "Descriptive pilot; low drift is not necessarily good; no claim about hidden internal beliefs.",
    }
    dump_json(directory / "summary.json", summary)

    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"

    lines = [
        "# AgenticPay 信念追踪实验", "",
        "**确定性假模型流程测试；以下数值不是 LLM 研究结果。**" if manifest["synthetic_only"] else "**真实接口运行的描述性小样本结果；不作显著性或内部机制结论。**",
        "", f"运行状态：{manifest['status']}；{len(episodes)} 局；{len(turns)} 次 live 调用；有效信念 {summary['valid_beliefs']}/{len(turns)}；重放探针对数 {len(probe_metrics)}。",
        "", "每项先在单局内按角色平均，再跨局平均。NA 不补零；每项有效局数见 summary.json。失败局仍保留，须同时检查失败率和逐局配对差值。",
        "", "| 指标 | context | maintained |", "|---|---:|---:|",
    ]
    display = {
        "buyer_nmae": "买方估计 NMAE", "seller_nmae": "卖方估计 NMAE",
        "buyer_final_nmae": "买方末轮 NMAE", "seller_final_nmae": "卖方末轮 NMAE",
        "buyer_coverage": "买方区间覆盖率", "seller_coverage": "卖方区间覆盖率",
        "buyer_width_norm": "买方归一化区间宽度", "seller_width_norm": "卖方归一化区间宽度",
        "buyer_confidence_brier": "买方区间事件 Brier", "seller_confidence_brier": "卖方区间事件 Brier",
        "buyer_drift_norm": "买方跨轮漂移", "seller_drift_norm": "卖方跨轮漂移",
        "buyer_interval_iou": "买方相邻区间 IoU", "seller_interval_iou": "卖方相邻区间 IoU",
        "buyer_strategy_switch": "买方策略判断切换率", "seller_strategy_switch": "卖方策略判断切换率",
        "buyer_parse_failure_rate": "买方信念格式失败率", "seller_parse_failure_rate": "卖方信念格式失败率",
        "valid_deal": "有效成交率（含不可成交场景，逐局结果需分开看）",
        "invalid_agreement": "越界成交率", "rounds": "完成轮数",
        "buyer_utility": "买方实现效用", "seller_utility": "卖方实现效用",
        "normalized_nash": "归一化 Nash（仅正剩余场景）",
        "trade_efficiency": "可成交场景成交率", "episode_error": "运行/协议失败率",
    }
    lines += [f"| {label} | {fmt(arm_means['context'][key])} | {fmt(arm_means['maintained'][key])} |" for key, label in display.items()]
    lines += [
        "", "逐局成交价、可成交性、双方效用见 episode_metrics.csv；配对差值为 maintained − context。",
        "", "固定对话重放：", "", "| 探针 | 角色 | 中点变化 / 标价 | NMAE差值（有状态−无状态） | 报价差值 |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [f"| {r['probe_id']} | {r['role']} | {fmt(r['midpoint_shift_norm'])} | {fmt(r['nmae_present_minus_removed'])} | {fmt(r['offer_shift'])} |" for r in probe_metrics]
    lines += [
        "", "解释限制：漂移小可能是机械复述；覆盖率必须结合宽度；策略标签没有客观真值。",
        "区间由模型自己选择，Brier 仅检验所报区间事件的置信度，不能据此声称整体校准改善。",
        "live 两组的公开对话可能分叉；固定前缀探针只观察所提供状态的输入敏感性。",
        "seed/temperature=0 不保证远端推理完全确定，服务端版本和 fingerprint 见原始记录。",
        "这些观测不能证明模型隐式表征中不存在或存在信念。", "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    evaluate_directory(parser.parse_args().directory)


if __name__ == "__main__":
    main()
