# peg_pipeline/peg_eval.py
import math
import os
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
import json
import random

from .peg_core import single_prefix_analysis, aggregate_local_results, compute_node_relevance

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def tvd(p, q, eps=1e-12):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * np.sum(np.abs(p - q))

def kl_div(p, q, eps=1e-12):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def fidelity_test_with_activity(model_wrapper, prefixes, id_to_activity, W, cfg, topk=1, n_samples=200, batch_size=256):
    if not W or not prefixes:
        return {"global": {"tv_diff": 0.0, "kl_diff": 0.0}}

    sample_prefixes = random.sample(prefixes, min(n_samples, len(prefixes)))
    global_diffs = {"tv_diff": [], "kl_diff": []}
    by_activity = defaultdict(lambda: {"tv_diff": [], "kl_diff": []})

    for prefix in sample_prefixes:
        try:
            p_base = model_wrapper.predict_proba([prefix], batch_size=batch_size)[0]
            top_k_acts = sorted(
                compute_node_relevance(W, id_to_activity).items(),
                key=lambda x: x[1], reverse=True
            )[:topk]
            top_k_acts = [act for act, _ in top_k_acts]

            positions_by_act = defaultdict(list)
            for pos, aid in enumerate(prefix):
                act = id_to_activity.get(aid, str(aid))
                if act in top_k_acts:
                    positions_by_act[act].append(pos)

            for act, pos_list in positions_by_act.items():
                if not pos_list:
                    continue
                try:
                    p_mask = model_wrapper.predict_with_embedding_zero(prefix, pos_list)
                    tv_diff = tvd(p_base, p_mask)
                    kl_diff = kl_div(p_base, p_mask)
                    by_activity[act]["tv_diff"].append(float(tv_diff))
                    by_activity[act]["kl_diff"].append(float(kl_diff))
                    global_diffs["tv_diff"].append(float(tv_diff))
                    global_diffs["kl_diff"].append(float(kl_diff))
                except Exception:
                    continue
        except Exception:
            continue

    global_result = {
        "tv_diff": float(np.mean(global_diffs["tv_diff"])) if global_diffs["tv_diff"] else 0.0,
        "kl_diff": float(np.mean(global_diffs["kl_diff"])) if global_diffs["kl_diff"] else 0.0
    }
    by_activity_result = {
        act: {
            "tv_diff": float(np.mean(scores["tv_diff"])) if scores["tv_diff"] else 0.0,
            "kl_diff": float(np.mean(scores["kl_diff"])) if scores["kl_diff"] else 0.0
        } for act, scores in by_activity.items()
    }

    return {"global": global_result, "by_activity": by_activity_result}

def sign_consistency(local_results, id_to_activity, threshold_pct=0.75):
    edge_signs = defaultdict(list)
    for res in local_results:
        for u_act, v, s_loc, sig in res:
            if sig == 0:
                continue
            v_act = id_to_activity.get(v, str(v))
            edge_signs[(u_act, v_act)].append(1 if s_loc > 0 else -1)

    sign_cons = {}
    for edge, signs in edge_signs.items():
        if not signs:
            continue
        pos_ratio = sum(1 for s in signs if s > 0) / len(signs)
        sign_cons[edge] = pos_ratio if pos_ratio >= threshold_pct or pos_ratio <= (1 - threshold_pct) else 0.5

    return sign_cons

def permutation_test_edges(prefixes, model_wrapper, local_results, id_to_activity, W, cfg, n_permutations=50, n_jobs=8):
    if not W or not prefixes:
        return {}

    # 采样前缀
    sample_size = min(cfg.get("perm_sample_size", 500), len(prefixes))
    sample_prefixes = random.sample(prefixes, sample_size)

    edge_counts = defaultdict(int)
    n_jobs = min(n_jobs, len(sample_prefixes)) if n_jobs > 0 else 1

    def permute_prefix_analysis(prefixes_batch):
        results = []
        for prefix in prefixes_batch:
            permuted = prefix.copy()
            random.shuffle(permuted)
            results.append(single_prefix_analysis(permuted, model_wrapper, id_to_activity, cfg))
        return results

    # 按批次处理
    batch_size = cfg.get("batch_size", 256)
    prefix_batches = [sample_prefixes[i:i + batch_size] for i in range(0, len(sample_prefixes), batch_size)]

    for _ in range(n_permutations):
        if n_jobs == 1:
            perm_results = []
            for batch in prefix_batches:
                perm_results.extend(permute_prefix_analysis(batch))
        else:
            perm_results = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes='100M')(
                delayed(permute_prefix_analysis)(batch) for batch in prefix_batches
            )
            perm_results = [item for sublist in perm_results for item in sublist]
        S_perm, _ = aggregate_local_results(perm_results)
        for edge in S_perm:
            edge_counts[edge] += 1

    results = {}
    for edge, w in W.items():
        v_act = id_to_activity.get(edge[1], str(edge[1]))
        edge_key = (edge[0], v_act)
        p_value = edge_counts.get(edge_key, 0) / n_permutations
        results[edge_key] = float(p_value)

    return results

def bootstrap_test_edges(prefixes, model_wrapper, local_results, id_to_activity, W, cfg, n_bootstraps=50):
    if not W or not prefixes:
        return {}

    sample_size = min(cfg.get("perm_sample_size", 500), len(prefixes))
    sample_prefixes = random.sample(prefixes, sample_size)

    edge_counts = defaultdict(int)
    n = len(sample_prefixes)

    for _ in range(n_bootstraps):
        sample_indices = random.choices(range(n), k=n)
        sample_batch = [sample_prefixes[i] for i in sample_indices]
        sample_results = [single_prefix_analysis(p, model_wrapper, id_to_activity, cfg) for p in sample_batch]
        S_sample, _ = aggregate_local_results(sample_results)
        for edge in S_sample:
            edge_counts[edge] += 1

    results = {}
    for edge, w in W.items():
        v_act = id_to_activity.get(edge[1], str(edge[1]))
        edge_key = (edge[0], v_act)
        p_value = edge_counts.get(edge_key, 0) / n_bootstraps
        results[edge_key] = float(p_value)

    return results

def tune_hyperparams(build_peg_pipeline, csv_path, out_dir, cfg, case_col, activity_col, time_col, hparams=None, run_eval=True):
    hparams = hparams or {
        "min_support_ratio": [0.005, 0.01, 0.05],
        "delta_effect": [0.001, 0.005, 0.01]
    }
    best_score = -float('inf')
    best_cfg = cfg.copy()

    for min_support_ratio in hparams.get("min_support_ratio", [0.01]):
        for delta_effect in hparams.get("delta_effect", [0.005]):
            cfg_temp = cfg.copy()
            cfg_temp["min_support_ratio"] = min_support_ratio
            cfg_temp["delta_effect"] = delta_effect
            out_temp = os.path.join(out_dir, f"min_support_{min_support_ratio}_delta_{delta_effect}")
            build_peg_pipeline(csv_path, out_temp, cfg_temp, case_col, activity_col, time_col, run_eval=run_eval)
            with open(os.path.join(out_temp, "peg_eval.json"), "r") as f:
                eval_data = json.load(f)
            score = eval_data["fidelity"]["global"]["tv_diff"]
            if score > best_score:
                best_score = score
                best_cfg["min_support_ratio"] = min_support_ratio
                best_cfg["delta_effect"] = delta_effect

    print(
        f"最佳参数: min_support_ratio={best_cfg['min_support_ratio']}, delta_effect={best_cfg['delta_effect']}, 得分: {best_score:.4f}")
    return best_cfg