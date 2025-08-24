# peg_pipeline/data_utils.py
import os
import random
import math
import numpy as np
import pandas as pd
import torch

DEFAULTS = {
    "case_col": "Case ID",
    "activity_col": "Activity",
    "time_col": "Complete Timestamp",
    "pad_id": 0,
    "min_prefix_len": 2,
    "max_prefix_len": None,
    "sample_prefix_limit": None,
    "att_agg_method": "mean",
    "delta_att_pctile": 50,
    "delta_pred": 0.01,
    "min_support": 2,
    "min_support_ratio": 0.005,  # 降低默认值
    "delta_effect": 0.005,      # 降低默认值
    "propagate": True,
    "prop_alpha": 0.1,
    "prop_iters": 5,
    "w_max": 1.0,
    "prune_edge_pctile": 50,
    "propagation_alpha": 0.05,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "n_jobs": 8,
    "random_seed": 42,
    "batch_size": 128,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 512,
    "dropout": 0.05,
    "max_seq_len": 50,
    "learning_rate": 0.001,
    "num_epochs": 3,
    "train_split": 0.8,
    "weight_decay": 0.01,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.5,
    "early_stopping_patience": 2,
    "save_best_model": True,
    "hist_bins": 30,
}

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_csv_prefixes(csv_path: str, case_col: str, activity_col: str, time_col: str, cfg: dict):
    df = pd.read_csv(csv_path)
    case_count = len(df[case_col].unique())
    traces = []
    all_acts = set()
    for _, group in df.groupby(case_col):
        acts = list(group.sort_values(time_col)[activity_col].astype(str))
        if len(acts) < 2:
            continue
        traces.append(acts)
        all_acts.update(acts)

    acts_list = sorted(list(all_acts))
    activity_to_id = {a: i + 1 for i, a in enumerate(acts_list)}
    id_to_activity = {i + 1: a for i, a in enumerate(acts_list)}
    pad_id = cfg.get("pad_id", 0)

    prefixes = []
    max_model_len = cfg.get("max_seq_len", 100) - 1
    for acts in traces:
        act_ids = [activity_to_id[a] for a in acts]
        max_len = len(act_ids) - 1
        max_len = min(max_len, max_model_len)
        if cfg.get("max_prefix_len") is not None:
            max_len = min(max_len, cfg["max_prefix_len"])
        for k in range(cfg.get("min_prefix_len", 1), max_len + 1):
            prefixes.append(act_ids[:k])

    print(f" 案例总数: {case_count:,}, 有效案例: {len(traces):,}, 唯一活动: {len(all_acts)}")
    print(
        f" 前缀总数: {len(prefixes):,}, 长度范围: {min(len(p) for p in prefixes) if prefixes else 0} - {max(len(p) for p in prefixes) if prefixes else 0}")

    if cfg.get("sample_prefix_limit") and len(prefixes) > cfg["sample_prefix_limit"]:
        prefixes = random.sample(prefixes, cfg["sample_prefix_limit"])
        print(f" 采样前缀到 {len(prefixes)} 条")

    return prefixes, activity_to_id, id_to_activity

def split_prefixes_for_training(prefixes, train_ratio=0.8, seed=42):
    random.seed(seed)
    shuffled = prefixes.copy()
    random.shuffle(shuffled)

    split_point = int(len(shuffled) * train_ratio)
    train_prefixes = shuffled[:split_point]
    test_prefixes = shuffled[split_point:]

    return train_prefixes, test_prefixes