# peg_pipeline/main.py
import argparse
import json
import sys
import os
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

from .data_utils import DEFAULTS, seed_all, ensure_dir, load_csv_prefixes, split_prefixes_for_training
from .model_utils import build_user_model, ModelWrapper
from .peg_core import single_prefix_analysis, aggregate_local_results, normalize_and_prune, propagate_refine, export_graph, compute_node_relevance
from .peg_eval import fidelity_test_with_activity, sign_consistency, permutation_test_edges, bootstrap_test_edges, tune_hyperparams
from .prediction import train_prediction_model, predict_next_activities, evaluate_with_accuracy

warnings.filterwarnings("ignore", message=".*nested tensors.*")

def build_peg_pipeline(csv_path: str, out_dir: str, cfg: dict, case_col: str, activity_col: str, time_col: str,
                       train_model=False, run_prediction=False, run_eval=False, eval_cfg=None):
    seed_all(cfg.get("seed", 42))
    ensure_dir(out_dir)

    print("\n=== PEG Pipeline 参数检查 ===")
    print(f"delta_att_pctile  = {cfg.get('delta_att_pctile')}")
    print(f"delta_pred       = {cfg.get('delta_pred')}")
    print(f"delta_effect     = {cfg.get('delta_effect')}")
    print(f"min_support      = {cfg.get('min_support')}")
    print(f"min_support_ratio= {cfg.get('min_support_ratio')}")
    print(f"prune_edge_pctile= {cfg.get('prune_edge_pctile')}")
    print(f"propagation_alpha= {cfg.get('propagation_alpha')}")
    print(f"hist_bins        = {cfg.get('hist_bins')}")
    print(f"train_model      = {train_model}")
    print(f"run_prediction   = {run_prediction}")

    print("加载数据...")
    prefixes, activity_to_id, id_to_activity = load_csv_prefixes(csv_path, case_col, activity_col, time_col, cfg)
    print(f" 前缀数: {len(prefixes)}, 活动数: {len(activity_to_id)}")

    # 改进真实标签提取：从完整序列生成测试前缀和真实标签
    true_next_activities = []
    test_prefixes = []
    for prefix in prefixes:
        if len(prefix) >= 2:
            test_prefixes.append(prefix[:-1])  # 去掉最后一个作为前缀
            true_next_activities.append(id_to_activity.get(prefix[-1], str(prefix[-1])))  # 真实下一活动

    if train_model:
        print("开始训练预测模型...")
        try:
            train_prefixes, val_prefixes = split_prefixes_for_training(
                prefixes, train_ratio=cfg.get("train_split", 0.8), seed=cfg.get("seed", 42)
            )
            print(f" 训练集: {len(train_prefixes)}, 验证集: {len(val_prefixes)}")
            wrapper = train_prediction_model(train_prefixes, activity_to_id, id_to_activity, cfg, out_dir)
            if run_prediction:
                print("生成预测结果...")
                predictions = predict_next_activities(wrapper, test_prefixes, true_next_activities, top_k=5)
                eval_results = evaluate_with_accuracy(predictions)
                print("预测性能 (PEG):", eval_results)
                pred_path = os.path.join(out_dir, "predictions_peg.json")
                with open(pred_path, "w", encoding="utf-8") as f:
                    json.dump({"predictions": predictions, "eval_results": eval_results}, f, indent=2, ensure_ascii=False)
                print(f"PEG预测结果保存到: {pred_path}")
                print("\n=== 预测示例 ===")
                for i, pred in enumerate(predictions[:3]):
                    print(f"前缀 {i + 1}: {' -> '.join(pred['prefix'])}")
                    print("预测的下一活动:")
                    for j, (activity, prob) in enumerate(pred['predictions'][:3]):
                        print(f"  {j + 1}. {activity}: {prob:.4f}")
                    print(f"真实下一活动: {pred['true_next']}")
                    print()
        except ImportError as e:
            print(f"导入预测模块失败: {e}")
            print("使用未训练模型进行PEG分析...")
            user_model, pad_id = build_user_model(activity_to_id, cfg)
            wrapper = ModelWrapper(user_model, activity_to_id, id_to_activity, pad_id)
    else:
        print("构建未训练模型...")
        user_model, pad_id = build_user_model(activity_to_id, cfg)
        wrapper = ModelWrapper(user_model, activity_to_id, id_to_activity, pad_id)

    print("开始PEG分析...")
    if test_prefixes:
        try:
            test_proba = wrapper.predict_proba([test_prefixes[0]], batch_size=cfg.get("batch_size", 64))[0]
            print(f" 测试预测维度: {test_proba.shape}")
        except Exception as e:
            print(f" 模型预测测试失败: {e}")
            return

    n_jobs = min(cfg.get("n_jobs", 8), len(prefixes)) if cfg.get("n_jobs", 8) > 0 else 1
    if n_jobs < cfg.get("n_jobs", 8):
        print(f"⚠️ n_jobs 已调整为 {n_jobs} (prefix 数量不足)")

    print(f" 开始局部分析（并行 jobs = {n_jobs}）...")
    if n_jobs == 1:
        local_results = []
        for prefix in tqdm(prefixes, desc="分析前缀"):
            local_results.append(single_prefix_analysis(prefix, wrapper, id_to_activity, cfg))
    else:
        local_results = Parallel(n_jobs=n_jobs)(
            delayed(single_prefix_analysis)(prefix, wrapper, id_to_activity, cfg) for prefix in tqdm(prefixes)
        )

    S, C = aggregate_local_results(local_results)
    print(f" 聚合候选边: {len(S)}")

    W_pruned = normalize_and_prune(S, C, cfg, total_prefixes=len(prefixes))
    print(f" 剪枝后: {len(W_pruned)}")

    if cfg.get("propagate", True):
        W_refined = propagate_refine(W_pruned, cfg)
    else:
        W_refined = W_pruned

    export_graph(W_refined, id_to_activity, out_dir, cfg)

    if run_eval:
        print("开始评估...")
        node_rel = compute_node_relevance(W_refined, id_to_activity)
        fid_res = fidelity_test_with_activity(wrapper, test_prefixes, id_to_activity, W_refined, cfg,
                                              topk=eval_cfg.get("topk", 3),
                                              n_samples=eval_cfg.get("fidelity_samples", 200),
                                              batch_size=cfg.get("batch_size", 256))
        sign_cons = sign_consistency(local_results, id_to_activity, threshold_pct=eval_cfg.get("sign_threshold", 0.75))
        perm_res = permutation_test_edges(test_prefixes, wrapper, local_results, id_to_activity, W_refined, cfg,
                                          n_permutations=eval_cfg.get("n_permutations", 20),
                                          n_jobs=min(cfg.get("n_jobs", 8), 8))
        boot_res = bootstrap_test_edges(test_prefixes, wrapper, local_results, id_to_activity, W_refined, cfg,
                                        n_bootstraps=eval_cfg.get("n_bootstraps", 20))

        eval_out = {
            "node_relevance": node_rel,
            "fidelity": fid_res,
            "sign_consistency": {f"{k[0]}->{k[1]}": v for k, v in sign_cons.items()},
            "permutation": {f"{k[0]}->{k[1]}": v for k, v in perm_res.items()},
            "bootstrap": {f"{k[0]}->{k[1]}": v for k, v in boot_res.items()}
        }
        with open(os.path.join(out_dir, "peg_eval.json"), "w", encoding="utf-8") as f:
            json.dump(eval_out, f, indent=2, ensure_ascii=False)
        print("评价结果保存: peg_eval.json")

    print("完成。输出目录:", out_dir)
    return out_dir

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="PEG pipeline with prediction support")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="./peg_output")
    parser.add_argument("--case_col", type=str, default=DEFAULTS["case_col"])
    parser.add_argument("--activity_col", type=str, default=DEFAULTS["activity_col"])
    parser.add_argument("--time_col", type=str, default=DEFAULTS["time_col"])
    parser.add_argument("--n_jobs", type=int, default=DEFAULTS["n_jobs"])
    parser.add_argument("--delta_att_pctile", type=float, default=DEFAULTS["delta_att_pctile"])
    parser.add_argument("--delta_pred", type=float, default=DEFAULTS["delta_pred"])
    parser.add_argument("--min_support", type=int, default=DEFAULTS["min_support"])
    parser.add_argument("--min_support_ratio", type=float, default=DEFAULTS["min_support_ratio"])
    parser.add_argument("--prune_edge_pctile", type=float, default=DEFAULTS["prune_edge_pctile"])
    parser.add_argument("--propagation_alpha", type=float, default=DEFAULTS["propagation_alpha"])
    parser.add_argument("--hist_bins", type=int, default=DEFAULTS["hist_bins"], help="Number of bins for weight distribution histogram")
    parser.add_argument("--device", type=str, default=DEFAULTS["device"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--n_permutations", type=int, default=20)
    parser.add_argument("--fidelity_samples", type=int, default=20)
    parser.add_argument("--n_bootstraps", type=int, default=20)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--tune-hparams", action="store_true")
    parser.add_argument("--train-model", action="store_true")
    parser.add_argument("--run-prediction", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULTS["num_epochs"])
    parser.add_argument("--train-split", type=float, default=DEFAULTS["train_split"])

    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update({
        "n_jobs": args.n_jobs,
        "delta_att_pctile": args.delta_att_pctile,
        "delta_pred": args.delta_pred,
        "min_support": args.min_support,
        "min_support_ratio": args.min_support_ratio,
        "prune_edge_pctile": args.prune_edge_pctile,
        "propagation_alpha": args.propagation_alpha,
        "hist_bins": args.hist_bins,
        "device": args.device,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "train_split": args.train_split,
    })

    eval_cfg = {
        "n_permutations": args.n_permutations,
        "fidelity_samples": args.fidelity_samples,
        "n_bootstraps": args.n_bootstraps,
        "topk": args.topk
    }

    if args.tune_hparams:
        from .peg_eval import tune_hyperparams
        tune_hyperparams(build_peg_pipeline, args.csv, args.out, cfg,
                         args.case_col, args.activity_col, args.time_col, run_eval=True)
        sys.exit(0)

    if not os.path.exists(args.csv):
        print("CSV 不存在:", args.csv)
        return

    build_peg_pipeline(args.csv, args.out, cfg, args.case_col, args.activity_col, args.time_col,
                       train_model=args.train_model, run_prediction=args.run_prediction,
                       run_eval=args.run_eval, eval_cfg=eval_cfg)

if __name__ == "__main__":
    parse_args_and_run()