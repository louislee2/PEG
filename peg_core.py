# peg_pipeline/peg_core.py
import math
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from pyvis.network import Network as PyVisNetwork
from joblib import Parallel, delayed
from tqdm import tqdm
from .peg_vis import create_simple_html_vis


def compute_node_relevance(W: dict, id_to_activity: dict):
    """
    计算节点重要性，基于边权重的绝对值之和
    """
    rel = defaultdict(float)
    for (u, v_idx), w in W.items():
        v = id_to_activity.get(v_idx, str(v_idx))
        rel[u] += abs(w)
        rel[v] += abs(w)
    return dict(rel)


def agg_attention_matrix(att_mats, method="mean"):
    if att_mats is None or len(att_mats) == 0:
        return None
    arr = np.array(att_mats)
    if arr.ndim == 3:
        if method == "mean":
            return np.mean(arr, axis=0)
        else:
            return np.max(arr, axis=0)
    elif arr.ndim == 4:
        if method == "mean":
            return np.mean(arr, axis=(0, 1))
        else:
            return np.max(arr, axis=(0, 1))
    else:
        return None


def compute_activity_attention(prefix, A, id_to_activity):
    if A is None:
        return {}
    col_sum = np.sum(A, axis=0)
    activity_scores = defaultdict(float)
    for pos, aid in enumerate(prefix):
        act = id_to_activity.get(aid, str(aid))
        activity_scores[act] += float(col_sum[pos])
    total = sum(abs(v) for v in activity_scores.values()) + 1e-12
    for k in activity_scores:
        activity_scores[k] = activity_scores[k] / total
    return dict(activity_scores)


def single_prefix_analysis(prefix, model_wrapper, id_to_activity, cfg):
    max_len = cfg.get("max_seq_len", 100) - 1
    if len(prefix) > max_len:
        prefix = prefix[:max_len]
    try:
        proba = model_wrapper.predict_proba([prefix], batch_size=cfg.get("batch_size", 64))[0]
    except Exception as e:
        print(f"预测失败: {e}")
        return []
    att_mats = model_wrapper.get_attention_matrices(prefix)
    A = agg_attention_matrix(att_mats, method=cfg.get("att_agg_method", "mean"))
    activity_scores = compute_activity_attention(prefix, A, id_to_activity) if A is not None else {}
    Pr_idx = [i for i, p in enumerate(proba) if p > cfg.get("delta_pred", 0.02)]
    Ar = []
    if activity_scores:
        acts = list(activity_scores.keys())
        scores = np.array([activity_scores[a] for a in acts])
        if len(scores) > 0:
            thr = np.percentile(scores, cfg.get("delta_att_pctile", 75))
            Ar = [acts[i] for i in range(len(acts)) if scores[i] > thr]
    if not Ar and len(prefix) > 0:
        recent_positions = min(3, len(prefix))
        recent_activities = [id_to_activity.get(prefix[-i], str(prefix[-i])) for i in range(1, recent_positions + 1)]
        Ar = list(set(recent_activities))
    results = []
    positions_by_act = defaultdict(list)
    for pos, aid in enumerate(prefix):
        act = id_to_activity.get(aid, str(aid))
        if act in Ar:
            positions_by_act[act].append(pos)
    if not positions_by_act:
        return []
    p_base = proba
    for u_act, pos_list in positions_by_act.items():
        per_v_deltas = defaultdict(list)
        for pos in pos_list:
            try:
                p_mask_input = model_wrapper.predict_with_embedding_zero(prefix, [pos])
                p_mask_att = model_wrapper.predict_with_attention_mask(prefix, [pos])
                for v in Pr_idx:
                    if v < len(p_base) and v < len(p_mask_input) and v < len(p_mask_att):
                        delta_input = float(p_base[v] - p_mask_input[v])
                        delta_att = float(p_base[v] - p_mask_att[v])
                        delta = (delta_input + delta_att) / 2.0
                        per_v_deltas[v].append(delta)
            except Exception as e:
                print(f"屏蔽预测失败: {e}")
                continue
        for v, vals in per_v_deltas.items():
            if len(vals) == 0:
                continue
            avg_d = float(np.mean(vals))
            att_score = activity_scores.get(u_act, 0.1)
            s_loc = math.copysign(abs(avg_d) * att_score, avg_d)
            sig = 1 if abs(avg_d) > cfg.get("delta_effect", 0.01) else 0
            results.append((u_act, v, s_loc, sig))
    return results


def aggregate_local_results(local_results):
    S = defaultdict(float)
    C = defaultdict(int)
    for res in local_results:
        for u_act, v, s_loc, sig in res:
            if sig == 0:
                continue
            S[(u_act, v)] += s_loc
            C[(u_act, v)] += 1
    return S, C


def normalize_and_prune(S: dict, C: dict, cfg: dict, total_prefixes: int):
    print(f" 原始候选边: {len(S)}")
    min_support = cfg.get("min_support", 5)
    min_support_ratio = cfg.get("min_support_ratio", 0.01)  # 降低默认值
    min_count = max(min_support, int(total_prefixes * min_support_ratio))
    print(f" 支持度阈值: min_support={min_support}, min_support_ratio={min_support_ratio}, min_count={min_count}")

    # 调试：输出支持度分布
    if C:
        supports = list(C.values())
        print(f" 支持度分布: 最小={min(supports)}, 最大={max(supports)}, 平均={np.mean(supports):.2f}")
        plt.figure(figsize=(8, 6))
        plt.hist(supports, bins=30, color='lightgreen', edgecolor='black')
        plt.title("Support Distribution")
        plt.xlabel("Support Count")
        plt.ylabel("Frequency")
        plt.savefig("support_distribution.png", dpi=500)
        plt.close()
        print("支持度分布图保存到: support_distribution.png")

    S_filt = {k: v for k, v in S.items() if C.get(k, 0) >= min_count}
    print(f" 满足支持度阈值 ({min_count}) 的边: {len(S_filt)}")
    if not S_filt:
        return {}
    abs_vals = np.array([abs(v) for v in S_filt.values()])
    max_abs = float(abs_vals.max()) if abs_vals.size else 1.0
    W_norm = {k: float(v / (max_abs + 1e-12)) for k, v in S_filt.items()}
    abs_norm = np.array([abs(v) for v in W_norm.values()])
    if abs_norm.size == 0:
        return {}
    original_pctile = cfg.get("prune_edge_pctile", 75)
    thr = np.percentile(abs_norm, original_pctile)
    W_pruned = {k: v for k, v in W_norm.items() if abs(v) > thr}
    attempts = 0
    while len(W_pruned) == 0 and original_pctile > 10 and attempts < 5:
        original_pctile -= 15
        thr = np.percentile(abs_norm, original_pctile)
        W_pruned = {k: v for k, v in W_norm.items() if abs(v) > thr}
        attempts += 1
        if len(W_pruned) > 0:
            print(f" 自动放宽剪枝到 pctile {original_pctile}")
    if len(W_pruned) == 0 and len(W_norm) > 0:
        k = min(20, len(W_norm))
        sorted_items = sorted(W_norm.items(), key=lambda x: abs(x[1]), reverse=True)
        W_pruned = dict(sorted_items[:k])
        print(f" 保留 top-{k} 边")
    print(f" 最终剪枝阈值: {thr:.4f}, 保留边数: {len(W_pruned)}")
    return W_pruned


def propagate_refine(W: dict, cfg: dict):
    prop_alpha = cfg.get("propagation_alpha", 0.1)
    prop_iters = cfg.get("prop_iters", 5)
    w_max = cfg.get("w_max", 1.0)
    W_new = W.copy()
    for t in range(prop_iters):
        rel = defaultdict(float)
        for (u, v), w in W_new.items():
            rel[v] += abs(w)
        max_rel = max(rel.values()) if rel else 1.0
        norm_rel = {k: math.tanh(v / (max_rel + 1e-12)) for k, v in rel.items()}
        W_new = {
            k: min(max(w + prop_alpha * w * norm_rel.get(k[1], 0.0), -w_max), w_max)
            for k, w in W_new.items()
        }
    return W_new


def export_graph(W: dict, id_to_activity: dict, out_dir: str, cfg: dict):
    G = nx.DiGraph()
    for (u_act, v_idx), w in W.items():
        v_act = id_to_activity.get(v_idx, str(v_idx))
        G.add_edge(u_act, v_act, weight=w)

    # 计算节点重要性用于动态节点大小
    node_rel = compute_node_relevance(W, id_to_activity)
    max_rel = max(node_rel.values()) if node_rel else 1.0
    node_sizes = {n: 500 + 1000 * (node_rel.get(n, 0.0) / max_rel) for n in G.nodes()}

    # NetworkX可视化
    try:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        weights = np.array([abs(d["weight"]) for _, _, d in G.edges(data=True)])
        max_w = weights.max() if weights.size else 1.0
        edge_widths = [(abs(d["weight"]) / max_w) * 5.0 for _, _, d in G.edges(data=True)]
        edge_colors = ['green' if d["weight"] > 0 else 'red' for _, _, d in G.edges(data=True)]
        nx.draw_networkx_nodes(G, pos, node_size=[node_sizes[n] for n in G.nodes()],
                               node_color='lightblue', alpha=0.8, linewidths=2, edgecolors='darkblue')
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='black')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=edge_widths,
                               edge_color=edge_colors, alpha=0.7, connectionstyle="arc3,rad=0.1")
        edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, alpha=0.8)
        plt.title("Process Explanation Graph (PEG)", fontsize=16, fontweight='bold', pad=20)
        plt.text(0.02, 0.98, f"Green=Promote, Red=Inhibit\nEdges: {G.number_of_edges()}, Nodes: {G.number_of_nodes()}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.axis('off')
        plt.tight_layout()
        png_path = f"{out_dir}/peg.png"
        plt.savefig(png_path, bbox_inches='tight', dpi=500, facecolor='white')
        plt.close()

        # 边权重分布直方图
        if weights.size > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(weights, bins=cfg.get("hist_bins", 30), color='skyblue', edgecolor='black')
            plt.title("Edge Weight Distribution", fontsize=14)
            plt.xlabel("Weight", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.tight_layout()
            hist_path = f"{out_dir}/weight_distribution.png"
            plt.savefig(hist_path, dpi=500, facecolor='white')
            plt.close()
            print(f"权重分布直方图保存到: {hist_path}")
    except Exception as e:
        print(f"NetworkX 可视化失败: {e}")

    # HTML via pyvis
    html_path = f"{out_dir}/peg.html"
    try:
        net = PyVisNetwork(height="800px", width="100%", directed=True, bgcolor="#ffffff")
        for n in G.nodes():
            size = node_sizes.get(n, 25) / 40.0
            net.add_node(n, label=n, title=str(n), size=size,
                         color={'background': '#97C2FC', 'border': '#2B7CE9'})
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 0.0)
            color = "#00AA00" if w > 0 else "#FF4444"
            width = max(1, abs(w) * 10)
            net.add_edge(u, v, value=abs(w), title=f"Weight: {w:.4f}", color=color,
                         width=width, arrows={'to': {'enabled': True, 'scaleFactor': 1.2}})
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        net.save_graph(html_path)
        print(f"PyVis 图保存到: {html_path}")
    except Exception:
        try:
            html_content = create_simple_html_vis(G, W)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"简单 HTML 图保存到: {html_path}")
        except Exception as e:
            print(f"HTML 可视化失败: {e}")

    # GEXF 输出
    gexf_path = f"{out_dir}/peg_graph.gexf"
    nx.write_gexf(G, gexf_path, encoding="utf-8")
    print(f"GEXF 图保存到: {gexf_path}")

    return gexf_path