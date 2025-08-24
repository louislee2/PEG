# peg_pipeline/peg_vis.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_simple_html_vis(G, W):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>PEG</title>
    <style>
    body {{ font-family: Arial, sans-serif; margin:20px; }}
    table {{ border-collapse: collapse; width:100%; }}
    th, td {{ border:1px solid #ddd; padding:8px; }}
    th {{ background:#f2f2f2; }}
    .positive {{ color: green; font-weight:bold; }}
    .negative {{ color: red; font-weight:bold; }}
    </style>
    </head>
    <body>
    <h1>Process Explanation Graph (PEG)</h1>
    <div>
    <h3>Edges</h3>
    <table>
    <tr><th>Source</th><th>Target</th><th>Weight</th><th>Relation</th></tr>
    """
    edges = [(u,v,d['weight']) for u,v,d in G.edges(data=True)]
    edges.sort(key=lambda x: abs(x[2]), reverse=True)
    for u,v,w in edges:
        relation = "促进" if w>0 else "抑制"
        weight_class = "positive" if w>0 else "negative"
        html += f"<tr><td>{u}</td><td>{v}</td><td class='{weight_class}'>{w:.4f}</td><td>{relation}</td></tr>"
    html += "</table></div></body></html>"
    return html
