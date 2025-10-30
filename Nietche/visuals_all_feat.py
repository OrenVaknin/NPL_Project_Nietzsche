import os, json, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.abspath(__file__))
TEXTFEATS = os.path.join(ROOT, "TextFeatures")
LABELS = os.path.join(TEXTFEATS, "label_chaps_to_periods.json")        # :contentReference[oaicite:8]{index=8}
GROUPS = os.path.join(TEXTFEATS, "label_chaps_by_book.json")           # :contentReference[oaicite:9]{index=9}

def load_json(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return json.load(f)

def main():
    full_path = os.path.join(TEXTFEATS, "paras_full_feature_matrix.json")
    if not os.path.exists(full_path):
        raise SystemExit("paras_full_feature_matrix.json not found.first run DataParsing.py ")

    X = np.array(load_json(full_path), dtype=float)
    X = np.delete(X, 14, axis=1)
    y = np.array(load_json(LABELS))
    keep = np.where(np.isin(y, [1,2,3,4]))[0]
    X = X[keep]; y = y[keep]

    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)

    readability = np.array(load_json(os.path.join(TEXTFEATS, "readability_scores.json")))  # FK grade :contentReference[oaicite:11]{index=11}
    ttr        = np.array(load_json(os.path.join(TEXTFEATS, "lexical_diversity_scores.json")))  # TTR :contentReference[oaicite:12]{index=12}

    func = np.array(load_json(os.path.join(TEXTFEATS, "function_word_profiles.json")))  # list[list] :contentReference[oaicite:13]{index=13}
    var = func.var(axis=0)
    top_idx = np.argsort(var)[::-1][:30]
    func_top = func[:, top_idx]

    # POS bigrams
    pos_bigrams = np.array(load_json(os.path.join(TEXTFEATS, "pos_bigram_frequencies.json")))  # list[list] :contentReference[oaicite:14]{index=14}
    pos_mean_by_period = []
    for cls in [1,2,3,4]:
        pos_mean_by_period.append(pos_bigrams[y==cls].mean(axis=0))
    pos_mean_by_period = np.array(pos_mean_by_period)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Features PCA", "Readability & TTR", "Function words heatmap (top 30)", "POS bigrams by period")
    )

    for cls, name in [(1,"Period 1"),(2,"Period 2"),(3,"Period 3"),(4,"Period 4")]:
        idx = np.where(y==cls)[0]
        fig.add_trace(go.Scatter(x=XY[idx,0], y=XY[idx,1], mode="markers", name=name, hoverinfo="skip"), row=1, col=1)

    for cls, name in [(1,"P1"),(2,"P2"),(3,"P3"),(4,"P4")]:
        idx = np.where(y==cls)[0]
        fig.add_trace(go.Box(y=readability[idx], name=f"{name} FK", boxmean='sd'), row=1, col=2)
    for cls, name in [(1,"P1"),(2,"P2"),(3,"P3"),(4,"P4")]:
        idx = np.where(y==cls)[0]
        fig.add_trace(go.Box(y=ttr[idx], name=f"{name} TTR", boxmean='sd'), row=1, col=2)

    fig.add_trace(go.Heatmap(z=func_top[:200], coloraxis="coloraxis"), row=2, col=1)  

    for i, name in enumerate(["P1","P2","P3","P4"]):
        fig.add_trace(go.Scatter(y=pos_mean_by_period[i], mode="lines", name=f"{name} POS-avg"), row=2, col=2)

    fig.update_layout(title="Feature Visualizations", height=900, width=1200, coloraxis=dict(showscale=False))
    fig.write_html("features_viz.html", include_plotlyjs="cdn")
    print("Saved: features_viz.html")

if __name__ == "__main__":
    main()
