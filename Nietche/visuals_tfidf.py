import os, json, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


from tfidf_only import load_texts_and_files, derive_labels_from_books, TOKENIZED_DIR, strip_digits, run_pairwise_comparisons, word_tfidf_logreg, char_tfidf_svm
pipe1 = word_tfidf_logreg
pipe2 = char_tfidf_svm
def top_terms_per_class(X, y, feature_names, k=20):
    tops = {}
    for cls in sorted(set(y)):
        chi, p = chi2(X, (np.array(y)==cls).astype(int))
        idx = np.argsort(chi)[::-1][:k]
        tops[cls] = [(feature_names[i], float(chi[i])) for i in idx]
    return tops
def append_html_block(target_html_path: str, block_html: str):
    """מכניס את הבלוק לפני </body>. אם אין קובץ, יוצר שלד בסיסי."""
    if os.path.exists(target_html_path):
        with open(target_html_path, "r", encoding="utf-8") as f:
            html = f.read()
        if "</body>" in html:
            html = html.replace("</body>", block_html + "\n</body>")
        else:
            html = html + "\n" + block_html
    else:
        html = f"<!doctype html><html><head><meta charset='utf-8'><title>Results</title></head><body>{block_html}</body></html>"
    with open(target_html_path, "w", encoding="utf-8") as f:
        f.write(html)

def figure_pairwise_f1_bars(rows):
    """
    rows: הפלט של run_pairwise_comparisons (רשימת dict-ים).
    מצייר עמודות Macro-F1 לכל זוג תקופות לכל מודל.
    """
    pair_order = ["1 vs 2", "1 vs 3", "1 vs 4", "2 vs 3", "2 vs 4", "3 vs 4"]
    models = sorted({r.get("pipe", r.get("model", "")) for r in rows} - {""})

    series = {}
    for m in models:
        f1_map = {r["pair"]: r["f1_mean"] for r in rows if r.get("pipe", r.get("model","")) == m}
        series[m] = [f1_map.get(p, None) for p in pair_order]

    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Bar(name=m, x=pair_order, y=series[m]))
    fig.update_layout(
        title="Pairwise Macro-F1 by Model",
        barmode="group",
        xaxis_title="Pair",
        yaxis_title="Macro-F1",
        height=420,
        margin=dict(l=40, r=20, t=60, b=60),
    )
    return fig

def add_pairwise_chart_via_import(target_html_path="tfidf_viz.html"):
    files, docs = load_texts_and_files(TOKENIZED_DIR)
    y_raw = derive_labels_from_books(files)
    keep = [i for i,t in enumerate(y_raw) if t in (1,2,3,4)]
    docs = [docs[i] for i in keep]; y = [y_raw[i] for i in keep]

    pipes = {
        "Word TF-IDF + LogisticRegression": pipe1,
        "Char TF-IDF + LinearSVC":          pipe2,
    }
    rows = run_pairwise_comparisons(pipes, files=[files[i] for i in keep], X=docs, y=y)

    fig = figure_pairwise_f1_bars(rows)
    snippet = """
<section id="pairwise-bars" style="margin-top:24px;">
  <h2>Pairwise Results – Macro-F1</h2>
  {div}
</section>
""".format(div=fig.to_html(full_html=False, include_plotlyjs=False))
    append_html_block(target_html_path, snippet)
    print(f"Appended pairwise chart to {target_html_path}")
def _subset_by_labels_dense(X, y, groups, keep):
    mask = np.isin(y, list(keep))
    return X[mask], y[mask], groups[mask]

def _grouped_metrics_logreg(X, y, groups, n_splits=2):
    """
    Group-aware CV by 'groups' (e.g., book). Returns acc±sd and macro-F1±sd.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    n_splits = max(2, min(n_splits, len(uniq)))
    gkf = GroupKFold(n_splits=n_splits)

    accs, f1s = []
    # Scaler+LR pipeline; class imbalance handled
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=10000, class_weight="balanced", solver="lbfgs")
    )
    for tr, te in gkf.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average="macro"))

    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s)), f"GroupKFold(n_splits={n_splits})"

def run_pairwise_handcrafted_logreg(X_full, y, groups, feature_sets=None, n_splits=2):
    """
    Pairwise classification for handcrafted features using ONLY LogisticRegression.
    Returns: list[dict] with keys:
      pair, pipe, feature_set, acc_mean, acc_std, f1_mean, f1_std, n_rows, cv
    - X_full: np.ndarray (dense), shape (N, D)
    - y: labels in {1,2,3,4}
    - groups: group ids (e.g., book) for group-CV
    - feature_sets: optional dict{name -> X_slice}; if None, uses {"all features": X_full}
    """
    if feature_sets is None:
        feature_sets = {"all features": X_full}

    pairs = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
    rows = []

    # keep only valid labels once
    valid_mask = np.isin(y, [1,2,3,4])
    yv = y[valid_mask]; gv = groups[valid_mask]
    Xv_full = X_full[valid_mask]

    for feat_name, Xs in feature_sets.items():
        # align slice with the same valid mask if Xs is X_full-derived
        Xv = Xs[valid_mask] if Xs.shape[0] == X_full.shape[0] else Xs

        for a, b in pairs:
            Xp, yp, gp = _subset_by_labels_dense(Xv, yv, gv, {a, b})
            n = int(len(yp))
            acc_m, acc_s, f1_m, f1_s, desc = _grouped_metrics_logreg(Xp, yp, gp, n_splits=n_splits)
            rows.append({
                "pair": f"{a} vs {b}",
                "pipe": "LogisticRegression (handcrafted)",
                "feature_set": feat_name,
                "acc_mean": acc_m, "acc_std": acc_s,
                "f1_mean": f1_m,  "f1_std": f1_s,
                "n_rows": n,
                "cv": desc
            })

    # short console summary
    print("\n=== Pairwise handcrafted (LogisticRegression) ===")
    for r in rows:
        print(f"{r['feature_set']:28} | {r['pair']:7} | acc {r['acc_mean']:.3f}±{r['acc_std']:.3f} "
              f"| F1 {r['f1_mean']:.3f}±{r['f1_std']:.3f} | N={r['n_rows']:5d} | {r['cv']}")
    return rows

def main():
    assert os.path.isdir(TOKENIZED_DIR), f"Missing dir: {TOKENIZED_DIR}"
    files, docs = load_texts_and_files(TOKENIZED_DIR)   # :contentReference[oaicite:4]{index=4}
    y_raw = derive_labels_from_books(files)              # :contentReference[oaicite:5]{index=5}
    keep = [i for i,t in enumerate(y_raw) if t in (1,2,3,4)]
    docs = [docs[i] for i in keep]; y = [y_raw[i] for i in keep]

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9,
                          sublinear_tf=True, preprocessor=strip_digits, lowercase=True)
    X = vec.fit_transform(docs)
    feats = np.array(vec.get_feature_names_out())
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X.toarray())

    tops = top_terms_per_class(X, y, feats, k=15)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("TF-IDF: מסמכים ב-PCA", "Top terms per period"))
    for cls, name in [(1,"Period 1"),(2,"Period 2"),(3,"Period 3"),(4,"Period 4")]:
        idx = [i for i,t in enumerate(y) if t==cls]
        fig.add_trace(go.Scatter(x=XY[idx,0], y=XY[idx,1], mode="markers", name=name, hoverinfo="skip"),
                      row=1, col=1)

    bars = []
    for cls in (1,2,3,4):
        words, scores = zip(*tops.get(cls, []))
        bars.append(go.Bar(x=list(words), y=list(scores), name=f"Period {cls}"))
    for b in bars:
        fig.add_trace(b, row=1, col=2)

    fig.update_layout(title="TF-IDF Visualization", barmode="group", height=600, width=1200)
    fig.write_html("tfidf_viz.html", include_plotlyjs="cdn")
    print("Saved: tfidf_viz.html")
    add_pairwise_chart_via_import("tfidf_viz.html")


if __name__ == "__main__":
    main()
