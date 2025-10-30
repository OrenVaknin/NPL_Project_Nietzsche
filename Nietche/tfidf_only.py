import os
import re
import json
from collections import Counter, defaultdict
from typing import List, Tuple
from importlib_metadata import files
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import spacy
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GroupKFold
)
from sklearn.metrics import classification_report
#
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2
from itertools import combinations




ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZED_DIR = os.path.join(ROOT, "Tokenized_Paras")
_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])

def lemma_tokenizer(text: str):
    doc = _nlp(text)
    return [t.lemma_ for t in doc if not t.is_space and not t.is_punct and not t.like_num]

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def strip_digits(text: str) -> str:
    return re.sub(r"\d+", " ", text)

def book_id(fname: str) -> str:
    s = fname.lower()
    s = re.sub(r"\.json$", "", s)
    s = re.sub(r"\s*\(\d{4}\)\s*", " ", s) 
    s = re.sub(r"\s+para_\d+$", "", s) 
    s = re.sub(r"\s+", " ", s).strip()
    return s 

def load_texts_and_files(d: str) -> Tuple[List[str], List[str]]:
    files = sorted([f for f in os.listdir(d) if f.endswith(".json")], key=natural_key)
    texts = []
    for fname in files:
        with open(os.path.join(d, fname), "r", encoding="utf-8") as f:
            sents = json.load(f)
        texts.append(" ".join(sents))
    return files, texts

BOOK_TO_PERIOD = {
    "the birth of tragedy": 1,
    "untimely meditations": 1,
    "human all too human": 2,
    "the dawn of day": 2,
    "the gay science": 2,
    "beyond good and evil": 3,
    "thus spake zarathustra": 3,
    "the antichrist": 4,
    "the twilight of the idols": 4,
}

def base_book(fname: str) -> str:
    s = fname.lower()
    s = re.sub(r"\.json$", "", s)
    s = re.sub(r"\s*\(\d{4}\)\s*", " ", s)
    s = re.sub(r"\s+para_\d+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def derive_labels_from_books(files):
    labels, missing = [], 0
    for f in files:
        b = base_book(f)
        y = BOOK_TO_PERIOD.get(b, 0)
        labels.append(y)
        if y == 0: missing += 1
    if missing:
        print(f"[warn] {missing} files without a mapped book; they will be dropped.")
    return labels

def filter_labeled(files, texts, labels):
    keep = [i for i, y in enumerate(labels) if y in (1,2,3,4)]
    if len(keep) != len(files):
        print(f"[info] Dropping {len(files) - len(keep)} files without valid labels.")
    return [files[i] for i in keep], [texts[i] for i in keep], [labels[i] for i in keep]

def baselines_info(y):
    c = Counter(y); maj = c.most_common(1)[0][1] / len(y)
    print(f"Class counts: {dict(c)}")
    print(f"Majority-class baseline: {maj:.3f}")
def make_sample_weights(y_list, alpha=1.4):
    """Calculate sample weights based on class frequencies."""
    c = Counter(y_list)
    n = len(y_list); k = len(c)
    base = {lbl: (n / (k * c[lbl])) for lbl in c}
    if alpha != 1.0:
        base = {lbl: (w ** alpha) for lbl, w in base.items()}
    return np.array([base[y] for y in y_list], dtype=float)
# ===== Group helpers =====
def build_group_index(files: List[str], y: List[int]):
    group_ids = [book_id(f) for f in files]
    groups_dict = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups_dict[g].append(i)

    groups, g_y, g_idx = [], [], []
    for g, idxs in groups_dict.items():
        labels = [y[i] for i in idxs]
        if len(set(labels)) > 1:
            print(f"[warn] Mixed labels in book '{g}': {Counter(labels)}. Taking majority.")
        majority = Counter(labels).most_common(1)[0][0]
        groups.append(g); g_y.append(majority); g_idx.append(idxs)
    return groups, g_y, g_idx

def make_group_cv_folds(groups: List[str], g_y: List[int], g_idx: List[List[int]], n_splits_hint=5):
    per_class = Counter(g_y)
    min_groups_per_class = min(per_class.values())
    folds = []
    desc = ""
    if min_groups_per_class >= 2 and len(per_class) > 1:
        n_splits = min(n_splits_hint, min_groups_per_class)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grp_idx = np.arange(len(groups))
        for tr_g, te_g in skf.split(grp_idx, g_y):
            tr = [i for k in tr_g for i in g_idx[k]]
            te = [i for k in te_g for i in g_idx[k]]
            folds.append((tr, te))
        desc = f"{n_splits}-fold Group-Stratified CV (by book)"
    else:
        n_splits = min(n_splits_hint, len(groups))
        n_splits = max(2, n_splits)
        gkf = GroupKFold(n_splits=n_splits)
        dummy = np.zeros(len(groups))
        for tr_g, te_g in gkf.split(dummy, groups=groups):
            tr = [i for k in tr_g for i in g_idx[k]]
            te = [i for k in te_g for i in g_idx[k]]
            folds.append((tr, te))
        desc = f"{n_splits}-fold GroupKFold (no stratify; singleton classes present)"
    return folds, desc

def eval_grouped(name: str, pipe: Pipeline, X: List[str], y: List[int], files: List[str]) -> None:
    groups, g_y, g_idx = build_group_index(files, y)

    folds, desc = make_group_cv_folds(groups, g_y, g_idx, n_splits_hint=5)
    accs, f1s = [], []
    for k, (tr_idx, te_idx) in enumerate(folds, start=1):
        Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
        Xte = [X[i] for i in te_idx];  yte = [y[i] for i in te_idx]

        w = make_sample_weights(ytr, alpha=1.5) 
        pipe.fit(Xtr, ytr, **{"clf__sample_weight": w})

        yp = pipe.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average="macro"))

    print(f"\n{name} | {desc}: acc {np.mean(accs):.3f}±{np.std(accs):.3f} "
        f"| macro-F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f}")
    tr_idx, te_idx = folds[0] 
    Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
    Xte = [X[i] for i in te_idx];  yte = [y[i] for i in te_idx]
    w = make_sample_weights(ytr, alpha=1.4)
    pipe.fit(Xtr, ytr, **{"clf__sample_weight": w})
    yp = pipe.predict(Xte)
    print(f"{name} | grouped holdout (CV fold #1):\n", classification_report(yte, yp, digits=3))
    make_word_enet= Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.9,
            max_features=50000,
            token_pattern=r'(?u)\b\w\w+\b',
            sublinear_tf=True,
            preprocessor=strip_digits,
            lowercase=True
        )),
        ("clf", LogisticRegression(
            solver="saga", penalty="elasticnet", l1_ratio=0.2,
            C=1.0, max_iter=5000, class_weight="balanced", n_jobs=-1
        ))
    ])
    make_fusion_enet = Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                max_features=50000,
                token_pattern=r'(?u)\b\w\w+\b',
                sublinear_tf=True,
                preprocessor=strip_digits,
                lowercase=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 7),
                min_df=3,
                sublinear_tf=True,
                preprocessor=None, 
                lowercase=True
            )),
        ])),
        ("clf", LogisticRegression(
            solver="saga", penalty="elasticnet", l1_ratio=0.2,
            C=1.0, max_iter=6000, class_weight="balanced", n_jobs=-1
        ))
    ])
#testing pairs classification
def subset_by_labels(files, X, y, keep_labels):
    keep = [i for i, lbl in enumerate(y) if lbl in keep_labels]
    return [files[i] for i in keep], [X[i] for i in keep], [y[i] for i in keep]

def eval_grouped_scores(pipe: Pipeline, X: List[str], y: List[int], files: List[str]):
    """Like eval_grouped, but returns mean acc/F1 and a text desc, without printing."""
    groups, g_y, g_idx = build_group_index(files, y)
    folds, desc = make_group_cv_folds(groups, g_y, g_idx, n_splits_hint=5)
    accs, f1s = [], []
    for tr_idx, te_idx in folds:
        Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
        Xte = [X[i] for i in te_idx]; yte = [y[i] for i in te_idx]
        w = make_sample_weights(ytr, alpha=1.5)
        pipe.fit(Xtr, ytr, **{"clf__sample_weight": w})
        yp = pipe.predict(Xte)
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, average="macro"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.std(f1s)), desc

def run_pairwise_comparisons(pipes, files, X, y):
    """
    pipes: dict name->pipeline
    For each pair {i,j} in {(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)},
    subset data and evaluate all pipes with grouped CV.
    """
    pairs = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
    results = []  # list of dict rows

    for (a,b) in pairs:
        f_sub, X_sub, y_sub = subset_by_labels(files, X, y, {a,b})
        for pname, pipe in pipes.items():
            acc_m, acc_s, f1_m, f1_s, desc = eval_grouped_scores(pipe, X_sub, y_sub, f_sub)
            results.append({
                "pair": f"{a} vs {b}",
                "pipe": pname,
                "acc_mean": acc_m, "acc_std": acc_s,
                "f1_mean": f1_m,  "f1_std": f1_s,
                "cv": desc,
                "n_docs": len(X_sub)
            })

    print("\n=== Pairwise Period Classification (group-aware CV) ===")
    header = f"{'Pair':8} {'Model':25} {'Acc±sd':12} {'F1±sd':12} {'N':5}  CV"
    print(header)
    print("-"*len(header))
    for row in results:
        print(f"{row['pair']:8} {row['pipe']:25} "
              f"{row['acc_mean']:.3f}±{row['acc_std']:.3f} "
              f"{row['f1_mean']:.3f}±{row['f1_std']:.3f} "
              f"{row['n_docs']:5d}  {row['cv']}")
    # Highlight the hardest pair per model
    print("\nHardest pair per model (lowest macro-F1):")
    for pname in pipes.keys():
        rows = [r for r in results if r["pipe"] == pname]
        worst = min(rows, key=lambda r: r["f1_mean"])
        print(f"- {pname}: {worst['pair']} (F1 {worst['f1_mean']:.3f}, Acc {worst['acc_mean']:.3f})")
    return results

def collapse_3_and_4(y):
    """Map period 4 -> 3, keep 1,2,3 as-is; returns new label list."""
    return [3 if lbl == 4 else lbl for lbl in y]

def run_three_class_vs_four(pipes, files, X, y):
    """
    Compare 4-class setup vs 3-class (3&4 merged) using grouped CV mean metrics.
    """
    print("\n=== Multiclass Comparison: 4-class vs 3-class (merge 3&4) ===")
    for pname, pipe in pipes.items():
        # 4-class
        acc_m4, acc_s4, f1_m4, f1_s4, desc4 = eval_grouped_scores(pipe, X, y, files)
        # 3-class
        y3 = collapse_3_and_4(y)
        acc_m3, acc_s3, f1_m3, f1_s3, desc3 = eval_grouped_scores(pipe, X, y3, files)
        print(f"{pname:25} | 4-class: acc {acc_m4:.3f}±{acc_s4:.3f} | F1 {f1_m4:.3f}±{f1_s4:.3f}  ({desc4})")
        print(f"{'':25} | 3-class: acc {acc_m3:.3f}±{acc_s3:.3f} | F1 {f1_m3:.3f}±{f1_s3:.3f}  ({desc3})")

word_tfidf_logreg = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words=None,
        max_df=0.8,
        min_df=5,
        sublinear_tf=True,
        preprocessor=strip_digits
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

char_tfidf_svm = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=5,
        sublinear_tf=True,
        preprocessor=strip_digits
    )),
    ("clf", LinearSVC(class_weight="balanced"))
])

def main():
    assert os.path.isdir(TOKENIZED_DIR), f"Missing dir: {TOKENIZED_DIR}"
    files, X = load_texts_and_files(TOKENIZED_DIR)
    y = derive_labels_from_books(files)
    files, X, y = filter_labeled(files, X, y)
    assert len(X) == len(y) and len(y) > 0, "Empty dataset after filtering."
    baselines_info(y)



    eval_grouped("Word TF-IDF + LogisticRegression", word_tfidf_logreg, X, y, files)
    eval_grouped("Char TF-IDF + LinearSVC",        char_tfidf_svm,      X, y, files)
    #  Pairwise experiments: (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
    pipes = {
        "Word TF-IDF + LogisticRegression": word_tfidf_logreg,
        "Char TF-IDF + LinearSVC":          char_tfidf_svm,
    }
    pairwise_results = run_pairwise_comparisons(pipes, files, X, y)
    run_three_class_vs_four(pipes, files, X, y)


if __name__ == "__main__":
    main()
