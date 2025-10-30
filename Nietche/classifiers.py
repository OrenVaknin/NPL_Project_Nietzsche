import os, re
import json
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import tree
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
TEXTFEATURES_DIR = os.path.join(ROOT, "TextFeatures")
TOKENIZED_DIR    = os.path.join(ROOT, "Tokenized_Paras")
MATRIX_PATH = os.path.join(TEXTFEATURES_DIR, "paras_full_feature_matrix.json")

def eval_grouped(name, X, y, groups, clf, n_splits=5):
    """Return mean accuracy using GroupKFold by book."""
    # ensure arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)

    uniq_groups = np.unique(groups)
    n_splits = max(2, min(n_splits, len(uniq_groups)))

    gkf = GroupKFold(n_splits=n_splits)
    accs = []

    for tr_idx, te_idx in gkf.split(X, y, groups):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), clf)
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xte)
        accs.append(accuracy_score(yte, yp))

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    print(f"{name}: {mean_acc:.3f} ± {std_acc:.3f} (GroupKFold, n_splits={n_splits})")
    return mean_acc


def run_classifier_grouped(textfeatures_dir, labels_path, groups_path):
    # load feature matrix + labels + groups
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        X_full = np.array(json.load(f), dtype=float)
    with open(labels_path, "r") as f:
        y = np.array(json.load(f))
    with open(groups_path, "r") as f:
        groups = np.array(json.load(f))

    assert len(X_full) == len(y) == len(groups), "Row count mismatch among features/labels/groups."

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=42, max_iter=10000)
    logreg = LogisticRegression(random_state=42, max_iter=10000)
    tree = DecisionTreeClassifier(random_state=42)

    def run_all(name_prefix, X):
        eval_grouped(f"{name_prefix} | LogisticRegression", X, y, groups, logreg)
        eval_grouped(f"{name_prefix} | MLP",               X, y, groups, mlp)
        eval_grouped(f"{name_prefix} | DecisionTree",      X, y, groups, tree)

    # all features
    run_all("all features", X_full)

    # without punctuation (last col)
    run_all("all features except punctuation", X_full[:, :-1])

    # slices
    run_all("POS only",            X_full[:, :9])
    run_all("sentiment only",      X_full[:, 9:10])
    run_all("tense only",          X_full[:, 10:13])
    run_all("entities only",       X_full[:, 13:14])
    run_all("punctuation only",    X_full[:, 14:15])

    # example: remove columns 16 and 6
    cols_to_remove = [6, 16]
    X_some = np.delete(X_full, cols_to_remove, axis=1)
    run_all("all features except columns 6 and 16", X_some)


def run_classifier_without_noise_grouped(textfeatures_dir, labels_path, groups_path):
    with open(labels_path, "r") as f:
        y = np.array(json.load(f))
    with open(groups_path, "r") as f:
        groups = np.array(json.load(f))
    with open(os.path.join(textfeatures_dir, "paras_full_feature_matrix.json"), "r") as f:
        X_full = np.array(json.load(f), dtype=float)
    noisy_cols = list(range(16, 26)) + [26]
    X_clean = np.delete(X_full, noisy_cols, axis=1)

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=42, max_iter=10000)
    logreg = LogisticRegression(random_state=42, max_iter=10000)
    tree = DecisionTreeClassifier(random_state=42)

    eval_grouped("NO-NOISE | MLP",          X_clean, y, groups, mlp)
    eval_grouped("NO-NOISE | DecisionTree", X_clean, y, groups, tree)
    eval_grouped("NO-NOISE | Logistic",     X_clean, y, groups, logreg)


def run_classifier_with_new_features_grouped(textfeatures_dir, labels_path, groups_path):
    with open(labels_path, "r") as f:
        y = np.array(json.load(f))
    with open(groups_path, "r") as f:
        groups = np.array(json.load(f))
    with open(os.path.join(textfeatures_dir, "paras_full_feature_matrix_updated.json"), "r") as f:
        X_updated = np.array(json.load(f), dtype=float)

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=42, max_iter=10000)
    logreg = LogisticRegression(random_state=42, max_iter=10000)
    tree = DecisionTreeClassifier(random_state=42)

    eval_grouped("UPDATED | MLP",              X_updated, y, groups, mlp)
    eval_grouped("UPDATED | DecisionTree",     X_updated, y, groups, tree)
    eval_grouped("UPDATED | LogisticRegression", X_updated, y, groups, logreg)
    


def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    textfeatures_dir = os.path.join(ROOT_DIR, 'TextFeatures')
    labels_path = os.path.join(textfeatures_dir, "label_chaps_to_periods.json")
    groups_path = os.path.join(textfeatures_dir, "label_chaps_by_book.json")  # book label per row

    print("=== GROUPED BY BOOK: ORIGINAL FEATURES ===")
    run_classifier_grouped(textfeatures_dir, labels_path, groups_path)

    print("\n=== GROUPED BY BOOK: WITHOUT NOISY FEATURES ===")
    run_classifier_without_noise_grouped(textfeatures_dir, labels_path, groups_path)

    # optional: updated features
    updated_path = os.path.join(textfeatures_dir, "paras_full_feature_matrix_updated.json")
    if os.path.exists(updated_path):
        print("\n=== GROUPED BY BOOK: NEW LINGUISTIC FEATURES ===")
        run_classifier_with_new_features_grouped(textfeatures_dir, labels_path, groups_path)
    else:
        print("\n[info] Updated feature matrix not found; run extract_new_features.py + update_feature_matrix.py first.")

if __name__ == "__main__":
    main()
