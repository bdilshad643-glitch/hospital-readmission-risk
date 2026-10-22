"""Train and evaluate 30-day readmission models, then explain the best one with SHAP.

    python src/train.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from data import categorical_columns, load

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SEED = 42
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def patient_split(X, y, groups, test_size, seed):
    """Split so that no patient appears on both sides (prevents leakage across encounters)."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y, groups))
    return tr, te


def logistic_model(cat, num):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.3, max_iter=3000))])


def boosting_model(cat, num):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ("num", "passthrough", num),
    ], verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
        categorical_features=[True] * len(cat) + [False] * len(num),
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def bootstrap_ci(y, p, metric, n=500, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def evaluate(name, y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, p), 4),
        "roc_auc_95ci": bootstrap_ci(y, p, roc_auc_score),
        "pr_auc": round(average_precision_score(y, p), 4),
        "threshold": round(thr, 4),
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "flagged_percent": round(100 * pred.mean(), 2),
        "brier": round(brier_score_loss(y, p), 4),
    }


def main():
    OUT.mkdir(exist_ok=True)
    X, y, groups = load()
    cat = categorical_columns(X)
    num = [c for c in X.columns if c not in cat]
    X = X[cat + num]
    print(f"{len(X):,} encounters, {groups.nunique():,} patients, "
          f"{100 * y.mean():.1f}% readmitted within 30 days")

    # train / validation (threshold tuning) / test, all split by patient
    trv, te = patient_split(X, y, groups, 0.2, SEED)
    tr_rel, va_rel = patient_split(X.iloc[trv], y.iloc[trv], groups.iloc[trv], 0.2, SEED + 1)
    tr, va = trv[tr_rel], trv[va_rel]
    X_tr, y_tr = X.iloc[tr], y.iloc[tr].to_numpy()
    X_va, y_va = X.iloc[va], y.iloc[va].to_numpy()
    X_te, y_te = X.iloc[te], y.iloc[te].to_numpy()

    models = {"Logistic regression": logistic_model(cat, num), "Gradient boosting": boosting_model(cat, num)}
    results, test_probs = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        thr = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
        p_te = model.predict_proba(X_te)[:, 1]
        test_probs[name] = p_te
        results.append(evaluate(name, y_te, p_te, thr))
        print(json.dumps(results[-1]))

    best = max(results, key=lambda r: r["roc_auc"])["model"]
    metrics = {
        "dataset": "UCI Diabetes 130-US Hospitals (1999-2008)",
        "encounters": int(len(X)),
        "patients": int(groups.nunique()),
        "readmitted_30d_percent": round(100 * float(y.mean()), 2),
        "split": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "test_results": results,
        "best_model": best,
        "note": "Threshold chosen to maximise F1 on the validation set; all metrics are on held-out test patients.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_curves(y_te, test_probs, results)
    plot_calibration(y_te, test_probs)
    plot_prior_admissions(X, y)
    explain(models["Gradient boosting"], X_te)
    print(f"done; best model by ROC AUC: {best}")


def plot_curves(y, probs, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for (name, p), r in zip(probs.items(), results):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {r['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {r['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="0.7", ls="--", label="chance")
    axes[1].axhline(y.mean(), color="0.7", ls="--", label=f"base rate {y.mean():.3f}")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve (test patients)")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall curve (test patients)")
    for ax in axes:
        ax.legend(frameon=False)
    fig.savefig(OUT / "roc_pr_curves.png")
    plt.close(fig)


def plot_calibration(y, probs):
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    for name, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", ms=4, label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="0.7", ls="--", label="perfectly calibrated")
    ax.set(xlabel="predicted risk", ylabel="observed readmission rate", title="Calibration (test patients)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)


def plot_prior_admissions(X, y):
    bucket = pd.cut(X["number_inpatient"], [-1, 0, 1, 2, 3, 5, 100], labels=["0", "1", "2", "3", "4-5", "6+"])
    rate = y.groupby(bucket, observed=False).mean() * 100
    count = y.groupby(bucket, observed=False).size()
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    ax.bar(rate.index.astype(str), rate.values, color="C0")
    for i, (r, n) in enumerate(zip(rate.values, count.values)):
        ax.text(i, r + 0.5, f"{r:.0f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="inpatient admissions in the previous year", ylabel="readmitted within 30 days (%)",
           title="Readmission risk rises with prior inpatient admissions")
    ax.set_ylim(0, rate.max() * 1.3)
    fig.savefig(OUT / "risk_by_prior_admissions.png")
    plt.close(fig)


def explain(pipe, X_te, n=3000):
    """SHAP values for the gradient-boosting model on a sample of test encounters."""
    sample = X_te.sample(n=min(n, len(X_te)), random_state=SEED)
    Z = pipe.named_steps["pre"].transform(sample)
    Z = pd.DataFrame(Z, columns=pipe.named_steps["pre"].get_feature_names_out())
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = explainer(Z)
    mean_abs = pd.Series(np.abs(sv.values).mean(0), index=Z.columns).sort_values(ascending=False)
    (OUT / "shap_importance.json").write_text(json.dumps(mean_abs.round(5).head(15).to_dict(), indent=2))

    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Mean |SHAP value|: average impact on readmission risk (log-odds)")
    plt.gcf().set_size_inches(8, 5.5)
    plt.tight_layout()
    plt.savefig(OUT / "shap_importance.png", dpi=130)
    plt.close("all")


if __name__ == "__main__":
    main()
