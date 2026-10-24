"""Train, evaluate, score. Usage: python train.py path/to/Palo_Alto_Networks.csv"""
import sys, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from attrition_core import *
warnings.filterwarnings("ignore")

path = sys.argv[1] if len(sys.argv) > 1 else "Palo_Alto_Networks.csv"
raw = pd.read_csv(path)
raw.insert(0, "EmployeeID", np.arange(1, len(raw) + 1))     # dataset has no ID column
df = engineer(raw.drop(columns="EmployeeID"))
y, X = df[TARGET], df.drop(columns=TARGET)

try:
    import xgboost; HAS_XGB = True
except ImportError:
    HAS_XGB = False
models = build_models(X, HAS_XGB)

# --- hold-out evaluation (stratified 80/20) ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rows, probs = [], {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]; probs[name] = p; pred = (p >= 0.5).astype(int)
    rows.append(dict(Model=name, Accuracy=accuracy_score(yte, pred), Precision=precision_score(yte, pred),
                     Recall=recall_score(yte, pred), F1=f1_score(yte, pred), ROC_AUC=roc_auc_score(yte, p)))
metrics = pd.DataFrame(rows).round(3)

# --- 5-fold CV (more stable on 1,470 rows) -> pick the best model ---
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = {n: cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1] for n, m in models.items()}
metrics["CV_ROC_AUC"] = [round(roc_auc_score(y, oof[n]), 3) for n in metrics["Model"]]
print(metrics.to_string(index=False)); metrics.to_csv("outputs/model_metrics.csv", index=False)
best = metrics.sort_values("CV_ROC_AUC", ascending=False).iloc[0]["Model"]
print("\nBest model (CV ROC-AUC):", best)

# --- final models fit on all data (used by the app for what-if) ---
final = models[best]; final.fit(X, y)
lr = models["Logistic Regression"]; lr.fit(X, y)
Zall = lr.named_steps["prep"].transform(X)
lr.mean_train_ = (Zall.toarray() if hasattr(Zall, "toarray") else Zall).mean(axis=0)

# --- score every employee with out-of-fold probabilities (no in-sample optimism) ---
scored = raw.copy()
scored["AttritionProbability"] = oof[best].round(4)
scored["RiskCategory"] = risk_category(oof[best])
scored["TopRiskDrivers"] = [reason_text(c) for c in reason_codes(lr, X, 3)]
scored.to_csv("data/scored_employees.csv", index=False)
joblib.dump({"model": final, "lr": lr, "best": best, "columns": list(X.columns),
             "metrics": metrics}, "models/attrition_model.joblib")

# --- charts for the paper ---
plt.figure(figsize=(6, 5))
for n, p in probs.items():
    f, t, _ = roc_curve(yte, p); plt.plot(f, t, label=f"{n} (AUC {roc_auc_score(yte, p):.2f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (hold-out)"); plt.legend(); plt.tight_layout(); plt.savefig("outputs/roc_curves.png", dpi=150); plt.close()

coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[pretty(n) for n in feature_names(lr)])
top = coef.reindex(coef.abs().sort_values(ascending=False).index[:15])[::-1]
plt.figure(figsize=(7, 6)); top.plot.barh(color=np.where(top > 0, "#c0392b", "#2980b9"))
plt.title("Logistic Regression drivers (red = raises attrition risk)"); plt.tight_layout()
plt.savefig("outputs/lr_drivers.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4)); plt.hist(oof[best], bins=30, color="#34495e")
plt.axvline(LOW_CUT, color="orange"); plt.axvline(HIGH_CUT, color="red")
plt.xlabel("Attrition probability (out-of-fold)"); plt.title("Risk score distribution")
plt.tight_layout(); plt.savefig("outputs/risk_distribution.png", dpi=150); plt.close()

print("\nRisk mix:\n", scored["RiskCategory"].value_counts(normalize=True).round(3).to_string())
print("\nActual attrition rate by risk band:\n", scored.groupby("RiskCategory")[TARGET].agg(["mean", "count"]).round(3).to_string())
print("\nBy department (mean risk):\n", scored.groupby("Department")["AttritionProbability"].mean().round(3).to_string())
print("\nTop LR drivers:\n", coef.reindex(coef.abs().sort_values(ascending=False).index[:10]).round(2).to_string())
