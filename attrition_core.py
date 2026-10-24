"""Shared logic: feature engineering, pipelines, risk bands, reason codes."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Attrition"
SAT_COLS = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
            "JobInvolvement", "WorkLifeBalance"]
LOW_CUT, HIGH_CUT = 0.30, 0.60


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four engineered features from the project brief."""
    d = df.copy()
    d["IncomePerExpYear"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)      # income-to-experience
    d["PromotionDelayRatio"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["LongNoPromotion"] = (d["YearsSinceLastPromotion"] >= 3).astype(int)         # promotion-delay flag
    d["EngagementScore"] = d[SAT_COLS].mean(axis=1)                                # engagement composite
    d["WorkloadStress"] = ((d["OverTime"] == "Yes") & (d["WorkLifeBalance"] <= 2)).astype(int)
    return d


def split_cols(X: pd.DataFrame):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return cat, num


def _prep(cat, num, scale):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler() if scale else "passthrough", num)])


def build_models(X: pd.DataFrame, xgb_available: bool = False) -> dict:
    cat, num = split_cols(X)
    models = {
        "Logistic Regression": Pipeline([("prep", _prep(cat, num, True)),
            ("clf", LogisticRegression(class_weight="balanced", C=0.1, max_iter=3000))]),
        "Random Forest": Pipeline([("prep", _prep(cat, num, False)),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
    }
    if xgb_available:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([("prep", _prep(cat, num, False)),
            ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=5.2, eval_metric="logloss",
                random_state=42, n_jobs=-1))])
    else:  # sklearn fallback so the project runs without xgboost
        models["Gradient Boosting"] = Pipeline([("prep", _prep(cat, num, False)),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=200,
                class_weight="balanced", random_state=42))])
    return models


def risk_category(p, low=LOW_CUT, high=HIGH_CUT):
    p = np.asarray(p)
    return np.where(p < low, "Low", np.where(p <= high, "Medium", "High"))


def feature_names(pipe: Pipeline):
    return list(pipe.named_steps["prep"].get_feature_names_out())


def pretty(name: str):
    return name.split("__", 1)[-1].replace("_", " = ", 1) if name.startswith("cat__") else name.split("__", 1)[-1]


def reason_codes(lr_pipe: Pipeline, raw_rows: pd.DataFrame, k: int = 3):
    """Top-k factors pushing each employee's risk UP vs. the average employee.
    contribution_j = coef_j * (x_j - mean_j) in the model's (scaled / one-hot) space."""
    prep, clf = lr_pipe.named_steps["prep"], lr_pipe.named_steps["clf"]
    Z = prep.transform(raw_rows)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    mean = lr_pipe.mean_train_ if hasattr(lr_pipe, "mean_train_") else Z.mean(axis=0)
    contrib = clf.coef_[0] * (Z - mean)
    raw_names = feature_names(lr_pipe)
    names = [pretty(n) for n in raw_names]
    is_cat = np.array([n.startswith("cat__") for n in raw_names])
    contrib = np.where(is_cat & (Z == 0), 0.0, contrib)   # only the category the employee actually has
    out = []
    for i in range(len(raw_rows)):
        order = np.argsort(-contrib[i])[:k]
        out.append([(names[j], float(contrib[i, j])) for j in order if contrib[i, j] > 0])
    return out


def reason_text(codes):
    return "; ".join(n for n, _ in codes) if codes else "No strong risk drivers"
