"""Streamlit dashboard. Run: streamlit run app.py   (after python train.py <csv>)"""
import joblib, numpy as np, pandas as pd, plotly.express as px, streamlit as st
from attrition_core import *

st.set_page_config(page_title="Attrition Risk | Palo Alto Networks", layout="wide")

@st.cache_data
def load():
    return pd.read_csv("data/scored_employees.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/attrition_model.joblib")

df, art = load(), load_model()
model, lr, cols = art["model"], art["lr"], art["columns"]
COLORS = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#c0392b"}

# ---------- sidebar filters ----------
st.sidebar.title("Filters")
depts = st.sidebar.multiselect("Department", sorted(df.Department.unique()), default=sorted(df.Department.unique()))
roles = st.sidebar.multiselect("Job role", sorted(df.JobRole.unique()), default=sorted(df.JobRole.unique()))
low, high = st.sidebar.slider("Risk thresholds (Low < a, High > b)", 0.05, 0.95, (LOW_CUT, HIGH_CUT), 0.05)
f = df[df.Department.isin(depts) & df.JobRole.isin(roles)].copy()
f["RiskCategory"] = risk_category(f.AttritionProbability, low, high)
page = st.sidebar.radio("Module", ["Attrition Risk Dashboard", "Employee Risk Profile",
                                   "Department-Level Risk", "Explainability & What-If"])
st.sidebar.caption(f"Model: {art['best']} | scores are out-of-fold (honest) probabilities")

# ---------- 1. dashboard ----------
if page == "Attrition Risk Dashboard":
    st.title("Attrition Risk Dashboard")
    c = st.columns(4)
    c[0].metric("Employees", f"{len(f):,}")
    c[1].metric("High-risk", int((f.RiskCategory == "High").sum()))
    c[2].metric("Avg risk", f"{f.AttritionProbability.mean():.1%}")
    c[3].metric("Historical attrition", f"{f.Attrition.mean():.1%}")
    l, r = st.columns(2)
    cnt = f.RiskCategory.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
    cnt.columns = ["Risk", "Employees"]
    l.plotly_chart(px.bar(cnt, x="Risk", y="Employees", color="Risk", color_discrete_map=COLORS,
                          title="Risk category distribution"), use_container_width=True)
    r.plotly_chart(px.histogram(f, x="AttritionProbability", nbins=30, title="Probability distribution"),
                   use_container_width=True)
    st.subheader("Highest-risk employees")
    st.dataframe(f.sort_values("AttritionProbability", ascending=False)
                 [["EmployeeID", "Department", "JobRole", "AttritionProbability", "RiskCategory", "TopRiskDrivers"]]
                 .head(25), use_container_width=True, hide_index=True)

# ---------- 2. employee profile ----------
elif page == "Employee Risk Profile":
    st.title("Employee Risk Profile")
    eid = st.selectbox("Employee ID", f.EmployeeID.tolist())
    e = f[f.EmployeeID == eid].iloc[0]
    c = st.columns(3)
    c[0].metric("Attrition probability", f"{e.AttritionProbability:.1%}")
    c[1].metric("Risk category", e.RiskCategory)
    c[2].metric("Department / role", f"{e.Department} / {e.JobRole}")
    st.progress(float(e.AttritionProbability))
    raw_row = df[df.EmployeeID == eid].drop(columns=["EmployeeID", "AttritionProbability", "RiskCategory", "TopRiskDrivers"])
    codes = reason_codes(lr, engineer(raw_row.drop(columns="Attrition"))[cols], 5)[0]
    st.subheader("Key contributing factors")
    if codes:
        st.dataframe(pd.DataFrame(codes, columns=["Factor", "Push toward leaving"]).round(2), hide_index=True)
    else:
        st.write("No strong risk drivers.")
    with st.expander("Full employee record"):
        st.dataframe(raw_row.T, use_container_width=True)

# ---------- 3. department view ----------
elif page == "Department-Level Risk":
    st.title("Department & Role Risk")
    g = f.groupby("Department").agg(Employees=("EmployeeID", "count"), AvgRisk=("AttritionProbability", "mean"),
                                    HighRisk=("RiskCategory", lambda s: (s == "High").sum())).reset_index()
    st.plotly_chart(px.bar(g, x="Department", y="AvgRisk", text_auto=".1%", title="Average risk by department"),
                    use_container_width=True)
    role = f.groupby(["Department", "JobRole"]).agg(Employees=("EmployeeID", "count"),
            AvgRisk=("AttritionProbability", "mean"), HighRisk=("RiskCategory", lambda s: (s == "High").sum())).reset_index()
    st.plotly_chart(px.bar(role.sort_values("AvgRisk"), x="AvgRisk", y="JobRole", color="Department",
                           orientation="h", title="Average risk by job role"), use_container_width=True)
    st.dataframe(role.sort_values("AvgRisk", ascending=False).round(3), hide_index=True, use_container_width=True)

# ---------- 4. explainability + what-if ----------
else:
    st.title("Explainability & What-If")
    coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[pretty(n) for n in feature_names(lr)])
    top = coef.reindex(coef.abs().sort_values(ascending=False).index[:15]).reset_index()
    top.columns = ["Feature", "Coefficient"]
    st.plotly_chart(px.bar(top[::-1], x="Coefficient", y="Feature", orientation="h",
                           title="Top drivers (positive = raises attrition risk)"), use_container_width=True)
    st.subheader("What-if scenario")
    eid = st.selectbox("Employee", df.EmployeeID.tolist(), key="wi")
    base = df[df.EmployeeID == eid].iloc[0]
    a, b, c = st.columns(3)
    ot = a.selectbox("OverTime", ["No", "Yes"], index=["No", "Yes"].index(base.OverTime))
    travel_opts = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    trav = a.selectbox("BusinessTravel", travel_opts, index=travel_opts.index(base.BusinessTravel))
    js = b.slider("JobSatisfaction", 1, 4, int(base.JobSatisfaction))
    wlb = b.slider("WorkLifeBalance", 1, 4, int(base.WorkLifeBalance))
    stock = c.slider("StockOptionLevel", 0, 3, int(base.StockOptionLevel))
    inc = c.slider("MonthlyIncome", 1000, 20000, int(base.MonthlyIncome), 100)
    ysp = c.slider("YearsSinceLastPromotion", 0, 15, int(base.YearsSinceLastPromotion))

    def prob(row):
        return float(model.predict_proba(engineer(pd.DataFrame([row]))[cols])[:, 1][0])
    orig = base.drop(["EmployeeID", "Attrition", "AttritionProbability", "RiskCategory", "TopRiskDrivers"]).to_dict()
    mod = {**orig, "OverTime": ot, "BusinessTravel": trav, "JobSatisfaction": js, "WorkLifeBalance": wlb,
           "StockOptionLevel": stock, "MonthlyIncome": inc, "YearsSinceLastPromotion": ysp}
    p0, p1 = prob(orig), prob(mod)
    m = st.columns(3)
    m[0].metric("Current", f"{p0:.1%}")
    m[1].metric("Scenario", f"{p1:.1%}", f"{(p1 - p0):+.1%}", delta_color="inverse")
    m[2].metric("Scenario category", str(risk_category([p1])[0]))
