"""HIT140 Assessment 3 - Objective 2 only.
Reproducible modelling for 104 match-level and 208 team-match observations.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

raw = pd.read_csv(ROOT / "208_world_cup_dataset.csv")
assert len(raw) == 208 and raw["match_id"].nunique() == 104
assert raw.groupby("match_id").size().eq(2).all()
assert raw.duplicated(["match_id", "team"]).sum() == 0

# Cross-check tournament totals against FIFA's published 104 matches / 308 goals.
assert int(raw["goals_scored"].sum()) == 308

# Add opponent information by reversing each two-row match.
opp_cols = ["team", "fifa_rank", "elo_rating", "squad_avg_age",
            "squad_total_value_eur_m", "goals_scored"]
opp = raw[["match_id"] + opp_cols].copy()
opp = opp.rename(columns={c: f"opponent_{c}" for c in opp_cols})
team = raw.merge(opp, on="match_id")
team = team[team["team"] != team["opponent_team"]].copy()
assert len(team) == 208

# Objective 2.1: one row per actual match. Exactly eight pre-match predictors.
# Difference features preserve the pre-match comparison while reducing the severe
# multicollinearity produced by entering both rankings/ratings as separate terms.
for frame in (team,):
    frame["fifa_rank_advantage"] = frame["opponent_fifa_rank"] - frame["fifa_rank"]
    frame["elo_advantage"] = frame["elo_rating"] - frame["opponent_elo_rating"]
    frame["age_difference"] = frame["squad_avg_age"] - frame["opponent_squad_avg_age"]
    frame["value_advantage_eur_m"] = (frame["squad_total_value_eur_m"] -
                                      frame["opponent_squad_total_value_eur_m"])
    frame["host_advantage"] = frame["host_nation"] - frame["opponent_host_nation"]

home = team[team["home_away"] == 1].copy()
home["goal_difference"] = home["goals_scored"] - home["opponent_goals_scored"]
p21 = [
    "fifa_rank_advantage", "elo_advantage", "age_difference",
    "value_advantage_eur_m", "host_nation", "opponent_host_nation",
    "stage_numeric", "match_number"
]
d21 = home[["match_id", "team", "opponent_team"] + p21 +
           ["goals_scored", "opponent_goals_scored", "goal_difference"]].copy()
assert d21.shape[0] == 104 and len(p21) == 8
d21.to_csv(OUT / "regression_2_1_dataset.csv", index=False)

# Objective 2.2: one row per team per match. Exactly eight pre-match predictors.
p22 = [
    "fifa_rank_advantage", "elo_advantage", "age_difference",
    "value_advantage_eur_m", "home_away", "host_nation",
    "stage_numeric", "match_number"
]
d22 = team[["match_id", "team", "opponent_team"] + p22 + ["goals_scored"]].copy()
assert d22.shape[0] == 208 and len(p22) == 8
d22.to_csv(OUT / "regression_2_2_dataset.csv", index=False)

def model_specs():
    return {
        "OLS": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": np.logspace(-3, 3, 25)}),
        "Lasso": (Lasso(max_iter=50000), {"model__alpha": np.logspace(-4, 1, 25)}),
        "ElasticNet": (ElasticNet(max_iter=50000), {
            "model__alpha": np.logspace(-4, 1, 18),
            "model__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9]
        }),
    }

def evaluate_task(df, predictors, target, groups, task_name):
    X = df[predictors].astype(float)
    y = df[target].astype(float)
    outer = GroupKFold(n_splits=5) if groups is not None else KFold(
        n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    predictions = np.full(len(df), np.nan)

    for name, (estimator, grid) in model_specs().items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        fold_mae, fold_rmse, fold_r2 = [], [], []
        pred_all = np.full(len(df), np.nan)

        split_iter = outer.split(X, y, groups) if groups is not None else outer.split(X, y)
        for tr, te in split_iter:
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            if grid:
                if groups is not None:
                    inner = GroupKFold(n_splits=4)
                    search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                                          cv=inner, n_jobs=-1)
                    search.fit(Xtr, ytr, groups=np.asarray(groups)[tr])
                else:
                    inner = KFold(n_splits=4, shuffle=True, random_state=SEED)
                    search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                                          cv=inner, n_jobs=-1)
                    search.fit(Xtr, ytr)
                fitted = search.best_estimator_
            else:
                fitted = pipe.fit(Xtr, ytr)
            pred = fitted.predict(Xte)
            pred_all[te] = pred
            fold_mae.append(mean_absolute_error(yte, pred))
            fold_rmse.append(mean_squared_error(yte, pred) ** 0.5)
            fold_r2.append(r2_score(yte, pred))
        rows.append({
            "model": name,
            "MAE_mean": np.mean(fold_mae), "MAE_sd": np.std(fold_mae, ddof=1),
            "RMSE_mean": np.mean(fold_rmse), "RMSE_sd": np.std(fold_rmse, ddof=1),
            "R2_mean": np.mean(fold_r2), "R2_sd": np.std(fold_r2, ddof=1),
        })
        if name == "OLS":
            predictions = pred_all

    comp = pd.DataFrame(rows).sort_values("MAE_mean")
    comp.to_csv(OUT / f"{task_name}_model_comparison.csv", index=False)

    # Statistical OLS on full data with HC3 robust standard errors.
    Xsm = sm.add_constant(X)
    ols = sm.OLS(y, Xsm).fit(cov_type="HC3")
    ci = ols.conf_int()
    coef = pd.DataFrame({
        "term": ols.params.index, "coefficient": ols.params.values,
        "robust_p": ols.pvalues.values,
        "ci_low": ci.iloc[:, 0].values, "ci_high": ci.iloc[:, 1].values
    })
    coef.to_csv(OUT / f"{task_name}_ols_coefficients.csv", index=False)

    vif = pd.DataFrame({
        "variable": predictors,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    vif.to_csv(OUT / f"{task_name}_vif.csv", index=False)
    resid = ols.resid
    bp = het_breuschpagan(resid, Xsm)
    diag = {
        "n": len(df), "predictors": len(predictors),
        "ols_r2": float(ols.rsquared), "ols_adj_r2": float(ols.rsquared_adj),
        "breusch_pagan_lm_p": float(bp[1]), "breusch_pagan_f_p": float(bp[3]),
        "best_model_by_cv_mae": comp.iloc[0]["model"],
        "best_cv_mae": float(comp.iloc[0]["MAE_mean"]),
        "best_cv_rmse": float(comp.iloc[0]["RMSE_mean"]),
        "best_cv_r2": float(comp.iloc[0]["R2_mean"])
    }
    with open(OUT / f"{task_name}_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    # Compact report-ready figures.
    plt.figure(figsize=(6.5, 4))
    plt.bar(comp["model"], comp["MAE_mean"], yerr=comp["MAE_sd"], capsize=4)
    plt.ylabel("5-fold CV MAE")
    plt.title(f"{task_name}: model comparison")
    plt.tight_layout()
    plt.savefig(FIG / f"{task_name}_model_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(5, 4.5))
    plt.scatter(y, predictions, alpha=0.65)
    lo = min(y.min(), np.nanmin(predictions)); hi = max(y.max(), np.nanmax(predictions))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual"); plt.ylabel("Cross-validated OLS prediction")
    plt.title(f"{task_name}: actual vs predicted")
    plt.tight_layout()
    plt.savefig(FIG / f"{task_name}_actual_vs_predicted.png", dpi=220)
    plt.close()
    return comp, diag

c21, g21 = evaluate_task(d21, p21, "goal_difference", None, "objective_2_1")
# Group by match_id so the two rows from one match never leak across folds.
c22, g22 = evaluate_task(d22, p22, "goals_scored", d22["match_id"].values, "objective_2_2")

summary = {
    "source_rows": len(raw), "unique_matches": int(raw["match_id"].nunique()),
    "total_goals": int(raw["goals_scored"].sum()),
    "objective_2_1": g21, "objective_2_2": g22
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
