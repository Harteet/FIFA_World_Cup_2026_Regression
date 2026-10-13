"""Week 9 data-analysis supplement for HIT140 Objective 2.
Run from repository root: python analysis_week9.py
Uses the existing 208_world_cup_dataset.csv; does not modify the original analysis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "week9_results"
OUT.mkdir(exist_ok=True)
raw = pd.read_csv(ROOT / "208_world_cup_dataset.csv")
assert len(raw) == 208 and raw.match_id.nunique() == 104
assert raw.groupby("match_id").size().eq(2).all()
assert int(raw.goals_scored.sum()) == 308
columns = ["team", "fifa_rank", "elo_rating", "squad_avg_age",
           "squad_total_value_eur_m", "host_nation", "goals_scored"]
opp = raw[["match_id"] + columns].rename(
    columns={col: "opponent_" + col for col in columns})
data = raw.merge(opp, on="match_id")
data = data[data.team != data.opponent_team].copy()
assert len(data) == 208
data["fifa_rank_advantage"] = data.opponent_fifa_rank - data.fifa_rank
data["elo_advantage"] = data.elo_rating - data.opponent_elo_rating
data["age_difference"] = data.squad_avg_age - data.opponent_squad_avg_age
data["value_advantage_eur_m"] = (
    data.squad_total_value_eur_m - data.opponent_squad_total_value_eur_m)
common = ["fifa_rank_advantage", "elo_advantage", "age_difference",
          "value_advantage_eur_m"]
p21 = common + ["host_nation", "opponent_host_nation", "stage", "match_number"]
p22 = common + ["home_away", "host_nation", "stage", "match_number"]
match = data[data.home_away == 1].copy()
match["goal_difference"] = match.goals_scored - match.opponent_goals_scored

rows = []
for task, frame, features, target in [
    ("2_1", match, p21, "goal_difference"),
    ("2_2", data, p22, "goals_scored"),
]:
    X, y = frame[features].astype(float), frame[target].astype(float)
    cv = (KFold(n_splits=5, shuffle=True, random_state=42)
          if task == "2_1" else GroupKFold(n_splits=5))
    groups = None if task == "2_1" else frame.match_id
    for name, transformer in [
        ("StandardScaler", StandardScaler()),
        ("YeoJohnson", PowerTransformer(method="yeo-johnson")),
    ]:
        model = make_pipeline(transformer, LinearRegression())
        pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        rows.append({"task": task, "preprocessing": name,
                     "MAE": mean_absolute_error(y, pred),
                     "RMSE": mean_squared_error(y, pred) ** 0.5,
                     "R2": r2_score(y, pred)})
    frame[["match_id", "team"] + features + [target]].head(4).to_csv(
        OUT / f"snippet_{task}.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4))
    fig.subplots_adjust(hspace=0.43, wspace=0.32)
    for ax, feature in zip(axes.flat, common):
        ax.scatter(frame[feature], y, s=12, alpha=0.55)
        slope, intercept = np.polyfit(frame[feature], y, 1)
        xx = np.linspace(frame[feature].min(), frame[feature].max(), 100)
        ax.plot(xx, slope * xx + intercept, linewidth=1.5)
        ax.set_xlabel(feature.replace("_", " "), fontsize=8)
        ax.set_ylabel(target.replace("_", " "), fontsize=8)
        ax.set_title(f"Pearson r = {np.corrcoef(frame[feature], y)[0, 1]:.2f}")
        ax.tick_params(labelsize=7)
    fig.savefig(OUT / f"scatter_{task}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
pd.DataFrame(rows).to_csv(OUT / "week9_preprocessing_metrics.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
