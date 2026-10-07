# HIT140 Assessment 3 - FIFA World Cup 2026 Regression

This repository is the reproducible code/data component for **Project Brief Objective 2 only**.

## Group
- S398541 Sakshi
- S397783 Arshdeep Singh
- S397295 Kajal
- S397214 Harteetpal Singh

## Objective 2
**2.1 Goal difference:** 104 rows, one row per FIFA World Cup 2026 match, with exactly eight explanatory variables available before the match.

**2.2 Team goals:** 208 rows, one team-specific row per match, with exactly eight explanatory variables available before the match.

## Reproducible analysis
Run:

```bash
pip install -r requirements.txt
python analysis_objective2.py
```

The script validates 208 team-match rows, 104 unique matches and 308 tournament goals, builds the two required modelling datasets, compares OLS, Ridge, Lasso and Elastic Net, and exports cross-validation metrics, robust OLS diagnostics, VIF tables and report-ready figures.

For Objective 2.2, GroupKFold is used by `match_id` so the two observations belonging to the same match cannot be split between training and validation folds. Hyperparameters for regularised models are tuned only inside the training portion of each outer fold.

## Data design
The model uses only variables known before kick-off. Difference/advantage features are used for team strength and squad characteristics to reduce redundant information and improve interpretability. Match stage and scheduled match number are known before kick-off and are retained as tournament-context predictors.

Penalty-shootout kicks are not treated as match goals; the modelling target is the official match score after normal/extra time.

## External validation and sources
Tournament-level validation is checked against FIFA's published total of 104 matches and 308 goals. Match results can be independently checked against FBref's 2026 World Cup Scores & Fixtures.

- FIFA World Cup 2026 statistics: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
- FIFA tournament summary: https://inside.fifa.com/tournament-organisation/fifa-world-cup-2026-by-the-numbers/on-the-pitch
- FBref World Cup scores and fixtures: https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures
- The Stats Don't Lie, World Cup 2026: https://www.thestatsdontlie.com/football/world-cup-2026/

## Repository files
- `208_world_cup_dataset.csv`: team-match source dataset.
- `104_world_cup_dataset.xlsx`: match-level source workbook retained for traceability.
- `analysis_objective2.py`: final reproducible analysis.
- `requirements.txt`: Python dependencies.
- `.github/workflows/objective2.yml`: automated reproducibility check and results artifact.
- `PROJECT_SCOPE.md`: confirms the assessed scope.

Generated `results/` and `figures/` folders are produced by the script and uploaded as a GitHub Actions artifact.

## Reproducibility note
All model performance reported in the group report should be taken from the generated result files rather than manually typed calculations.
