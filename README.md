# ML World Cup Predictor

A portfolio project that builds a model to predict outcomes for the 2026 FIFA
World Cup, using 150+ years of international football results. The project
demonstrates an end-to-end data science workflow in Python: sourcing and
cleaning real-world data, engineering domain-specific features (a custom Elo
rating system), and using those features to train a machine learning model
(Random Forest) for match outcome prediction.

## Motivation

Most public World Cup prediction projects either ignore long-run team
strength entirely or rely on FIFA's official rankings, which update slowly
and weight results in ways that aren't very transparent. This project
instead builds a custom Elo rating from the full match history, so that
"team strength" is a feature engineered from first principles rather than an
external label — and so that the reasoning behind every prediction can be
traced back to actual results.

## Methodology

The project is built in three stages, each developed first in a notebook and
then promoted into tested, reusable code in `src/` once the approach is
validated:

1. **Data loading & cleaning** (`notebooks/international_match_data.ipynb`,
   [`data_loading.py`](src/ml_world_cup_predictor/data_loading.py))
   Downloads the
   [International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
   dataset via `kagglehub`, profiles every column for nulls/cardinality/range,
   reconciles results with shootout outcomes, fixes known data gaps, and
   derives a `result` (W/D/L) column for every match.

2. **Elo rating system** (`notebooks/elo_creation.ipynb`,
   [`elo_creation.py`](src/ml_world_cup_predictor/elo_creation.py))
   Implements a custom Elo rating, updated match-by-match across the full
   history, with:
   - a home-advantage offset, removed for matches played on neutral ground
   - per-tournament K-factors, so World Cup and continental championship
     results move ratings more than friendlies
   - a goal-difference multiplier, so heavier wins/losses move ratings further
   - a minimum-games filter, so teams with too sparse a history don't
     distort the rating pool

   This produces an Elo time series for every team, which is the core
   engineered feature for the prediction model.

3. **Match outcome prediction (in progress)**
   A Random Forest classifier, trained on Elo ratings and match context
   (e.g. home/away/neutral, tournament) to predict Win/Draw/Loss for each
   fixture, ultimately used to simulate the 2026 World Cup. Not yet built —
   see [Roadmap](#roadmap).

## Project structure

```
.
├── data/                                # downloaded dataset (gitignored after first run)
├── notebooks/
│   ├── international_match_data.ipynb   # data loading, cleaning, exploration
│   └── elo_creation.ipynb               # Elo rating system development
├── src/ml_world_cup_predictor/
│   ├── config.py                        # shared constants (starting Elo, K-factors, etc.)
│   ├── data_loading.py                  # dataset download/caching + load helpers
│   └── elo_creation.py                  # Elo calculation, promoted from elo_creation.ipynb
├── main.py
├── pyproject.toml
└── uv.lock
```

Notebooks are used as a scratch pad for exploration; once an approach is
validated, the logic is refactored into a function in `src/` and imported
back into the notebook. This keeps the notebooks readable as a narrative of
the analysis while keeping the reusable logic testable and importable.

## Data

[International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
(Kaggle, via `martj42`) — ~49,000 men's full international matches, including
results, shootout outcomes, and goalscorers. Downloaded automatically on
first run via `kagglehub` and cached locally in `data/`.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# clone the repo, then from the project root:
uv sync

# launch the notebooks
uv run jupyter lab
```

Running either notebook will download the dataset automatically on first use
(via `kagglehub`) and cache it under `data/`.

## Roadmap

- [x] Load, clean, and validate the international match results dataset
- [x] Build a custom Elo rating system with tournament weighting and
      home-advantage adjustment
- [ ] Engineer a feature set (Elo, recent form, tournament context) for
      match-level prediction
- [ ] Train and evaluate a Random Forest classifier for match outcomes
- [ ] Simulate the 2026 World Cup bracket using the trained model
- [ ] Write up results and model evaluation

## Tech stack

Python 3.13 · pandas · NumPy · scikit-learn (planned) · seaborn / Plotly ·
Jupyter · uv
