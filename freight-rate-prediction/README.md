# Freight Rate Prediction

Predicts the posted rate (USD) of a freight load from its lane, distance, equipment, weight, date and two market signals.
Trained on `data/train_test.csv` (Jan-Oct 2025), predicts the 12,000 loads in `data/validation.csv` (Nov-Dec 2025).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# put the provided files in data/ with these names:
#   train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv

python src/validate.py        # forward-chained backtest + unseen-city stress test  -> outputs/*.csv
python src/train_predict.py   # final model -> validation_predictions.csv + outputs/december_predictions.csv
python score.py --predictions validation_predictions.csv \
                --december-predictions outputs/december_predictions.csv \
                --output-dir outputs                      # -> outputs/candidate_december.png
```

Everything is deterministic (fixed seeds); a full run takes about a minute on a laptop.

## Layout

| File | Purpose |
|---|---|
| `src/features.py` | Data cleaning + feature engineering (every step commented with its reason) |
| `src/model.py` | Two-stage model: robust linear trend, then gradient boosting on the residual |
| `src/validate.py` | Time-based backtest, baselines, unseen-city stress test |
| `src/train_predict.py` | Final fit on all labeled data, writes the submission files |
| `score.py` | Provided scorer (unchanged) |

## Approach in short

1. **Split by time, never randomly.** The task is to predict the future (Nov-Dec), so each backtest fold trains on the past and predicts the next two months.
2. **Data-quality fixes:** negative weights -> absolute value (sign errors); weight capped at 47,500 -> flag; missing weight -> median by equipment; missing `market_index` -> same-day median across loads.
3. **Geography as coordinates, not city names:** 8 validation cities never appear in training.
4. **Robust to label spikes:** ~0.6% of loads have rates 3-6x normal. Huber loss (stage 1) and absolute-error loss (stage 2) ignore them instead of chasing them.
5. **Two stages** so the model can extrapolate the slow upward trend into months it has not seen.
6. **December chart:** `market_index` and `quote_signal` are not in the chart file, so they are taken from `validation.csv` for the same dates (feature columns only, no labels used).
