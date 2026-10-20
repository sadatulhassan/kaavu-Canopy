# Q1: Prediction churn versus error

## Research question
As training-set size increases, does disagreement between independently trained
models shrink as quickly as prediction error, and does this differ by model family?

## Included files
- `q1_churn_experiment.py`: runnable starter experiment
- Outputs generated when it runs:
  - `q1_results.csv`
  - `q1_summary.csv`

## Setup
Use Python 3.10+ and install dependencies:

```bash
python -m pip install numpy pandas scikit-learn
```

Run:

```bash
python q1_churn_experiment.py
```

## Experimental design
- Training sizes: 250, 500, 1,000, 2,000, 4,000, 8,000
- A single fixed holdout test set is used for every model and condition.
- The same 20 feature columns are used in every condition.
- Model families: Logistic Regression, Decision Tree, Histogram Gradient Boosting.
- Data-driven control: vary training subsets while holding the estimator seed fixed.
- Seed-driven control: hold the training subset fixed and vary the estimator seed.

## Metrics
- Accuracy: fraction of correct predictions.
- Error rate: `1 - accuracy`.
- Churn: fraction of test points on which a pair of models predicts different labels.
- Churn/error: a descriptive ratio; do not interpret it when error is zero or tiny.

## Important limitations
1. This first run uses a synthetic dataset to validate the pipeline, not to make
   a general scientific claim.
2. Logistic regression is nearly deterministic for a fixed dataset and settings;
   its seed-driven churn may be zero.
3. Tree and boosting seed-driven churn depends on the estimator's actual
   randomness; changing `random_state` does not guarantee different predictions.
4. The data-driven subsets are sampled without replacement from a fixed training
   pool. For the seed-driven control, each n uses a deterministic nested subset.
5. The script reports averages across model pairs, but the pairs are not
   statistically independent. For a formal report, add uncertainty intervals
   using repeated outer datasets or a suitable bootstrap.
6. A model's hyperparameters and preprocessing are held fixed across n. Do not
   tune them separately at each training size unless tuning is explicitly part
   of the research question.

## Next steps
1. Run the starter script and inspect the CSV files.
2. Plot accuracy, error rate, and churn against n, separately for each family
   and control.
3. Check whether churn falls at the same rate as error.
4. After the pipeline works, repeat on a real, appropriate dataset and clearly
   document its source, features, split, and any preprocessing.
