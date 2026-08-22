# Fraud Detection with Imbalanced Classification

Classification-based fraud detection system for a payments company, focused on
correctly handling severe class imbalance and choosing evaluation metrics that
reflect real business priorities rather than misleading accuracy scores.

## Business Context

Acting as a Data Scientist on a risk team, this project builds a system to flag
potentially fraudulent transactions for review, prioritizing catching real fraud
(recall) over avoiding false alarms (precision) — a deliberate business trade-off,
since missing fraud is costlier to a payments company than flagging a legitimate
transaction for manual review.

## Dataset

**Note: This dataset is synthetically generated**, since real fraud data is not
publicly available due to banking privacy regulations. It simulates realistic,
overlapping fraud/legitimate transaction patterns (transaction amount, time of day,
account age, transaction velocity, distance from home, merchant category, foreign
transaction flag) with a **~1.5% fraud rate**, matching real-world fraud prevalence.

- 12,000 transactions | 179 fraudulent (1.5%)

## Key Findings

### 1. Why accuracy is a dangerous metric here
A trivial model that predicts "legitimate" for every transaction would score
**~98.8% accuracy** while catching **zero fraud cases** — proof that accuracy alone
is meaningless on severely imbalanced data. This project uses **precision, recall,
and F1-score** instead, evaluated via the confusion matrix.

### 2. Baseline model struggles with the minority class
A default Logistic Regression model achieved 99% accuracy but only **11% recall**
for fraud (catching 4 of 36 fraud cases in the test set) — it defaults to predicting
the majority class whenever uncertain, since that's overwhelmingly what it saw
during training.

### 3. Addressing imbalance improves recall, at a real cost to precision
Using `class_weight='balanced'` raised **recall to 67%** (catching 24 of 36 fraud
cases), but **precision dropped to 5%** — meaning roughly 19 out of 20 flagged
transactions are false alarms. This is a genuine trade-off, not a bug: given the
business priority on catching fraud, this was accepted as a reasonable baseline,
while acknowledging the operational cost of reviewing many false positives.

### 4. A "stronger" model was tested and rejected
Random Forest with the same class-weighting achieved **0% recall** — worse than
the simple baseline. This is because `class_weight='balanced'` doesn't translate
the same way to Random Forest's bootstrap-sampled trees, which often end up with
few or no fraud examples given how rare they are. This confirms that a more
sophisticated model isn't automatically better for a given problem — it was tested
empirically and rejected, rather than assumed to be superior.

### 5. Decision threshold tuning
Since the model outputs a probability rather than a hard label, the classification
threshold was tuned (tested 0.2–0.7) to explore the precision/recall trade-off
directly. The default 0.5 threshold was kept as a reasonable balance, given the
business priority on recall without making precision unworkably low.

## Technical Highlights

- Diagnosed severe class imbalance (1.5% positive class) and demonstrated why
  accuracy fails as an evaluation metric in this setting
- Used `stratify=y` in the train/test split to preserve class ratio in both sets
- Built and interpreted a full **confusion matrix**, connecting false negatives
  and false positives to real business costs before choosing which to prioritize
- Compared **Logistic Regression vs. Random Forest** under class imbalance and
  made a data-driven (not assumption-driven) model choice
- Tuned the **decision threshold** using `predict_proba()` instead of relying on
  the default 0.5 cutoff, understanding precision/recall as a continuous trade-off

## Tools & Libraries
- Python (Pandas, NumPy)
- Scikit-learn (Logistic Regression, Random Forest, metrics)
- Matplotlib, Seaborn

## How to Run
```bash
git clone https://github.com/tjin9/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

## Project Structure
```
├── data/              # Dataset (not included - synthetic, generation logic in notebook)
├── notebooks/         # Analysis + modeling notebook
├── images/            # Exported chart images
└── requirements.txt
```
## Author

**Tasneem** — Computer Science student, self-studying Data Science & AI.
[GitHub: tjin9](https://github.com/tjin9)
