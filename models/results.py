# Import in packages
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Read in boosted predictions and ground truth data
preds = pd.read_parquet("data/boosted_preds.parquet")
ground_truth = pd.read_csv("data/ground_truth.csv")

# Create id features to align with mentions data
ground_truth['id_1870'] = "ALB-CN-1870-" + ground_truth['1870_line'].astype(str)
ground_truth['id_1880'] = "ALB-CN-1880-" + ground_truth['1880_line'].astype(str)

# Join matches on to ground truth data
data = pd.merge(ground_truth, preds, how='left', left_on=['id_1870', 'id_1880'], right_on=['unique_id_l', 'unique_id_r'])

# Drop ID columns from data (don't need them)
data_subset = data.drop(['1870_line', '1880_line', 'unique_id_l', 'unique_id_r'], axis=1)

# Density plot to show distribution of match probabilities
plt.figure()
sns.kdeplot(data_subset['match_probability'])
plt.title("Distribution of Match Probabilities (subset of ground truth)")

# Within data_subset, a majority of our probabilities are at a higher scale -
# meaning that our EM model is confident in a lot of these candidates being matches.

plt.figure()
sns.kdeplot(preds['match_probability'])
plt.title("Distribution of Match Probabilities (all data)")

# On the other hand, in looking at the distribution for our match probabilities
# from preds, our model has much less confidence in the probability values,
# with a majority of values converging around 0.4.

# Barchart of match probabilities by confidence
agg_confidence = data_subset.groupby('confidence')['match_probability'].mean().reset_index()
plt.figure()
sns.barplot(x="confidence", y="match_probability", data=agg_confidence)
plt.xlabel("Confidence")
plt.ylabel("Match Probability")
plt.title("EM Match Probability by Confidence")

# Barchart of relative match probabilities by confidence
agg_confidence = data_subset.dropna().groupby('confidence')['relatives_match_probability'].mean().reset_index()
plt.figure()
sns.barplot(x="confidence", y="relatives_match_probability", data=agg_confidence)
plt.xlabel("Confidence")
plt.ylabel("Match Probability")
plt.title("EM Relative Match Probability by Confidence")

# Scatterplot of score/match_probability
plt.figure()
sns.scatterplot(x="score", y="match_probability", data=data_subset)
plt.xlabel("Score")
plt.ylabel("Match Probability")
plt.title("EM Match Probability by Score")

# Investigate the contents of ground truth

# Look at the summary stats for score/confidence
print(ground_truth[['score', 'confidence']].describe())

print(ground_truth['score'].nunique(), ground_truth['confidence'].nunique())

print(ground_truth['score'].value_counts().head(20))

print(ground_truth['confidence'].value_counts())

print(pd.crosstab(ground_truth['confidence'], ground_truth['score']))

# 1. Treat confidence=3 as a "perfect" match (essentially match=1), and measure
# auc, log loss, brier score, accuracy, and calibration

binary_eval = data_subset.loc[
    # Treat confidence as binary label for 0/3 being 3 = 1
    data_subset['confidence'].isin([0, 3]),
    ['confidence', 'match_probability']
].dropna(subset=['match_probability'])
y_true = (binary_eval['confidence'] == 3).astype(int)
y_prob = binary_eval['match_probability']

# Get AUC
auc = roc_auc_score(y_true, y_prob)
# Get Brier score
brier = brier_score_loss(y_true, y_prob)
# Get log loss
log_loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
# Get overall "accuracy" (if predicted match >= 0.5, it's a match)
accuracy = (y_true == (y_prob >= 0.5)).mean()
print(f"AUC: {auc:.4f}")
print(f"Brier: {brier:.4f}")
print(f"Log loss: {log_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"n = {len(binary_eval)} (dropped {len(data_subset) - len(binary_eval)} unmatched rows, unmatched being when confidence != 0 or 3)")

# Calibration plot
def calibration_data(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    df_cal = pd.DataFrame({'bin': bin_ids, 'y_true': y_true, 'y_prob': y_prob})
    grouped = df_cal.groupby('bin').agg(
        mean_pred=('y_prob', 'mean'),
        mean_true=('y_true', 'mean'),
        n=('y_true', 'count')
    ).reset_index()
    return grouped

cal = calibration_data(y_true.values, y_prob.values)
plt.figure()
plt.plot(cal['mean_pred'], cal['mean_true'], marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Observed fraction confidence=3')
plt.legend()
plt.title("Model Calibration Plot")

# Our model's AUC = 0.937 indicates that our model is effective in discriminating
# "true" matches ahead of non-matches. Further, an overall "accuracy" = 0.8198
# is solid, meaning our model is over 80% effective in determining a true match,
# however, we know the distribution of confidence values are right-skew, so with
# any lenient model, it's only a slight improvement over a dataset with a
# majority of the observations being a match. A brier score of 0.1823 is solid,
# and aligns similarly to the AUC's performance, however, where the brier score
# stands out is that it indicates our model isn't well-calibrated as it's not
# as low as the AUC might've suggested. Lastly, and most notably, the logloss
# (1.2630) is very poor, and highlights that oftentimes many of our model
# predictions are overconfident. For example, with instances like Dabney
# Johnson, because the name is common, our model treats that as a "match" and
# is unable to discern which Dabney Johnson is which at the same time, and our
# EM probabilities are considerably high for a true non-match. While "boosting"
# provides better insight into which exact match belongs to each candidate
# pair, it can't penalize poor relative probabilities. The calibration confirms
# this, as our highly confident predictions (around 1 for a probability) only
# correspond to a true match rate at around 75%, and our less confident
# predictions (~0.4) correspond to a true match rate of around 0% with the
# ground truth data provided.

# Display all figures created above at once, in a single terminal run
plt.show()