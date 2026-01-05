"""
Created on Fri Jun 13 14:39:21 2025

@author: bilalnaseer
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression

# Loading data
df = pd.read_csv("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/creditcard.csv")

# Preview the structure
print(df.head())
print(df.info())

# Q1 
#Preprocessing + Class Distribution

# 1.1 Check basic structure and missing values
structure_info = df.info()
missing_values = df.isnull().sum()

# 1.2 Check class distribution
class_counts = df['Class'].value_counts()
class_percentages = df['Class'].value_counts(normalize=True) * 100

# 1.3 Bar plot for class distribution
plt.figure(figsize=(6,4))
bars = plt.bar(class_counts.index.map({0: "Non-Fraud", 1: "Fraud"}), class_counts.values, color=["skyblue", "salmon"])
plt.title("Distribution of Transaction Classes")
plt.ylabel("Count")
plt.xlabel("Transaction Type")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 200, f'{height:.0f}', ha='center')
plt.tight_layout()
plt.savefig("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/q1_class_distribution.png")
plt.close()

# Output key findings
(class_counts, class_percentages, missing_values)

# Q2
# Separate fraud and non-fraud transactions by 'Amount'
fraud = df[df['Class'] == 1]['Amount']
non_fraud = df[df['Class'] == 0]['Amount']

# Welch's t-test for unequal variances
t_stat, p_val = ttest_ind(fraud, non_fraud, equal_var=False)

# 99.5% confidence interval
mean_diff = fraud.mean() - non_fraud.mean()
se_diff = np.sqrt(fraud.var(ddof=1)/len(fraud) + non_fraud.var(ddof=1)/len(non_fraud))
ci_low = mean_diff - 2.807 * se_diff
ci_high = mean_diff + 2.807 * se_diff

# Create boxplot
plt.figure(figsize=(6,4))
plt.boxplot([non_fraud, fraud], labels=["Non-Fraud", "Fraud"], patch_artist=True,
            boxprops=dict(facecolor="lightblue"), medianprops=dict(color="black"))
plt.title("Transaction Amounts: Fraud vs. Non-Fraud")
plt.ylabel("Amount ($)")
plt.tight_layout()
plt.savefig("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/q2_amount_boxplot.png")
plt.close()

# Output results
{
    "Non-Fraud Mean": non_fraud.mean(),
    "Fraud Mean": fraud.mean(),
    "T-statistic": t_stat,
    "P-value": p_val,
    "99.5% CI": (ci_low, ci_high)
}

# Q3
# Convert time to hourly bins
df['Hour'] = (df['Time'] // 3600).astype(int)

# Group by hour and calculate fraud rate
hourly_stats = df.groupby('Hour')['Class'].agg(['count', 'sum'])
hourly_stats['FraudRate'] = hourly_stats['sum'] / hourly_stats['count']

# Generate plot (you will update this path in your system)
plt.figure(figsize=(8, 4))
plt.plot(hourly_stats.index, hourly_stats['FraudRate'], marker='o', color='indianred')
plt.title("Fraud Rate by Hour of Transaction")
plt.xlabel("Hour of Day (0–47)")
plt.ylabel("Fraud Rate")
plt.xticks(ticks=np.arange(0, 49, 2))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/q3_fraud_by_hour.png")
plt.close()

# Output for written report
{
    "Hour with Highest Fraud Rate": int(hourly_stats['FraudRate'].idxmax()),
    "Highest Fraud Rate": round(hourly_stats['FraudRate'].max(), 4),
    "Lowest Fraud Rate": round(hourly_stats['FraudRate'].min(), 4)
}


# Q4
# Drop 'Time' and isolate features/target
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Scale 'Amount' feature
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Train-test split with class stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15971846, stratify=y
)

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=15971846)
rf.fit(X_train, y_train)

# Predictions and probabilities
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_proba), 4))

# ROC Curve plot
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve - Baseline Random Forest")
plt.tight_layout()
plt.savefig("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/q4_rf_roc_curve.png")
plt.close()

#Q5
# Sample 20% of the data for faster processing
df_sampled = df.sample(frac=0.2, random_state=15971846)

# Preprocessing
X = df_sampled.drop(['Class', 'Time'], axis=1)
y = df_sampled['Class']
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=15971846
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=15971846)
rf.fit(X_train, y_train)

# Feature importances
importances = rf.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10], palette="Blues_d")
plt.title("Top 10 Most Important Features - Random Forest")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/q5_rf_feature_importance.png")
plt.close()

# Print top 10 features for your written report
print(feat_imp.head(10))

#Q6
# Generate Precision-Recall curve
y_scores = rf.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_scores)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title("Precision-Recall Curve - Random Forest")
plt.tight_layout()
plt.savefig("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/q6_rf_pr_curve.png")
plt.close()

# Report summary statistics for reasoning
{
    "Average Precision": round(np.mean(precision), 4),
    "Max Precision": round(np.max(precision), 4),
    "Min Recall": round(np.min(recall), 4)
}

#Q7
# Train Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=15971846)
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, digits=4)
roc_auc = round(roc_auc_score(y_test, y_proba), 4)

# Plot ROC curve
RocCurveDisplay.from_estimator(lr, X_test, y_test)
plt.title("ROC Curve - Logistic Regression")
plt.tight_layout()
plt.savefig("/Users/bilalnaseer/Documents/Spring '25/PODS/Kaggle_Projects/q7_lr_roc_curve.png")
plt.close()

print(conf_matrix, class_report, roc_auc)




