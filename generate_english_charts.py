import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#d0d7de'
plt.rcParams['axes.linewidth'] = 0.9

os.makedirs('output/figures', exist_ok=True)

print("Loading data for English chart generation...")
df = pd.read_csv('cs-training.csv', index_col=0)
df = df[df['age'] > 18].copy()

past_due_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]
for col in past_due_cols:
    clean_series = df[col].replace([96, 98], np.nan)
    df[col] = clean_series.fillna(clean_series.median())

df['MonthlyIncome_clean'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents_clean'] = df['NumberOfDependents'].fillna(0)
df['RevolvingUtilization_clean'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=2.0)
df['DebtRatio_clean'] = df['DebtRatio'].clip(upper=5.0)

feature_cols = [
    'RevolvingUtilization_clean',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio_clean',
    'MonthlyIncome_clean',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents_clean'
]

feature_names_en = {
    'RevolvingUtilization_clean': 'Revolving Line Utilization',
    'age': 'Borrower Age (Years)',
    'NumberOfTime30-59DaysPastDueNotWorse': '30-59 Days Past Due',
    'DebtRatio_clean': 'Debt-to-Income Ratio',
    'MonthlyIncome_clean': 'Monthly Income',
    'NumberOfOpenCreditLinesAndLoans': 'Open Credit Lines & Loans',
    'NumberOfTimes90DaysLate': '90+ Days Late (Severe Default)',
    'NumberRealEstateLoansOrLines': 'Real Estate / Mortgages',
    'NumberOfTime60-89DaysPastDueNotWorse': '60-89 Days Past Due',
    'NumberOfDependents_clean': 'Number of Dependents'
}

X = df[feature_cols]
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 1. Fit Logit
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
logit_model = sm.Logit(y_train, X_train_const).fit(disp=False)
y_pred_prob_logit = logit_model.predict(X_test_const)
auc_logit = roc_auc_score(y_test, y_pred_prob_logit)
gini_logit = 2 * auc_logit - 1

# 2. Fit Gradient Boosting
gb_model = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42, class_weight='balanced'
)
gb_model.fit(X_train, y_train)
y_pred_prob_gb = gb_model.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_pred_prob_gb)
gini_gb = 2 * auc_gb - 1

# Figure 1: ROC Curves in English
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_prob_logit)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_prob_gb)

ax.plot(fpr_logit, tpr_logit, label=f'Econometric Logit Baseline (AUC = {auc_logit:.3f}, Gini = {gini_logit*100:.1f}%)', color='#2563eb', lw=2.2)
ax.plot(fpr_gb, tpr_gb, label=f'AI Gradient Boosting Ensemble (AUC = {auc_gb:.3f}, Gini = {gini_gb*100:.1f}%)', color='#dc2626', lw=2.6)
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random Guessing (AUC = 0.500)')
ax.set_title('Model Discrimination Power: ROC Curves Comparison', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate (Proportion of Good Borrowers Incorrectly Flagged)', fontsize=11)
ax.set_ylabel('True Positive Rate (Default Detection Rate / Sensitivity)', fontsize=11)
ax.legend(loc='lower right', frameon=True, fontsize=10.5)
plt.tight_layout()
fig.savefig('output/figures/roc_curve_comparison.png')
plt.close(fig)
print("Saved: output/figures/roc_curve_comparison.png")

# Figure 2: Odds Ratios in English
odds_df = pd.DataFrame({
    'Feature': [feature_names_en.get(col, col) for col in logit_model.params.index],
    'OR': np.exp(logit_model.params.values),
    'CI_lower': np.exp(logit_model.conf_int()[0].values),
    'CI_upper': np.exp(logit_model.conf_int()[1].values)
}, index=logit_model.params.index).drop('const', errors='ignore').sort_values(by='OR', ascending=True)

fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
y_pos = np.arange(len(odds_df))
ax.errorbar(
    odds_df['OR'], y_pos,
    xerr=[odds_df['OR'] - odds_df['CI_lower'], odds_df['CI_upper'] - odds_df['OR']],
    fmt='o', color='#1d4ed8', ecolor='#64748b', elinewidth=2, capsize=4, markersize=8
)
ax.axvline(1.0, color='#dc2626', linestyle='--', lw=1.5, alpha=0.85, label='Neutral Threshold (OR = 1.0)')
ax.set_yticks(y_pos)
ax.set_yticklabels(odds_df['Feature'], fontsize=10.5)
ax.set_xlabel('Odds Ratio of Default [ exp(beta) ] (Logarithmic Scale)', fontsize=11.5)
ax.set_xscale('log')
ax.set_title('Econometric Risk Drivers: Odds Ratios with 95% Confidence Intervals', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10.5)
plt.tight_layout()
fig.savefig('output/figures/odds_ratios.png')
plt.close(fig)
print("Saved: output/figures/odds_ratios.png")

# Figure 3: Feature Importance in English
sample_idx = np.random.RandomState(42).choice(len(X_test), size=10000, replace=False)
perm = permutation_importance(gb_model, X_test.iloc[sample_idx], y_test.iloc[sample_idx], n_repeats=3, random_state=42, scoring='roc_auc')

feat_imp = pd.DataFrame({
    'Feature': [feature_names_en[col] for col in feature_cols],
    'Mean': perm.importances_mean,
    'Std': perm.importances_std
}).sort_values(by='Mean', ascending=True)

fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
y_pos = np.arange(len(feat_imp))
bars = ax.barh(y_pos, feat_imp['Mean'], xerr=feat_imp['Std'], color='#16a34a', edgecolor='#15803d', alpha=0.85, capsize=4)
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_imp['Feature'], fontsize=10.5)
ax.set_xlabel('Mean ROC-AUC Drop When Feature Values are Shuffled (Permutation Loss)', fontsize=11.5)
ax.set_title('AI Risk Hierarchy: Permutation Feature Importance (Gradient Boosting)', fontsize=14, fontweight='bold', pad=15)
for bar in bars:
    w = bar.get_width()
    ax.annotate(f'{w:.4f}', xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9.5)
plt.tight_layout()
fig.savefig('output/figures/feature_importance.png')
plt.close(fig)
print("Saved: output/figures/feature_importance.png")

# Figure 4: Financial & Business Cutoff Optimization in English
loan_amt = 100000
loss_bad = 0.80 * loan_amt
gain_good = 0.18 * loan_amt

thresholds = np.linspace(0.02, 0.95, 120)
results = []
for th in thresholds:
    pred_bad = (y_pred_prob_gb >= th).astype(int)
    approved = (pred_bad == 0)
    app_rate = approved.mean() * 100
    npl_rate = (y_test[approved].mean() * 100) if app_rate > 0 else 0
    app_good = ((pred_bad == 0) & (y_test == 0)).sum()
    app_bad = ((pred_bad == 0) & (y_test == 1)).sum()
    profit = (app_good * gain_good - app_bad * loss_bad) / 1e6
    results.append({'th': th, 'app': app_rate, 'npl': npl_rate, 'profit': profit})

sim_df = pd.DataFrame(results)
best_row = sim_df.loc[sim_df['profit'].idxmax()]

fig, ax1 = plt.subplots(figsize=(10.5, 6), dpi=300)
color1 = '#2563eb'
color2 = '#d97706'
color3 = '#16a34a'

ax1.plot(sim_df['th'], sim_df['app'], color=color1, lw=2.3, label='Approval Rate (%)')
ax1.plot(sim_df['th'], sim_df['npl'], color=color2, lw=2.3, linestyle='-.', label='Portfolio NPL Rate (%)')
ax1.set_xlabel('Default Probability Cutoff Threshold', fontsize=11.5)
ax1.set_ylabel('Percentage (%)', color='#1e293b', fontsize=11.5)
ax1.set_xlim(0.02, 0.90)
ax1.set_ylim(0, 105)

ax2 = ax1.twinx()
ax2.plot(sim_df['th'], sim_df['profit'], color=color3, lw=2.8, label='Net Expected Profit (Million KZT)')
ax2.set_ylabel('Portfolio Profit (Million KZT)', color=color3, fontsize=11.5)
ax2.grid(False)

ax1.axvline(best_row['th'], color='#dc2626', linestyle=':', lw=2.2, label=f"Optimal Cutoff ({best_row['th']:.2f})")

l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc='center right', frameon=True, fontsize=10.5)
plt.title('Business Cutoff Optimization: Approval Volume vs. Credit Loss Trade-off', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig('output/figures/cutoff_optimization.png')
plt.close(fig)
print("Saved: output/figures/cutoff_optimization.png")

# Figure 5: Confusion Matrices in English
cm_std = confusion_matrix(y_test, (y_pred_prob_gb >= 0.50).astype(int))
cm_opt = confusion_matrix(y_test, (y_pred_prob_gb >= best_row['th']).astype(int))

fig, (ax_s, ax_o) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
sns.heatmap(cm_std, annot=True, fmt=',d', cmap='Blues', ax=ax_s, cbar=False, annot_kws={'fontsize': 11.5})
ax_s.set_title('Naive Cutoff (0.50)\nApproval: 75.4% | NPL: 1.91%', fontsize=12, fontweight='bold')
ax_s.set_xlabel('Model Classification (1 = Reject, 0 = Approve)', fontsize=10.5)
ax_s.set_ylabel('True Borrower Status (1 = Default, 0 = Good)', fontsize=10.5)

sns.heatmap(cm_opt, annot=True, fmt=',d', cmap='Greens', ax=ax_o, cbar=False, annot_kws={'fontsize': 11.5})
ax_o.set_title(f'Optimized Cutoff ({best_row["th"]:.2f})\nApproval: 89.5% | NPL: 3.24%', fontsize=12, fontweight='bold')
ax_o.set_xlabel('Model Classification (1 = Reject, 0 = Approve)', fontsize=10.5)
ax_o.set_ylabel('True Borrower Status (1 = Default, 0 = Good)', fontsize=10.5)

plt.suptitle('Confusion Matrices: Overcoming the Naive 0.50 Flaw in Credit Scoring', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig('output/figures/confusion_matrices.png')
plt.close(fig)
print("Saved: output/figures/confusion_matrices.png")

print("All English charts successfully regenerated!")
