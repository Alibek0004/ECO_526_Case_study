import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)

print("="*70)
print("1. DATA PREPARATION & CLEANING")
print("="*70)

df_raw = pd.read_csv('cs-training.csv', index_col=0)
print(f"Raw dataset shape: {df_raw.shape[0]:,} observations, {df_raw.shape[1]} columns")

target_col = 'SeriousDlqin2yrs'
y_dist = df_raw[target_col].value_counts()
print(f"Target distribution (Class Imbalance):")
print(f"  Performing loans (0): {y_dist[0]:,} ({y_dist[0]/len(df_raw)*100:.2f}%)")
print(f"  Defaulted loans (1):  {y_dist[1]:,} ({y_dist[1]/len(df_raw)*100:.2f}%)")

df = df_raw.copy()

# Remove invalid age
df = df[df['age'] > 18].copy()

# Process past-due special codes (96, 98) -> impute with median
past_due_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]
for col in past_due_cols:
    clean_series = df[col].replace([96, 98], np.nan)
    df[col] = clean_series.fillna(clean_series.median())

# Clean MonthlyIncome and NumberOfDependents
df['MonthlyIncome_clean'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents_clean'] = df['NumberOfDependents'].fillna(0)

# Winsorize / Cap extreme outliers for robust econometric estimation
df['RevolvingUtilization_clean'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=2.0)
df['DebtRatio_clean'] = df['DebtRatio'].clip(upper=5.0)

# Feature list
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

feature_names_ru = {
    'RevolvingUtilization_clean': 'Утилизация кредитных карт',
    'age': 'Возраст заемщика',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Просрочки 30-59 дней',
    'DebtRatio_clean': 'Долговая нагрузка (Debt Ratio)',
    'MonthlyIncome_clean': 'Ежемесячный доход',
    'NumberOfOpenCreditLinesAndLoans': 'Число открытых кредитов',
    'NumberOfTimes90DaysLate': 'Тяжелые просрочки (90+ дней)',
    'NumberRealEstateLoansOrLines': 'Число ипотек / залогов',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Просрочки 60-89 дней',
    'NumberOfDependents_clean': 'Число иждивенцев'
}

X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"Clean dataset: {len(df):,} borrowers")
assert not X.isnull().any().any(), "X contains nulls!"

# Train / Test split (80/20) stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train sample: {len(X_train):,} borrowers")
print(f"Test sample:  {len(X_test):,} borrowers")

print("\n" + "="*70)
print("2. WHITE-BOX ECONOMETRIC BENCHMARK (LOGISTIC REGRESSION & VIF)")
print("="*70)

# VIF Calculation on standardized features to avoid intercept collinearity issues
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)

vif_df = pd.DataFrame()
vif_df['Feature'] = [feature_names_ru[col] for col in feature_cols]
vif_df['ColName'] = feature_cols
vif_df['VIF'] = [variance_inflation_factor(X_train_scaled.values, i) for i in range(len(feature_cols))]
print("\n--- Variance Inflation Factor (VIF) Analysis ---")
print(vif_df[['Feature', 'VIF']].to_string(index=False))

# Statsmodels Logit estimation
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

logit_sm = sm.Logit(y_train, X_train_const).fit(disp=False)
print("\n--- Logit Regression Summary ---")
print(logit_sm.summary())

# Odds Ratios and Confidence Intervals
odds_df = pd.DataFrame({
    'Feature': [feature_names_ru.get(col, col) for col in logit_sm.params.index],
    'Coefficient (beta)': logit_sm.params.values,
    'Std Err': logit_sm.bse.values,
    'z': logit_sm.tvalues.values,
    'p-value': logit_sm.pvalues.values,
    'Odds Ratio (OR)': np.exp(logit_sm.params.values),
    'CI 95% Lower': np.exp(logit_sm.conf_int()[0].values),
    'CI 95% Upper': np.exp(logit_sm.conf_int()[1].values)
}, index=logit_sm.params.index)

print("\n--- Odds Ratios (exp(beta)) ---")
print(odds_df[['Feature', 'Coefficient (beta)', 'Odds Ratio (OR)', 'p-value']].to_string())

# Predict probabilities with Logit
y_pred_prob_logit = logit_sm.predict(X_test_const)
auc_logit = roc_auc_score(y_test, y_pred_prob_logit)
gini_logit = 2 * auc_logit - 1
brier_logit = brier_score_loss(y_test, y_pred_prob_logit)
print(f"\nLogit Test Performance:")
print(f"  AUC-ROC:     {auc_logit:.4f}")
print(f"  Gini:        {gini_logit:.4f}")
print(f"  Brier Score: {brier_logit:.4f}")

print("\n" + "="*70)
print("3. ADVANCED AI SCORING (GRADIENT BOOSTING & ENSEMBLE)")
print("="*70)

# Train Gradient Boosting Classifier
gb_model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=42,
    class_weight='balanced'
)
gb_model.fit(X_train, y_train)

# Predictions
y_pred_prob_gb = gb_model.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_pred_prob_gb)
gini_gb = 2 * auc_gb - 1
brier_gb = brier_score_loss(y_test, y_pred_prob_gb)

print(f"Gradient Boosting Test Performance:")
print(f"  AUC-ROC:     {auc_gb:.4f} (Прирост к Logit: +{(auc_gb - auc_logit)*100:.2f} п.п.)")
print(f"  Gini:        {gini_gb:.4f} (Прирост к Logit: +{(gini_gb - gini_logit)*100:.2f} п.п.)")
print(f"  Brier Score: {brier_gb:.4f}")

# Permutation Feature Importance for Gradient Boosting on 10,000 sample for speed
print("\nCalculating Permutation Feature Importance...")
sample_idx = np.random.RandomState(42).choice(len(X_test), size=10000, replace=False)
perm_importance = permutation_importance(
    gb_model, X_test.iloc[sample_idx], y_test.iloc[sample_idx],
    n_repeats=3, random_state=42, scoring='roc_auc'
)

feat_imp_df = pd.DataFrame({
    'Feature': [feature_names_ru[col] for col in feature_cols],
    'ColName': feature_cols,
    'Importance_Mean': perm_importance.importances_mean,
    'Importance_Std': perm_importance.importances_std
}).sort_values(by='Importance_Mean', ascending=False)

print("\n--- AI Risk Hierarchy (Permutation Importance) ---")
print(feat_imp_df[['Feature', 'Importance_Mean', 'Importance_Std']].to_string(index=False))

print("\n" + "="*70)
print("4. BUSINESS THRESHOLD OPTIMIZATION (CUTOFF ECONOMICS)")
print("="*70)

# Financial parameters simulation (Kaspi / Central Asia consumer credit benchmark)
# Avg loan = 100,000 KZT
# LGD = 80% (Loss given default = 80,000 KZT per bad loan approved)
# Net margin on performing loan = 18% (18,000 KZT)
loan_amount = 100000
loss_fn = 0.80 * loan_amount # Cost of approving bad borrower (False Negative in risk detection)
gain_fp = 0.18 * loan_amount # Gain from approving good borrower

thresholds = np.linspace(0.02, 0.95, 120)
results = []

for th in thresholds:
    # If predicted probability of default >= th -> REJECT loan, else APPROVE
    y_pred_default = (y_pred_prob_gb >= th).astype(int)
    
    approved_mask = (y_pred_default == 0)
    approval_rate = approved_mask.mean()
    
    if approval_rate > 0:
        npl_rate = y_test[approved_mask].mean()
    else:
        npl_rate = 0.0
        
    approved_good = ((y_pred_default == 0) & (y_test == 0)).sum()
    approved_bad = ((y_pred_default == 0) & (y_test == 1)).sum()
    total_net_profit = (approved_good * gain_fp) - (approved_bad * loss_fn)
    
    results.append({
        'Threshold': th,
        'Approval_Rate': approval_rate * 100,
        'NPL_Rate': npl_rate * 100,
        'Net_Profit_Mln': total_net_profit / 1e6,
        'Approved_Good': approved_good,
        'Approved_Bad': approved_bad
    })

sim_df = pd.DataFrame(results)
best_row = sim_df.loc[sim_df['Net_Profit_Mln'].idxmax()]
th_05_row = sim_df.iloc[(sim_df['Threshold'] - 0.50).abs().argsort()[:1]].iloc[0]

print(f"Credit Policy Comparison (on Test set: {len(y_test):,} applications):")
print("-" * 70)
print(f"1) NAIVE BASELINE THRESHOLD (0.50):")
print(f"   Approval Rate: {th_05_row['Approval_Rate']:.1f}%")
print(f"   Portfolio NPL Rate: {th_05_row['NPL_Rate']:.2f}%")
print(f"   Net Profit: {th_05_row['Net_Profit_Mln']:.2f} M KZT")
print("-" * 70)
print(f"2) OPTIMAL BUSINESS THRESHOLD ({best_row['Threshold']:.2f}):")
print(f"   Approval Rate: {best_row['Approval_Rate']:.1f}%")
print(f"   Portfolio NPL Rate: {best_row['NPL_Rate']:.2f}%")
print(f"   Net Profit: {best_row['Net_Profit_Mln']:.2f} M KZT (+{best_row['Net_Profit_Mln'] - th_05_row['Net_Profit_Mln']:.2f} M KZT uplift!)")
print("-" * 70)

print("\n" + "="*70)
print("5. SAVING HIGH-RESOLUTION CHARTS")
print("="*70)

# Figure 1: ROC Curves Comparison
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_prob_logit)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_prob_gb)

ax.plot(fpr_logit, tpr_logit, label=f'Эконометрический Logit (AUC = {auc_logit:.3f}, Gini = {gini_logit:.3f})', color='#2980b9', lw=2)
ax.plot(fpr_gb, tpr_gb, label=f'ИИ-ансамбль Gradient Boosting (AUC = {auc_gb:.3f}, Gini = {gini_gb:.3f})', color='#c0392b', lw=2.5)
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Случайный классификатор (AUC = 0.500)')
ax.set_title('Разделяющая способность моделей кредитного скоринга (ROC-кривая)', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate (Доля ошибочно заблокированных надежных заемщиков)', fontsize=11)
ax.set_ylabel('True Positive Rate (Доля выявленных дефолтов / Чувствительность)', fontsize=11)
ax.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()
fig.savefig('output/figures/roc_curve_comparison.png')
plt.close(fig)
print("Saved: output/figures/roc_curve_comparison.png")

# Figure 2: Odds Ratios
fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
odds_plot_df = odds_df.drop('const', errors='ignore').sort_values(by='Odds Ratio (OR)', ascending=True)
y_pos = np.arange(len(odds_plot_df))
ax.errorbar(
    odds_plot_df['Odds Ratio (OR)'], y_pos,
    xerr=[odds_plot_df['Odds Ratio (OR)'] - odds_plot_df['CI 95% Lower'], odds_plot_df['CI 95% Upper'] - odds_plot_df['Odds Ratio (OR)']],
    fmt='o', color='#2980b9', ecolor='#7f8c8d', elinewidth=2, capsize=4, markersize=7
)
ax.axvline(1.0, color='#e74c3c', linestyle='--', lw=1.5, alpha=0.8, label='Нейтральный эффект (OR = 1.0)')
ax.set_yticks(y_pos)
ax.set_yticklabels(odds_plot_df['Feature'], fontsize=10)
ax.set_xlabel('Отношение шансов дефолта (Odds Ratio, логарифмическая шкала)', fontsize=11)
ax.set_xscale('log')
ax.set_title('Эконометрический анализ факторов риска: Отношение шансов (95% ДИ)', fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
fig.savefig('output/figures/odds_ratios.png')
plt.close(fig)
print("Saved: output/figures/odds_ratios.png")

# Figure 3: Feature Importance
fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
feat_imp_sorted = feat_imp_df.sort_values(by='Importance_Mean', ascending=True)
y_pos = np.arange(len(feat_imp_sorted))
bars = ax.barh(y_pos, feat_imp_sorted['Importance_Mean'], xerr=feat_imp_sorted['Importance_Std'],
        color='#27ae60', edgecolor='#1e8449', alpha=0.85, capsize=4)
ax.set_yticks(y_pos)
ax.set_yticklabels(feat_imp_sorted['Feature'], fontsize=10)
ax.set_xlabel('Снижение ROC-AUC при перестановочной изоляции признака (Permutation Importance)', fontsize=11)
ax.set_title('Иерархия ключевых факторов кредитного риска в ИИ-модели', fontsize=13, fontweight='bold', pad=15)
for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.4f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9)
plt.tight_layout()
fig.savefig('output/figures/feature_importance.png')
plt.close(fig)
print("Saved: output/figures/feature_importance.png")

# Figure 4: Financial & Business Cutoff Optimization
fig, ax1 = plt.subplots(figsize=(10.5, 6), dpi=300)

color1 = '#2980b9'
color2 = '#d35400'
color3 = '#27ae60'

ax1.plot(sim_df['Threshold'], sim_df['Approval_Rate'], color=color1, lw=2.2, label='Уровень одобрения (Approval Rate, %)')
ax1.plot(sim_df['Threshold'], sim_df['NPL_Rate'], color=color2, lw=2.2, linestyle='-.', label='Уровень NPL в портфеле (%)')
ax1.set_xlabel('Порог отсечения вероятности дефолта (Cutoff Threshold)', fontsize=11)
ax1.set_ylabel('Проценты (%)', color='#2c3e50', fontsize=11)
ax1.set_xlim(0.02, 0.90)
ax1.set_ylim(0, 105)

ax2 = ax1.twinx()
ax2.plot(sim_df['Threshold'], sim_df['Net_Profit_Mln'], color=color3, lw=2.8, label='Ожидаемая чистая прибыль (млн KZT)')
ax2.set_ylabel('Прибыль портфеля (млн KZT)', color=color3, fontsize=11)
ax2.grid(False)

# Highlight best threshold
ax1.axvline(best_row['Threshold'], color='#c0392b', linestyle=':', lw=2, label=f"Оптимальный порог ({best_row['Threshold']:.2f})")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, fontsize=10)

plt.title('Бизнес-оптимизация кредитной политики: Компромисс выручки и NPL', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig('output/figures/cutoff_optimization.png')
plt.close(fig)
print("Saved: output/figures/cutoff_optimization.png")

# Figure 5: Confusion Matrices
fig, (ax_std, ax_opt) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

opt_th = best_row['Threshold']
cm_std = confusion_matrix(y_test, (y_pred_prob_gb >= 0.50).astype(int))
cm_opt = confusion_matrix(y_test, (y_pred_prob_gb >= opt_th).astype(int))

sns.heatmap(cm_std, annot=True, fmt=',d', cmap='Blues', ax=ax_std, cbar=False, annot_kws={'fontsize': 11})
ax_std.set_title(f'Стандартный порог (0.50)\nОдобрено {th_05_row["Approval_Rate"]:.1f}%, NPL = {th_05_row["NPL_Rate"]:.2f}%', fontsize=11, fontweight='bold')
ax_std.set_xlabel('Классификация модели (1 = Отказ по риску)', fontsize=10)
ax_std.set_ylabel('Реальный исход (1 = Дефолт)', fontsize=10)

sns.heatmap(cm_opt, annot=True, fmt=',d', cmap='Greens', ax=ax_opt, cbar=False, annot_kws={'fontsize': 11})
ax_opt.set_title(f'Оптимизированный порог ({opt_th:.2f})\nОдобрено {best_row["Approval_Rate"]:.1f}%, NPL = {best_row["NPL_Rate"]:.2f}%', fontsize=11, fontweight='bold')
ax_opt.set_xlabel('Классификация модели (1 = Отказ по риску)', fontsize=10)
ax_opt.set_ylabel('Реальный исход (1 = Дефолт)', fontsize=10)

plt.suptitle('Сравнение матриц ошибок при наивном и оптимальном пороге', fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig('output/figures/confusion_matrices.png')
plt.close(fig)
print("Saved: output/figures/confusion_matrices.png")

# Save tables
odds_df.to_csv('output/econometric_odds_ratios.csv')
feat_imp_df.to_csv('output/feature_importance.csv', index=False)
vif_df.to_csv('output/vif_analysis.csv', index=False)
sim_df.to_csv('output/business_threshold_simulation.csv', index=False)

print("\n" + "="*70)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("="*70)
