# AI-Powered Credit Scoring: Resolving the Speed vs. Risk Dilemma

**Course:** ECO526: AI for Financial Technologies  
**Focus Case:** The Kaspi.kz Super App Ecosystem (Fintech, Marketplace, Payments)  
**Deliverable:** 33-Slide Interactive HTML Presentation, Empirical Models & Econometric Benchmark  

---

## 🚀 Live Interactive Presentation

👉 **[Open Live Presentation Slides](https://<YOUR-USERNAME>.github.io/<REPO-NAME>/)**  
*(Works out-of-the-box on any browser, laptop, or mobile screen)*

* **Keyboard Navigation:** Use `←` / `→` or `Space` to navigate slides.
* **Fullscreen:** Press `F` to toggle full-screen presentation mode.
* **Interactive Features:** Click on any of the 5 charts to zoom; click on Q&A cards to reveal defense answers.

---

## 📁 Repository Structure

* `index.html` / `presentation.html` — Self-contained, responsive 33-slide interactive presentation (Light-Tech design, Space Grotesk + Plus Jakarta Sans).
* `ai_credit_scoring_case_study.md` — Complete academic paper draft with speaker scripts, literature citations, and Q&A defense prep.
* `build_credit_model.py` — Python script training both models on 150,000 real borrowers:
  1. *White-Box Benchmark:* Logistic Regression (`statsmodels.Logit`), VIF test, Odds Ratios.
  2. *Predictive AI:* Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`), Permutation Importance.
* `generate_english_charts.py` — High-resolution 300-DPI English figure generator.
* `output/figures/` — Generated high-res charts:
  * `roc_curve_comparison.png` — Discrimination power comparison (Logit AUC: 0.856 vs. AI AUC: 0.868, +2.46 pp Gini).
  * `odds_ratios.png` — Econometric risk drivers with 95% Confidence Intervals.
  * `feature_importance.png` — Permutation feature importance loss ranking.
  * `cutoff_optimization.png` — Financial profit optimization curve (+33.3M KZT net margin uplift).
  * `confusion_matrices.png` — Policy comparison (Naive 0.50 threshold vs. Optimal 0.73 cutoff).
* `output/` — CSV summary tables (`vif_analysis.csv`, `econometric_odds_ratios.csv`, `feature_importance.csv`, `business_threshold_simulation.csv`).

---

## 📊 Key Empirical & Business Takeaways

1. **Subsecond Automation is Safe:** Gradient Boosting achieves **AUC = 0.8681** (Gini = 73.62%), outperforming traditional linear scorecards by **+2.46 pp**.
2. **Business Cutoff Economics:** Shifting from the naive 0.50 threshold to an optimal 0.73 cutoff increases approval rates from 75.4% to **89.5%** while keeping portfolio NPL at 3.24% (well within Kaspi's 5.4% norm), generating **+33.34M KZT profit uplift** per 30,000 applications.
3. **Regulatory Explainability (XAI):** Game-theoretic SHAP decomposition provides compliant adverse action notices and protects against algorithmic bias (avoiding the Apple Card / Goldman Sachs scandal).
4. **Drift Resilience:** Continuous monitoring of the Population Stability Index (PSI) prevents model failure during inflation or interest rate shocks (avoiding Zillow's $880M iBuying collapse).
