# CV Bullet Points & Thesis Application Highlights: Early Loan Default Risk Detection

This document provides ATS-friendly, technically precise bullet points and concise project descriptions for Master's thesis applications, CVs/resumes, and technical interviews in financial risk modeling, fraud/behavioral analytics, and machine learning.

Every phrase and metric has undergone a line-by-line audit against the **Section 12 Claims Boundary Matrix** to guarantee that no unsupported causal, deployment, or generalization claims are made.

---

## 1. Primary 3-Bullet Set (Recommended for CV / Resume)

* **Engineered a leakage-safe temporal feature pipeline** on 1.05M banking transactions, extracting 13 post-disbursement behavioral features across 30–90-day windows and auditing major target-leakage and calendar-censoring shortcuts with 9 automated tests.
* **Achieved +0.18 to +0.36 PR-AUC improvement** by adding post-disbursement transaction dynamics to an origination/pre-loan baseline across locked 25-fold paired CV splits (N=234), with 92–100% paired win rates across Logistic Regression, Random Forest, and HistGradientBoosting.
* **Demonstrated model-dependent information arrival** across 30–90-day windows, with substantial signal available by 60 days for Logistic Regression and HistGradientBoosting and additional gains by 90 days for non-linear models; incremental lift also persisted among borrowers remaining strictly solvent throughout the observation window.

---

## 2. Compact 2-Bullet Set (Space-Constrained Option)

* **Developed a leakage-safe behavioral credit risk pipeline** on 1.05M transactions ($N=234$ completed loans), achieving $+0.18$ to $+0.36$ PR-AUC lift (92–100% paired win rates across 25 identical CV splits) by adding post-disbursement dynamics to an origination baseline.
* **Demonstrated model-dependent information arrival** across 30–90-day windows with incremental lift persisting among strictly solvent borrowers, supported by chronological holdout testing and secondary 656-loan sensitivity analysis.

---

## 3. Thesis Application / Statement of Purpose Paragraph

> **Early Loan Default Risk Detection from Transaction Behaviour (Czech Commercial Bank Data)**:  
> Conducted an applied machine learning investigation into early post-disbursement credit risk surveillance on 1.05M banking transactions. To evaluate whether early account activity provides incremental predictive information beyond loan origination scorecards, I engineered a leakage-safe temporal pipeline across 30, 60, and 90-day observation windows, identifying and auditing major target-leakage and calendar-censoring shortcuts. Across identical 25-fold paired cross-validation splits on a primary completed-loan cohort ($N=234$, $13.25\%$ default base rate), adding post-disbursement transaction features improved PR-AUC by $+0.18$ to $+0.36$ (92–100% paired win rates) across linear and non-linear models. Information progression analysis revealed model-dependent arrival patterns, where linear models capture substantial predictive signal within 60 days while gradient-boosted trees capture additional interaction gains through 90 days. Robustness analyses showed that incremental lift persists among borrowers remaining strictly non-negative throughout the window, with temporal robustness supported on a chronological holdout and consistent lift observed in a secondary 656-loan sensitivity cohort.

---

## 4. Line-by-Line Claims Boundary Audit

Every statement in the bullet points has been audited against the analytical verification criteria:

| Bullet Statement | Audit Criteria | Audit Status | Verification & Traceability |
| :--- | :--- | :---: | :--- |
| *"Engineered a leakage-safe temporal feature pipeline on 1.05M banking transactions..."* | Supported & Traceable | **PASS** | Exact count from Czech Bank dataset (1,056,320 transactions). |
| *"...extracting 13 post-disbursement behavioral features across 30–90-day windows..."* | Supported & Traceable | **PASS** | Exactly 13 dynamic features implemented across 30d, 60d, 90d. |
| *"...and auditing major target-leakage and calendar-censoring shortcuts with 9 automated tests."* | Supported & De-biased | **PASS** | Full-lifetime balance separation (100%) and duration shortcut (99.4%) verified; 9 pytest tests pass. |
| *"Achieved +0.18 to +0.36 PR-AUC improvement... across locked 25-fold paired CV splits (N=234)..."* | Numerical traceability & Paired framing | **PASS** | Traceable to Table 1 (Ablation Results): Mean $\Delta$ ranges from $+0.184$ (LogReg 30d) to $+0.359$ (HGB 90d). Framed as paired CV splits, not independent datasets. |
| *"...with 92–100% paired win rates across Logistic Regression, Random Forest, and HistGradientBoosting."* | Paired win rates | **PASS** | Traceable to Table 1 (Ablation Results): Fold win rates are 23/25 (92%), 24/25 (96%), or 25/25 (100%). |
| *"Demonstrated model-dependent information arrival across 30–90-day windows..."* | Criterion 2, 4 & 9: No universal optimal window | **PASS** | Preserves model dependence: LogReg plateaus at 60d; HGB gains at 90d; RF flat 30-60d then gains at 90d. Avoids claiming universal window. |
| *"...with substantial signal available by 60 days for Logistic Regression and HistGradientBoosting and additional gains by 90 days for non-linear models..."* | Criterion 2 & 4: Model dependence | **PASS** | Exactly matches the approved synthesis wording. |
| *"...incremental lift also persisted among borrowers remaining strictly solvent throughout the observation window."* | Criterion 2 & 7: Non-causal solvent framing | **PASS** | Uses "persisted" rather than "proves"; reflects positive lift on $\min(\text{balance}) \ge 0$ cohort (Mean $\Delta \approx +0.14\text{ to }+0.27$). Zero causal language. |
| *No causal words ("causes", "protective", "cash burn velocity", "prevents")* | Criterion 2 & 11: Non-causal language | **PASS** | Zero causal terminology present. |
| *No deployment or threshold guarantees* | Criterion 10: No operational overstatement | **PASS** | Zero deployment or tranche policy claims included. |

---

## 5. Technical Keywords for Applicant Tracking Systems (ATS)

* **Domain & Risk**: Credit Risk Modeling, Behavioral Risk Surveillance, Early Default Detection, Financial Telemetry, Retail Banking Analytics, Rare-Event Classification, Imbalanced Data.
* **Methodology & Rigor**: Precision-Recall AUC (PR-AUC), Temporal Leakage Prevention, Identical Paired Cross-Validation, Chronological Out-of-Time (OOT) Holdout, Right-Censoring Resolution, Right-Truncation Quarantine, Solvent-Subcohort Stress Testing, Model Ablation Studies.
* **Modeling & Tools**: Logistic Regression (L2 Regularized), Random Forest, HistGradientBoosting, Scikit-Learn, Pandas, NumPy, Pytest Unit Testing.
