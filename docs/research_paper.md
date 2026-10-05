# Early Loan Default Risk Detection from Transaction Behaviour: Research Synthesis & Empirical Evaluation

**Working Title**: Early Loan Default Risk Detection from Transaction Behaviour  
**Author**: Applied Data Science & Risk Analytics Portfolio  
**Dataset**: PKDD'99 Financial Data (Czech Bank / Berka Dataset)  
**Status**: Complete — Final Research Synthesis  
**Code Repository**: `loan-default-risk`  

---

## Abstract

This study investigates whether early post-disbursement transaction behaviour provides incremental predictive information for distinguishing loans that eventually default from those successfully repaid, and how predictive capacity unfolds across 30, 60, and 90-day observation windows. Using the historical PKDD'99 Czech Bank database, we evaluate a primary headline cohort of completed contracts ($N=234$; 203 repaid, 31 defaulted; $13.25\%$ base rate) where ground truth is observed without right-censoring. We implement a leakage-safe temporal feature engineering pipeline and evaluate paired ablations between a historical/origination baseline (Model A, 13 features) and a dynamic post-disbursement behavioural model (Model B, 26 features) across identical 5-fold $\times$ 5-repeat stratified cross-validation partitions (25 evaluation units). 

Model B achieves substantial and consistent improvements in Precision-Recall AUC (PR-AUC) across all evaluated model families, with mean paired improvements ranging from $+0.184$ to $+0.359$ and paired win rates between $92\%$ and $100\%$. Information arrival analysis reveals that the timing of useful predictive information is model-dependent: linear models (Logistic Regression) capture substantial predictive lift within 60 days ($+0.040$ PR-AUC over 30 days) and largely plateau by 90 days ($+0.009$, $44\%$ win rate), whereas tree-based gradient boosting captures additional non-linear interaction signal between 60 and 90 days ($+0.061$, $72\%$ win rate). Subcohort stress testing on borrowers maintaining strictly non-negative balances throughout the observation window shows that incremental predictive lift persists (Mean $\Delta_{\text{PR-AUC}} \approx +0.14$ to $+0.27$, win rates $76\%\text{--}96\%$), providing evidence that the signal is not solely attributable to an active unauthorized overdraft. Chronological out-of-time evaluation on 1996–1997 loans ($N=78$, 9 defaults) provides supporting evidence of temporal robustness, while a secondary sensitivity analysis on an expanded cohort ($N=656$, 75 defaults) demonstrates that the incremental behavioural lift is not an artifact of filtering to completed contracts. We conclude with a formal claims boundary separating defensible portfolio highlights from technical report limitations.

---

## 1. Introduction & Research Problem

In credit risk management, underwriting models traditionally rely on static applicant characteristics, credit bureau scores, and pre-loan banking history established at or before the loan origination date. Once funds are disbursed, however, a lender with ongoing account-monitoring access observes high-frequency transaction dynamics—such as payroll inflows, cash withdrawals, standing order executions, and daily balance trajectories. 

While late-stage credit risk management focuses on collections and recovery following formal delinquency (e.g., 90 days past due), **early post-disbursement behavioral monitoring** represents a proactive intermediate stage. This study addresses the following central research question:

> **Using only transaction behaviour observed after loan disbursement, can early behavioural patterns distinguish loans that eventually default from loans that are successfully repaid, and how early does useful predictive information become available?**

### Formal Research Hypotheses
1. **Hypothesis 1 (Incremental Information Value)**: Post-disbursement dynamic transaction features extracted from an early observation window ($W \in \{30, 60, 90\}$ days) provide incremental predictive power for distinguishing eventual default from successful repayment beyond a baseline containing static origination attributes and pre-loan account history ($\text{PR-AUC}_{\text{Model B}} > \text{PR-AUC}_{\text{Model A}}$).
2. **Hypothesis 2 (Model-Dependent Information Progression)**: Predictive capacity accumulates across post-disbursement time horizons, but the rate of information arrival and plateau timing vary across model families rather than adhering to a single universal optimal window.
3. **Hypothesis 3 (Sub-Overdraft Behavioural Signal)**: The incremental predictive information in post-disbursement transaction patterns is not solely an artifact of accounts having already breached an unauthorized overdraft within the window; positive predictive lift persists among borrowers who maintain non-negative balances throughout the observation period.

---

## 2. Portfolio Context & Methodological Throughline

This investigation forms the fourth project in an applied analytics portfolio, complementing three previous empirical studies:
1. **SCANIA Industrial Telemetry Analytics**: Demonstrates physical condition monitoring, equipment degradation dynamics, vehicle-disjoint validation, and false-alarm vs. lead-time trade-offs.
2. **FAERS Drug Safety Signal Detection**: Demonstrates large-scale observational pharmacovigilance, disproportionality statistics (PRR/ROR), reporting bias adjustment, and negative-control validation.
3. **Nordic Grid Electricity Forecasting**: Demonstrates physical time-series forecasting, chronological cross-validation, and multi-step lead-time leakage prevention.

### Methodological Throughline: Deconfounding Apparent Signals
Across both the FAERS pharmacovigilance and financial risk analytics projects, a central methodological challenge is **distinguishing a genuine predictive or safety signal from an apparent signal driven by a plausible confound, shortcut, or reporting artefact**.
* In the **FAERS project**, high reporting rates for widespread drugs can produce inflated signal metrics without reflecting true pharmacological toxicity; negative-control drug-event pairs are utilized to verify that background noise does not generate false alarms.
* In this **banking risk project**, an apparent behavioral signal could easily be trivialized if the model simply flags accounts that have already crossed into an unauthorized negative balance. We address this by conducting a **solvent-subcohort stress test**, evaluating whether the behavioral model retains predictive lift among borrowers whose balances remain strictly non-negative throughout the observation period.
* While the statistical procedures differ (disproportionality negative controls vs. subcohort stratification), both methods address the same fundamental scientific requirement: **challenging whether an observed predictive association is merely explained by an obvious alternative mechanism**.

---

## 3. Data Architecture, Label Semantics & Censoring

The project utilizes the PKDD'99 Financial Data (Berka dataset), capturing operations from a Czech commercial bank between 1993 and 1998 across 682 loans, 5,369 clients, 4,500 accounts, and 1,056,320 transactions.

### 3.1 Outcome Label Semantics
The bank recorded loan status using four administrative categories:
* **Status `A`** ($N=203$): Finished contract, loan successfully repaid without issues $\to$ **Repaid / Good (Class 0)**.
* **Status `B`** ($N=31$): Finished contract, loan not repaid / in default $\to$ **Default / Bad (Class 1)**.
* **Status `C`** ($N=403$): Running contract, payments on schedule as of December 1998 $\to$ **Right-Censored**.
* **Status `D`** ($N=45$): Running contract, client in debt / default in progress $\to$ **Active Default (Class 1)**.

### 3.2 Primary Headline Cohort ($N=234$)
The headline empirical analysis is conducted strictly on completed contracts:
$$\text{Primary Cohort} = \{ \text{Status } A \} \cup \{ \text{Status } B \} \quad (N = 234; \; 203 \text{ Good}, \; 31 \text{ Default})$$
* **Default Base Rate**: $13.25\%$ (31 / 234).
* **Ground-Truth Certainty**: The completed cohort is prioritized as the primary headline dataset because every contract has reached maturity. The ultimate outcome is definitively known without right-censoring.

### 3.3 Right-Truncation & Calendar Censoring
The transaction database terminates on **1998-12-31**. For any observation window of duration $W \in \{30, 60, 90\}$ days, a loan originated at date $T_{\text{loan}}$ requires transactions through $T_{\text{loan}} + W$. If $T_{\text{loan}} + W > \text{1998-12-31}$, the post-disbursement window is truncated by the dataset export cutoff.
* **Primary Completed Cohort ($N=234$)**: The latest completed loan was originated in December 1997. Across all windows ($W \in \{30, 60, 90\}$ days), exactly **0 loans are right-truncated**. All 234 loans possess complete transaction histories across all three windows.
* **Expanded Cohort ($N=682$)**: Includes running loans disbursed through December 1998. At $W=90$ days, loans disbursed after October 2, 1998 lack a full 90-day window. Exactly **26 loans are right-truncated**. Rather than imputing missing days with zero activity, these 26 loans are explicitly quarantined, leaving $N=656$ loans (581 good, 75 default; $11.43\%$ base rate).
* **Censoring Status of Running Loans**: Because Status `C` contracts were active as of December 1998, their ultimate repayment status after 1998 is unobserved. Consequently, the expanded cohort serves strictly as a **secondary sensitivity analysis** to test whether findings transfer beyond completed contracts, rather than replacing the primary ground truth.

---

## 4. Leakage-Prevention Methodology & De-Biasing

In observational financial data, unmonitored feature extraction frequently introduces catastrophic data leakage. We implemented four architectural controls to ensure empirical integrity:

```mermaid
flowchart TD
    subgraph RawData["Raw Banking Transactions (1993 - 1998)"]
        Pre["Pre-Loan Transactions<br/>(trans_date < loan_date)"]
        Disb["Day 0 Disbursement<br/>(trans_date == loan_date)"]
        Post["Post-Disbursement Window<br/>(loan_date <= trans_date <= loan_date + W)"]
        Future["Post-Window / Lifetime Data<br/>(trans_date > loan_date + W)"]
    end

    subgraph FeatureSpace["Leakage-Isolated Feature Space"]
        ModelA["Model A Features (13)<br/>Static Origination + Pre-Loan Controls"]
        ModelB["Model B Features (26)<br/>Model A + Windowed Dynamics"]
    end

    Pre -->|Summarized strictly as static history| ModelA
    Disb -->|Valid post-disbursement credit/debit| ModelB
    Post -->|Extracted within [0, W] window| ModelB
    Future -.->|QUARANTINED / FORBIDDEN| FeatureSpace
```

### 4.1 Temporal Demarcation & Day 0 Rule
* **Post-Disbursement Window**: Defined strictly as $T_{\text{loan}} \le \text{trans\_date} \le T_{\text{loan}} + W$.
* **Day 0 Inclusion**: Transactions occurring on the loan origination date ($T_{\text{loan}}$) are included in the post-disbursement window. Domain audit confirms that the loan disbursement itself appears as an incoming credit on Day 0, immediately followed in many instances by client cash withdrawals or fee debits.
* **Pre-Loan Isolation**: Transactions prior to disbursement ($\text{trans\_date} < T_{\text{loan}}$) are forbidden from windowed behavioral aggregations. They enter the model exclusively through two static pre-loan control features: `feat_had_pre_loan_overdraft` and `feat_pre_loan_tx_count`.

### 4.2 Auditing Target Leakage & Censoring Shortcuts
Two prominent data shortcuts were identified and audited:
1. **Full-Lifetime Minimum Balance Leakage**:
   * Computing the minimum account balance across the contract's entire lifetime yields a feature where $\min(\text{balance}) < 0$ achieves **100% precision and 100% recall** (100% separation) in separating Good from Default loans across all 234 completed contracts.
   * *De-biasing*: Full-lifetime minimum balance perfectly separates the observed good and bad loan-status labels in this dataset, making it a target-leakage shortcut for early prediction. Relying on full-lifetime balance substitutes a retrospective outcome-correlated marker for an early risk prediction signal.
2. **Calendar Duration Censoring Shortcut**:
   * In a naive model across all 682 loans, loan duration predicts loan status with $99.4\%$ accuracy because loans with durations of 48 or 60 months originated after 1994 could not mature before the December 1998 data export.
   * *De-biasing*: Loan duration was identified as a calendar-censoring shortcut when separating completed from running contracts across the full dataset, but it does not provide the same predictive shortcut for default within the completed A/B cohort. `duration` is explicitly retained as one of the 11 static Model A origination features, but is recognized as non-predictive of default in the completed cohort.

### 4.3 Preprocessing Isolation & Unit Testing
* All continuous feature scalers (`StandardScaler`) and imputation parameters are fitted strictly inside each cross-validation training fold and applied to out-of-fold validation sets.
* The feature extraction pipeline was validated by a test suite of 9 automated pytest unit tests (`tests/test_leakage.py`), enforcing temporal boundaries, aggregate prohibitions, and entity disjointness.

---

## 5. Experimental Design & Model Ablations

### 5.1 Reusable Identical Cross-Validation Folds
To ensure that differences in performance reflect genuine information gains rather than sample partitioning noise, fold assignments were generated **once** and locked:
* **Protocol**: 5-fold `StratifiedKFold` $\times$ 5 repeats = **25 evaluation units**.
* **Cohort Lock**: Formed on the primary cohort ($N=234$, 31 defaults). Every test fold contains exactly 46 or 47 loans and 6 or 7 defaults (median: 6.0).
* **Mandatory Reuse**: The exact same 25 fold assignments were reused across all three observation windows ($W \in \{30, 60, 90\}$), both feature spaces (Model A vs. Model B), and all three machine learning algorithms.

### 5.2 Feature Space Ablation
* **Model A (Historical Baseline, 13 features)**:
  * *Static Origination (11)*: `amount`, `duration`, `payments`, `payment_to_amount_ratio`, `account_age_days`, `client_age_years`, `client_is_female`, district `salary`, `unemployment`, `crimes`, `entrepreneurs`.
  * *Pre-Loan History (2)*: `feat_had_pre_loan_overdraft`, `feat_pre_loan_tx_count`.
* **Model B (Dynamic Behavioural Model, 26 features)**:
  * All 13 Model A features.
  * *Post-Disbursement Windowed Dynamics (13)*: `trans_count`, `inflow_total`, `outflow_total`, `net_cash_flow`, `outflow_to_inflow_ratio`, `cash_withdrawal_amount`, `cash_withdrawal_ratio`, `balance_mean`, `balance_min`, `balance_slope`, `had_negative_balance`, `had_sanction_interest`, `min_bal_to_payment_ratio`.

### 5.3 Evaluated Model Families
1. **Logistic Regression (L2 Regularized)**: Linear baseline with standard scaling and nested 5-fold cross-validation tuning $C \in \{0.01, 0.1, 1.0, 10.0\}$ optimizing Average Precision with `class_weight='balanced'`.
2. **Random Forest**: Non-linear ensemble with shallow depth (`max_depth=4`, `min_samples_leaf=5`, `n_estimators=100`, `class_weight='balanced_subsample'`).
3. **HistGradientBoosting**: Non-linear boosting ensemble with constrained complexity (`max_iter=100`, `max_leaf_nodes=15`, `min_samples_leaf=10`, `class_weight='balanced'`).

### 5.4 Evaluation Metrics
Given severe class imbalance ($13.25\%$ default base rate), the primary evaluation metric is **Precision-Recall AUC (PR-AUC / Average Precision)**. Secondary metrics include ROC-AUC and Brier Score. For every evaluation split $k \in \{1, \dots, 25\}$, we compute the paired delta:
$$\Delta_{\text{PR-AUC}}^{(k)} = \text{PR-AUC}_{\text{Model B}}^{(k)} - \text{PR-AUC}_{\text{Model A}}^{(k)}$$
Paired win rate is defined as the proportion of splits where $\Delta^{(k)} > 0$. Statistical significance is assessed via the Wilcoxon signed-rank test.

---

## 6. Empirical Results & Findings

### 6.1 Primary Ablation Results: Model A vs Model B ($N=234$)
Table 1 presents the performance of Model A and Model B across the 25 evaluation splits.

#### Table 1: Model A vs Model B Performance on Primary Completed Cohort ($N=234$)

| Model Family | Window | Model A PR-AUC | Model B PR-AUC | Mean $\Delta$ | Median $\Delta$ | Std $\Delta$ | Paired Win Rate | Wilcoxon $p$-value | Model A ROC-AUC | Model B ROC-AUC | Model A Brier | Model B Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **30d** | $0.524 \pm 0.117$ | $0.708 \pm 0.124$ | **+0.184** | +0.192 | 0.160 | **92.0%** (23/25) | $1.0 \times 10^{-5}$ | $0.755 \pm 0.088$ | $0.863 \pm 0.076$ | $0.179 \pm 0.026$ | $0.135 \pm 0.023$ |
| **Logistic Regression** | **60d** | $0.524 \pm 0.117$ | $0.748 \pm 0.109$ | **+0.224** | +0.265 | 0.136 | **96.0%** (24/25) | $1.2 \times 10^{-7}$ | $0.755 \pm 0.088$ | $0.886 \pm 0.062$ | $0.179 \pm 0.026$ | $0.109 \pm 0.024$ |
| **Logistic Regression** | **90d** | $0.524 \pm 0.117$ | $0.757 \pm 0.101$ | **+0.233** | +0.251 | 0.128 | **96.0%** (24/25) | $1.2 \times 10^{-7}$ | $0.755 \pm 0.088$ | $0.886 \pm 0.057$ | $0.179 \pm 0.026$ | $0.112 \pm 0.020$ |
| **HistGradientBoosting** | **30d** | $0.333 \pm 0.116$ | $0.613 \pm 0.165$ | **+0.280** | +0.290 | 0.194 | **96.0%** (24/25) | $2.0 \times 10^{-6}$ | $0.678 \pm 0.098$ | $0.809 \pm 0.123$ | $0.139 \pm 0.025$ | $0.092 \pm 0.032$ |
| **HistGradientBoosting** | **60d** | $0.333 \pm 0.116$ | $0.631 \pm 0.141$ | **+0.298** | +0.277 | 0.158 | **100.0%** (25/25) | $6.0 \times 10^{-8}$ | $0.678 \pm 0.098$ | $0.833 \pm 0.106$ | $0.139 \pm 0.025$ | $0.088 \pm 0.026$ |
| **HistGradientBoosting** | **90d** | $0.333 \pm 0.116$ | $0.692 \pm 0.127$ | **+0.359** | +0.335 | 0.168 | **96.0%** (24/25) | $1.2 \times 10^{-7}$ | $0.678 \pm 0.098$ | $0.872 \pm 0.082$ | $0.139 \pm 0.025$ | $0.080 \pm 0.021$ |
| **Random Forest** | **30d** | $0.351 \pm 0.131$ | $0.602 \pm 0.165$ | **+0.251** | +0.232 | 0.172 | **92.0%** (23/25) | $1.5 \times 10^{-6}$ | $0.694 \pm 0.111$ | $0.816 \pm 0.095$ | $0.138 \pm 0.014$ | $0.105 \pm 0.011$ |
| **Random Forest** | **60d** | $0.351 \pm 0.131$ | $0.589 \pm 0.150$ | **+0.238** | +0.270 | 0.165 | **92.0%** (23/25) | $2.0 \times 10^{-6}$ | $0.694 \pm 0.111$ | $0.843 \pm 0.095$ | $0.138 \pm 0.014$ | $0.099 \pm 0.012$ |
| **Random Forest** | **90d** | $0.351 \pm 0.131$ | $0.628 \pm 0.141$ | **+0.278** | +0.288 | 0.154 | **96.0%** (24/25) | $1.2 \times 10^{-7}$ | $0.694 \pm 0.111$ | $0.861 \pm 0.090$ | $0.138 \pm 0.014$ | $0.093 \pm 0.014$ |

> [!NOTE]
> **Statistical Independence Caveat**: While Wilcoxon signed-rank tests demonstrate significance ($p < 10^{-5}$ across all comparisons), the 25 evaluation splits originate from 5 repeats of 5-fold CV and are not statistically independent. The robust primary evidence of superiority lies in the high paired win rates ($92\%\text{--}100\%$) across identical test partitions.

### 6.2 Information Progression Across Windows (30d vs 60d vs 90d)
Table 2 reports the paired differences between observation windows for Model B across the identical 25 folds.

#### Table 2: Paired Differences Between Windows (Model B)

| Model Family | Window Comparison | Mean $\Delta$ | Median $\Delta$ | Std $\Delta$ | Paired Win Rate | Wilcoxon $p$-value | Empirical Progression Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **60d vs 30d** | **+0.040** | **+0.030** | 0.090 | **72.0%** (18/25) | **0.039** | Statistically significant linear signal gain |
| **Logistic Regression** | **90d vs 60d** | **+0.009** | **-0.017** | 0.080 | 44.0% (11/25) | 0.615 | **Signal plateau**: diminishing returns by 90d |
| **Logistic Regression** | **90d vs 30d** | **+0.049** | **+0.025** | 0.110 | 56.0% (14/25) | 0.120 | Net positive progression over 30d |
| **HistGradientBoosting** | **60d vs 30d** | **+0.018** | +0.031 | 0.167 | 60.0% (15/25) | 0.578 | Gradual non-linear improvement |
| **HistGradientBoosting** | **90d vs 60d** | **+0.061** | **+0.072** | 0.136 | **72.0%** (18/25) | **0.017** | Non-linear interaction gain at 90d |
| **HistGradientBoosting** | **90d vs 30d** | **+0.079** | **+0.058** | 0.184 | 60.0% (15/25) | 0.063 | Cumulative progression over 30d |
| **Random Forest** | **60d vs 30d** | -0.013 | -0.024 | 0.133 | 40.0% (10/25) | 0.596 | Flat performance across 30d to 60d |
| **Random Forest** | **90d vs 60d** | **+0.040** | **+0.048** | 0.093 | **64.0%** (16/25) | **0.039** | Tree splitting gain from expanded window |
| **Random Forest** | **90d vs 30d** | **+0.026** | +0.005 | 0.150 | 52.0% (13/25) | 0.542 | Modest net progression over 30d |

#### Empirical Synthesis of Progression
The evidence demonstrates that **the timing of useful predictive information is model-dependent**:
* For linear models (Logistic Regression), substantial predictive information is detectable within 60 days, with minimal incremental gain achieved by extending observation to 90 days.
* For gradient boosting (HistGradientBoosting), substantial information is present early, but additional interaction signal is captured between 60 and 90 days.
* For Random Forest, performance remains flat from 30 to 60 days before improving at 90 days.
* **Conclusion**: There is no universal "optimal" observation window; rather, linear trajectory signals emerge rapidly, while complex multi-month interaction dynamics benefit from a longer 90-day window.

---

## 7. Robustness Analyses & Sensitivity Testing

To evaluate model stability under various observational conditions, we conducted three distinct robustness/sensitivity analyses:
1. **Solvent-subcohort analysis**: evaluating predictive lift on borrowers whose account balance remained strictly non-negative throughout the observation period;
2. **Chronological out-of-time evaluation**: evaluating model robustness across calendar cohorts (trained on 1993–1995, tested on 1996–1997);
3. **Expanded-cohort sensitivity analysis**: evaluating whether behavioural lift persists across the broader loan book ($N=656$) at 90 days.

### 7.1 Solvent-Subcohort Analysis
To verify that Model B does not merely function as an indicator of accounts that have already entered unauthorized overdraft within the window, we evaluated performance on borrowers whose account balance remained strictly non-negative ($\min(\text{balance}) \ge 0$) throughout the observation period:
* **30-day window**: 2 defaults entered negative balance; **29 defaults remained solvent** ($N=232$).
* **60-day window**: 4 defaults entered negative balance; **27 defaults remained solvent** ($N=230$).
* **90-day window**: 7 defaults entered negative balance; **24 defaults remained solvent** ($N=227$).

#### Table 3: Performance on Strictly Solvent Borrowers

| Model Family | Window | Solvent Cohort (N / Def) | Solvent Model A PR-AUC | Solvent Model B PR-AUC | Mean $\Delta$ | Paired Win Rate | Empirical Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **30d** | 232 / 29 | 0.503 | **0.680** | **+0.177** | 92.0% | Positive lift without active overdraft |
| **Logistic Regression** | **60d** | 230 / 27 | 0.513 | **0.698** | **+0.184** | 92.0% | Positive lift without active overdraft |
| **Logistic Regression** | **90d** | 227 / 24 | 0.499 | **0.671** | **+0.172** | 84.0% | Positive lift without active overdraft |
| **HistGradientBoosting** | **30d** | 232 / 29 | 0.333 | **0.589** | **+0.255** | 96.0% | Positive lift without active overdraft |
| **HistGradientBoosting** | **60d** | 230 / 27 | 0.333 | **0.602** | **+0.268** | 96.0% | Positive lift without active overdraft |
| **HistGradientBoosting** | **90d** | 227 / 24 | 0.346 | **0.610** | **+0.265** | 96.0% | Positive lift without active overdraft |
| **Random Forest** | **30d** | 232 / 29 | 0.350 | **0.566** | **+0.216** | 88.0% | Positive lift without active overdraft |
| **Random Forest** | **60d** | 230 / 27 | 0.361 | **0.542** | **+0.182** | 80.0% | Positive lift without active overdraft |
| **Random Forest** | **90d** | 227 / 24 | 0.362 | **0.500** | **+0.138** | 76.0% | Positive lift without active overdraft |

> [!IMPORTANT]
> **Substantive Finding**: Model B retains large, positive predictive increments among borrowers who remain solvent throughout the window (Mean $\Delta \approx +0.14$ to $+0.27$, win rates $76\%\text{--}96\%$). This provides evidence that the incremental predictive signal is not solely attributable to borrowers who had already entered negative balance during the observation window.

### 7.2 Chronological Out-of-Time (OOT) Holdout
To evaluate temporal robustness across changing macroeconomic conditions, we implemented a chronological split:
* **Training Set**: Loans originated 1993–1995 ($N = 156$, 22 defaults = $14.10\%$).
* **Holdout Test Set**: Loans originated 1996–1997 ($N = 78$, 9 defaults = $11.54\%$).
* *Holdout Context*: All 9 test defaults maintained non-negative balances throughout their first 90 days post-disbursement.

#### Table 4: Chronological Out-of-Time Performance ($N_{\text{test}} = 78$, 9 Defaults)

| Model Family | Window | OOT Model A PR-AUC | OOT Model B PR-AUC | OOT $\Delta_{\text{PR-AUC}}$ | OOT Model A ROC-AUC | OOT Model B ROC-AUC | OOT $\Delta_{\text{ROC-AUC}}$ | OOT Model B Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **30d** | 0.345 | **0.594** | **+0.249** | 0.630 | **0.831** | **+0.201** | 0.136 (vs 0.177) |
| **Logistic Regression** | **60d** | 0.345 | **0.630** | **+0.286** | 0.630 | **0.895** | **+0.266** | 0.119 (vs 0.177) |
| **Logistic Regression** | **90d** | 0.345 | **0.569** | **+0.224** | 0.630 | **0.837** | **+0.208** | 0.119 (vs 0.177) |
| **HistGradientBoosting** | **30d** | 0.148 | **0.296** | **+0.147** | 0.576 | **0.812** | **+0.235** | 0.127 (vs 0.145) |
| **HistGradientBoosting** | **60d** | 0.148 | **0.365** | **+0.217** | 0.576 | **0.841** | **+0.264** | 0.118 (vs 0.145) |
| **HistGradientBoosting** | **90d** | 0.148 | **0.426** | **+0.278** | 0.576 | **0.731** | **+0.155** | 0.096 (vs 0.145) |
| **Random Forest** | **30d** | 0.161 | **0.203** | **+0.042** | 0.572 | **0.739** | **+0.167** | 0.124 (vs 0.136) |
| **Random Forest** | **60d** | 0.161 | **0.282** | **+0.122** | 0.572 | **0.813** | **+0.242** | 0.106 (vs 0.136) |
| **Random Forest** | **90d** | 0.161 | **0.284** | **+0.124** | 0.572 | **0.791** | **+0.219** | 0.109 (vs 0.136) |

> [!NOTE]
> **Methodological Framing of OOT Holdout**: The chronological test set contains only 9 defaults, so OOT point estimates have substantial sampling uncertainty and should be interpreted as **supporting evidence of temporal robustness** rather than definitive prospective performance estimates.

### 7.3 Secondary Sensitivity Analysis: Expanded Cohort ($N=656$) at 90 Days
To test whether the Model B improvement is sensitive to the completed-loan selection rule, we evaluated the models on the right-truncation-filtered expanded cohort ($N=656$, 581 good, 75 default; $11.43\%$ base rate) at 90 days across 25 CV splits.

#### Table 5: Sensitivity Analysis on Expanded Cohort ($N=656$) at 90 Days

| Model Family | Expanded Model A PR-AUC | Expanded Model B PR-AUC | Mean $\Delta$ | Paired Win Rate | Wilcoxon $p$-value | Expanded Model A ROC-AUC | Expanded Model B ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | $0.591 \pm 0.084$ | **$0.841 \pm 0.069$** | **+0.251** | **100.0%** (25/25) | $6.0 \times 10^{-8}$ | $0.805 \pm 0.054$ | **$0.937 \pm 0.030$** |
| **Logistic Regression** | $0.577 \pm 0.091$ | **$0.785 \pm 0.092$** | **+0.208** | **100.0%** (25/25) | $6.0 \times 10^{-8}$ | $0.800 \pm 0.059$ | **$0.908 \pm 0.060$** |
| **Random Forest** | $0.538 \pm 0.104$ | **$0.756 \pm 0.084$** | **+0.218** | **96.0%** (24/25) | $1.2 \times 10^{-7}$ | $0.805 \pm 0.060$ | **$0.928 \pm 0.027$** |

> [!NOTE]
> **Methodological Role of Expanded Cohort**: The results provide **secondary sensitivity evidence that the incremental behavioural signal is not solely an artifact of selecting completed loans**. However, because Status `C` contracts remain right-censored ("OK so far" as of December 1998), this analysis remains strictly secondary and must not be described as proof of external or population-level generalization.

### 7.4 Illustrative Risk-Tranche Ranking Yields
Table 6 illustrates out-of-fold risk concentration for Logistic Regression Model B at 60 days.

#### Table 6: Out-of-Fold Risk Review Tranche Yields (LogReg 60d, Average per Test Fold)

| Risk Tranche | Review Volume / Fold | True Defaults Captured / Fold | Precision (PPV) | Recall (Sensitivity) | Enrichment over Base Rate ($13.25\%$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 2%** | 1.0 loan | 0.96 default | **96.0%** | 15.6% | **7.2x** |
| **Top 5%** | 2.0 loans | 1.84 defaults | **92.0%** | 29.9% | **6.9x** |
| **Top 10%** | 5.0 loans | 3.84 defaults | **76.8%** | **62.1%** | **5.8x** |
| **Top 15%** | 7.0 loans | 4.36 defaults | 62.3% | 70.6% | 4.7x |
| **Top 20%** | 9.0 loans | 4.80 defaults | 53.3% | 77.6% | 4.0x |

> [!WARNING]
> **Illustrative Caveat on Denominators**: In a cross-validation fold of 46–47 loans, the top 2% tranche corresponds to exactly **1 loan**, and the top 5% tranche corresponds to **2 loans**. These figures illustrate ranking concentration rather than operational policy guarantees or deployment thresholds.

---

## 8. Feature Interpretation (Strictly Observational & Non-Causal)

Table 7 displays standardized coefficients from Logistic Regression Model B fitted on the 60-day window.

#### Table 7: Standardized Feature Coefficients (Logistic Regression 60d)

| Standardized Feature | Coefficient | Sign | Observational Statistical Association |
| :--- | :---: | :---: | :--- |
| `feat_w60_cash_withdrawal_ratio` | **+1.927** | Positive | Higher proportion of debits conducted via cash is associated with higher predicted default risk. |
| `feat_account_age_days` | **-1.620** | Negative | Longer pre-loan banking relationship tenure is associated with lower predicted default risk. |
| `feat_pre_loan_tx_count` | **+1.198** | Positive | Higher pre-loan transaction volume is conditionally associated with higher predicted risk after controlling for tenure. |
| `feat_w60_trans_count` | **-1.196** | Negative | Higher post-disbursement transaction frequency is associated with lower predicted default risk. |
| `feat_had_pre_loan_overdraft` | **+1.147** | Positive | History of unauthorized overdraft prior to loan is associated with higher predicted default risk. |
| `feat_w60_balance_min` | **-1.023** | Negative | Higher minimum balance within the window is associated with lower predicted default risk. |
| `feat_district_entrepreneurs` | **-0.804** | Negative | Originating in districts with higher entrepreneurial proportion is associated with lower predicted default risk. |
| `feat_w60_had_negative_balance` | **+0.694** | Positive | Experiencing a negative balance within the window is associated with higher predicted default risk. |
| `feat_monthly_payment` | **+0.538** | Positive | Higher scheduled monthly installment is associated with higher predicted default risk. |

> [!CAUTION]
> **Strict Non-Causal Disclaimer**: These coefficients reflect observational statistical associations within a regularized linear model. They must not be interpreted as causal levers or behavioral policies (e.g., restricting cash withdrawals will not causally reduce default rates). They represent observed empirical markers of distress within this historical banking sample.

---

## 9. Claims Boundary & Evidence Matrix

To maintain scientific integrity in academic presentations and technical interviews, claims are partitioned into safe portfolio statements versus technical report nuances:

| Analytical Topic | Safe for CV / Portfolio Presentation | Technical-Report-Only Nuance (Do Not Oversell) |
| :--- | :--- | :--- |
| **Model A vs B Incremental Lift** | **Approved**: Achieved $+0.18$ to $+0.36$ PR-AUC improvement from adding post-disbursement behavioural features across identical 25-fold paired CV splits, with $92\%\text{--}100\%$ paired win rates. | Report details: Specific fold-by-fold standard deviations, fold sample sizes, and Wilcoxon signed-rank test dependence caveats. |
| **Information Arrival Progression** | **Approved**: Demonstrated model-dependent information arrival across 30–90-day behavioural windows, with substantial signal available by 60 days for Logistic Regression and HistGradientBoosting and additional 90-day gains for non-linear models. | Report details: Avoid claiming a universal "optimal" observation window. Detail the exact contrast between linear plateaus and gradient-boosted tree interaction gains. |
| **Solvent-Subcohort Analysis** | **Approved**: Showed that incremental predictive signal persists among borrowers remaining non-negative throughout the observation window. | Report details: Provide exact counts of defaults crossing zero (2 at 30d, 4 at 60d, 7 at 90d). Avoid claiming this "proves" distress absence. |
| **Leakage & De-Biasing Audit** | **Approved**: Identified and audited major temporal and target-leakage shortcuts, including full-lifetime balance separation and calendar-duration censoring. | Report details: Full tabular breakdown demonstrating 100% target leakage separation and calendar duration censoring mathematics. |
| **Chronological Out-of-Time Holdout** | **Approved**: Evaluated temporal robustness using a chronological out-of-time holdout (1996–1997). | **Do not claim standalone high precision**: Avoid quoting "PR-AUC 0.630 on OOT" as a definitive prospective metric given only 9 test defaults. |
| **Expanded Cohort Sensitivity** | **Approved**: Observed consistent incremental behavioural signal in a secondary 656-loan sensitivity cohort. | Report details: Clearly label as secondary sensitivity analysis; explain that Status `C` contracts remain right-censored. |
| **Risk-Tranche Review Yields** | **Do Not Include on CV**: Do not quote "96% precision in top 2% tranche" on resumes (corresponds to literally 1 loan per fold). | Report details: Present purely as an illustrative ranking exercise, explicitly noting small fold denominators (46–47 loans per fold). |
| **Feature Interpretability** | **Approved**: Identified empirical associations between default risk and cash withdrawal intensity, balance trajectory, and relationship tenure. | Report details: Strict non-causal framing. Avoid terms like "protective factor", "causes default", or "cash burn velocity". |

---

## 10. Threats to Validity & Methodological Limitations

1. **Small Positive Sample Size**:
   * The primary completed cohort contains only 31 default events ($N=234$). While repeated 5-fold cross-validation mitigates fold-splitting variance across 25 evaluation units, individual point estimates retain inherent sampling uncertainty.
2. **Historical Single-Institution Context**:
   * The data captures operations from a single Czech commercial bank between 1993 and 1998 during post-communist economic transition. Transaction dynamics may not transfer directly to modern digital lending, credit card portfolios, or mobile payment platforms.
3. **Completed-Loan Cohort Selection**:
   * Restricting the headline cohort to matured loans eliminates right-censoring but overrepresents shorter-duration contracts (12–36 months) and contracts disbursed early in the observation period (1993–1995).
4. **Right-Censoring in Status `C` Loans**:
   * In the expanded cohort ($N=656$), Status `C` contracts ("OK so far" as of December 1998) may have experienced subsequent defaults after the study cutoff. This cohort functions strictly as a sensitivity check rather than an alternative ground truth.
5. **Sampling Fragility in OOT Holdout**:
   * The chronological holdout (1996–1997) contains only 9 defaults ($N=78$). The consistent superiority of Model B across models and windows provides supporting evidence of temporal robustness, but point estimates have wide sampling intervals.
6. **Observational & Non-Causal Feature Interpretations**:
   * Model coefficients reflect observational statistical associations within regularized models. They represent distress symptoms rather than causal mechanisms or actionable intervention policies.
7. **Absence of External Institutional Validation**:
   * All evaluations represent internal validation within a single banking dataset without testing on modern external credit portfolios.

---

## 11. Final Research Conclusions

1. **Incremental Predictive Value (Strongest Finding)**:
   * Post-disbursement transaction behaviour observed within early post-disbursement windows provides substantial and consistent incremental predictive information beyond static loan origination characteristics and pre-loan banking history (PR-AUC gains of $+0.18$ to $+0.36$; paired win rates $92\%\text{--}100\%$).
2. **Model-Dependent Information Arrival (Second Finding)**:
   * Useful predictive information emerges early, but information accumulation is model-dependent: linear models capture the vast majority of useful signal within 60 days, whereas gradient-boosted trees capture additional non-linear interaction gains through 90 days. There is no single universal optimal monitoring window.
3. **Sub-Overdraft Predictive Robustness (Third Finding)**:
   * The incremental behavioral lift persists among borrowers who remain strictly solvent throughout the observation window, providing evidence that the signal is not solely attributable to an active unauthorized overdraft.
4. **Methodological Contribution**:
   * The study demonstrates that rigorous credit risk modeling on observational banking data requires strict temporal boundaries, transparent de-biasing of target leakage, careful handling of calendar censoring, and paired cross-validation across identical partitions.
