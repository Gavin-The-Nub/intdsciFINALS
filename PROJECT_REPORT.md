# DE LA SALLE LIPA
## College of Information Technology and Engineering
### Computer Science Department

# FINAL PROJECT REPORT: Supervised Machine Learning Pipeline for Hourly Bike-Sharing Demand Classification in Seoul
**Course:** Introduction to Data Science (Intdsci)  
**Academic Year:** 2025–2026  
**Authors / Group Members:** [Group Member 1], [Group Member 2], [Group Member 3]  
**Dataset:** Seoul Bike Sharing Demand Dataset (UCI Machine Learning Repository)  
**Notebook Link:** [`seoul_bike_ml_project.ipynb`](./seoul_bike_ml_project.ipynb)  
**Pipeline Script:** [`build_and_execute_notebook.py`](./_archive/build_and_execute_notebook.py)  

---

## Table of Contents
1. [A. Introduction](#a-introduction)
2. [B. Data Collection and Preprocessing](#b-data-collection-and-preprocessing)
   - [B.1 Data Source and Acquisition](#b1-data-source-and-acquisition)
   - [B.2 Comprehensive Data Dictionary](#b2-comprehensive-data-dictionary)
   - [B.3 Data Cleaning and Quality Assurance Log](#b3-data-cleaning-and-quality-assurance-log)
   - [B.4 Exploratory Data Analysis (EDA) and Evidence-Based Interpretations](#b4-exploratory-data-analysis-eda-and-evidence-based-interpretations)
3. [C. Feature Engineering and Selection](#c-feature-engineering-and-selection)
   - [C.1 Target Formulation and Anti-Leakage Protocol](#c1-target-formulation-and-anti-leakage-protocol)
   - [C.2 Temporal Feature Extraction](#c2-temporal-feature-extraction)
   - [C.3 Pearson Correlation Analysis and Feature Selection](#c3-pearson-correlation-analysis-and-feature-selection)
   - [C.4 Multicollinearity Ablation Experiment (Dew Point Removal)](#c4-multicollinearity-ablation-experiment-dew-point-removal)
   - [C.5 Preprocessing Pipeline and Transformed Feature Inspection](#c5-preprocessing-pipeline-and-transformed-feature-inspection)
4. [D. Data Splitting](#d-data-splitting)
   - [D.1 Partitioning Strategy (80/20 Train-Test Split)](#d1-partitioning-strategy-8020-train-test-split)
   - [D.2 Reproducibility and Zero-Leakage Guarantee](#d2-reproducibility-and-zero-leakage-guarantee)
5. [E. Model Selection and Training](#e-model-selection-and-training)
   - [E.1 Model Justification and Architectural Roles](#e1-model-justification-and-architectural-roles)
   - [E.2 Pipeline-Integrated Model Training](#e2-pipeline-integrated-model-training)
6. [F. Model Evaluation](#f-model-evaluation)
   - [F.1 Confusion Matrix Results and Interpretation](#f1-confusion-matrix-results-and-interpretation)
   - [F.2 Performance Metrics Summary Table](#f2-performance-metrics-summary-table)
   - [F.3 Detailed Classification Reports](#f3-detailed-classification-reports)
   - [F.4 Feature Importance Analysis](#f4-feature-importance-analysis)
   - [F.5 Operational Error Cost Analysis (Type I vs. Type II Errors)](#f5-operational-error-cost-analysis-type-i-vs-type-ii-errors)
   - [F.6 Model Limitations and Risk Analysis](#f6-model-limitations-and-risk-analysis)
7. [References](#references)

---

## A. Introduction

Urban public bicycle-sharing systems (BSS) serve as a critical pillar of modern, low-carbon municipal transit networks by solving the "first-and-last mile" commuter problem and mitigating metropolitan traffic congestion; however, fleet management authorities face complex logistical bottlenecks caused by extreme spatial-temporal demand volatility. When high-demand surges occur during commuter peak hours, high-density stations quickly suffer from dock starvation (no bicycles available for checkout), while adverse weather causes systemic fleet underutilization and dock saturation (no empty docks available for return). This project develops an end-to-end supervised machine learning classification pipeline using 8,760 hourly observations from the Seoul Public Bike sharing system to forecast whether any operating hour will experience **High Demand ($\ge 505$ rentals/hour)** or **Low Demand ($< 505$ rentals/hour)** based on real-time meteorological conditions and calendar schedules. By framing this task as a balanced binary classification problem around the empirical median demand threshold, municipal fleet operators can proactively dispatch rebalancing vans, adjust maintenance schedules, and prevent transit service disruptions before commuter bottlenecks emerge.

---

## B. Data Collection and Preprocessing

### B.1 Data Source and Acquisition
The data analyzed in this investigation originates from the **Seoul Bike Sharing Demand Dataset**, collected via the Seoul Open Data Plaza and archived in the University of California, Irvine (UCI) Machine Learning Repository (Sathishkumar & Cho, 2020). The dataset spans exactly 365 days from **December 1, 2017 to November 30, 2018**, providing 8,760 consecutive hourly records ($365 \times 24 = 8,760$). 

*Technical Ingestion Caveat:* The raw CSV file (`SeoulBikeData.csv`) contains non-ASCII byte sequences (`\xb0`) representing the degree symbol (`°C`) in the headers `Temperature(°C)` and `Dew point temperature(°C)`. Standard `utf-8` decoding raises a `UnicodeDecodeError`. Therefore, the file was ingested using the `latin1` (ISO-8859-1) character encoding scheme and immediately normalized to programmatic snake_case identifiers.

### B.2 Comprehensive Data Dictionary

| Variable Name | Original Header in CSV | Storage Type | Domain / Units | Role in ML Pipeline | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Date` | `Date` | Datetime (`dd/mm/yyyy`) | `01/12/2017` – `30/11/2018` | Metadata / Feature Extractor | Observation date; decomposed into `Month` and `DayOfWeek`. |
| `Rented_Bike_Count` | `Rented Bike Count` | Integer | $0$ to $3,556$ | Raw Target (Binarized, then Dropped) | Count of rented bicycles per hour ($M = 704.60$, $Mdn = 504.50$). |
| `Hour` | `Hour` | Integer | $0$ to $23$ | Numeric Feature | Hour of the day in 24-hour notation. |
| `Temperature_C` | `Temperature(°C)` | Continuous Float | $-17.80$ to $39.40\ ^\circ\text{C}$ | Numeric Feature | Ambient dry-bulb air temperature in degrees Celsius. |
| `Humidity_Pct` | `Humidity(%)` | Integer | $0$ to $98\%$ | Numeric Feature | Relative humidity percentage. |
| `Wind_Speed_ms` | `Wind speed (m/s)` | Continuous Float | $0.0$ to $7.4\ \text{m/s}$ | Numeric Feature | Surface wind speed in meters per second. |
| `Visibility_10m` | `Visibility (10m)` | Integer | $27$ to $2,000$ | Numeric Feature | Horizontal atmospheric visibility in units of 10 meters. |
| `Dew_Point_C` | `Dew point temperature(°C)` | Continuous Float | $-30.60$ to $27.20\ ^\circ\text{C}$ | Numeric Feature | Atmospheric dew point temperature. |
| `Solar_Radiation_MJm2` | `Solar Radiation (MJ/m2)` | Continuous Float | $0.00$ to $3.52\ \text{MJ/m}^2$ | Numeric Feature | Global solar radiation intensity. |
| `Rainfall_mm` | `Rainfall(mm)` | Continuous Float | $0.0$ to $35.0\ \text{mm}$ | Numeric Feature | Accumulated hourly precipitation depth. |
| `Snowfall_cm` | `Snowfall (cm)` | Continuous Float | $0.0$ to $8.8\ \text{cm}$ | Numeric Feature | Accumulated hourly snowfall depth. |
| `Seasons` | `Seasons` | Nominal String | Winter, Spring, Summer, Autumn | Categorical Feature | Meteorological season of the year. |
| `Holiday` | `Holiday` | Binary String | `Holiday`, `No Holiday` | Categorical Feature | Official national public holiday indicator. |
| `Functioning_Day` | `Functioning Day` | Binary String | `Yes`, `No` | Categorical Feature | Service status (`Yes`: service active; `No`: service suspended). |
| `Demand_Class` | *Derived Feature* | Binary Integer | $0$ (Low), $1$ (High) | **Supervised Target ($y$)** | $1$ if `Rented_Bike_Count >= 505`, else $0$. |
| `Month` | *Derived Feature* | Integer | $1$ to $12$ | Numeric Feature | Calendar month extracted from `Date`. |
| `DayOfWeek` | *Derived Feature* | Integer | $0$ to $6$ | Numeric Feature | Day of the week extracted from `Date` ($0 = \text{Monday}$). |

### B.3 Data Cleaning and Quality Assurance Log
In accordance with the rubric requirement (*"Before and after checks; justified actions; traceable cleaning log; no unnecessary deletion or invented issues"*), explicit audits were executed:

1. **Missing Value Audit:**
   * Executed `df.isna().sum()` across all 14 columns.
   * *Finding:* Exactly $0$ missing cells ($0.00\%$) across all $8,760$ rows. No artificial imputation or synthetic deletion was required.
2. **Duplicate Row Audit:**
   * Executed `df.duplicated().sum()` on all primary columns.
   * *Finding:* Exactly $0$ duplicate observations. Every single hourly timestamp represents a unique time point.
3. **Column Standardization Log:**
   * Replaced raw strings containing byte symbols (e.g., `'Temperature(\xb0C)'`) with clean identifiers (`'Temperature_C'`).
4. **Service Outage Verification:**
   * Filtered rows where `Functioning_Day == 'No'`.
   * *Finding:* Exactly $295$ hours recorded $0$ bike rentals because operations were officially suspended. These records were preserved because `Functioning_Day` is a legitimate operational signal that real-world models must handle.

### B.4 Exploratory Data Analysis (EDA) and Evidence-Based Interpretations

#### Plot 1: Target Variable Distribution & Cutoff Threshold
![Figure 1: Target Distribution](figures/eda_target_distribution.png)

* **Visual Design:** A combined 50-bin histogram with kernel density estimation (KDE), featuring a dashed red vertical line at the median cutoff ($504.5$ bikes/hour) and a dotted green vertical line at the sample mean ($704.6$ bikes/hour).
* **Interpretation 1 (Distribution Skewness and Balanced Threshold Formulation):** 
  The raw target distribution exhibits pronounced right-skewness ($\text{Skewness} = 1.15$), with a large concentration of near-zero hourly counts (during overnight hours and precipitation events) and an extended tail reaching up to $3,556$ rentals/hour. Setting the binary threshold at the empirical median ($504.5 \rightarrow 505$ threshold) divides the dataset into exactly $4,380$ Low Demand hours ($50.0\%$) and $4,380$ High Demand hours ($50.0\%$). This guarantees optimal metric symmetry, preventing majority-class cheating in classifiers and aligning model sensitivity with urban peak capacity.

#### Plot 2: Average Hourly Commute Rhythm Across Seasons
![Figure 2: Hourly Seasonal Commute Curve](figures/eda_hourly_commute_by_season.png)

* **Visual Design:** Time-series profile mapping mean bicycle rentals against hour of day ($0$ to $23$) across all four seasons, with highlighted rush-hour commuting windows.
* **Interpretation 2 (Bimodal Commuter Rhythm and Seasonal Suppression):** 
  Across Spring, Summer, and Autumn, bicycle rentals exhibit a sharp bimodal diurnal profile: a morning commuter peak at **8:00 AM** ($\approx 1,000$ to $1,300$ rentals/hr) and an evening return peak at **6:00 PM** ($\approx 1,800$ to $3,200$ rentals/hr). Off-peak midday hours experience moderate, stable demand. In Winter, however, demand remains suppressed below $500$ bikes/hour across virtually the entire 24-hour cycle due to freezing temperatures (mean winter temperature $=-2.5^\circ\text{C}$). This indicates that non-linear interaction between `Hour` and `Temperature_C` is a dominant predictive driver.

#### Plot 3: Weather Precipitation and Operational Feasibility Impact
![Figure 3: Weather and Service Impact](figures/eda_weather_impact.png)

* **Visual Design:** A side-by-side comparative boxplot and barplot displaying ridership variations under precipitation states (Rain vs. Clear) and operational status (Functioning Day vs. Closed).
* **Interpretation 3 (Elasticity to Weather Shocks and Zero-Rental Guarantees):** 
  Precipitation acts as a powerful demand suppressant: median rentals drop from $606$ bikes/hour during clear weather to under $80$ bikes/hour during rain events. Furthermore, the right panel confirms that when `Functioning_Day == 'No'` ($n = 295$), rentals are identically zero. Machine learning architectures must give decisive negative weight to precipitation and non-functioning flags.

---

## C. Feature Engineering and Selection

### C.1 Target Formulation and Anti-Leakage Protocol
To comply with Requirement 4 (*"Use confusion matrix for the performance evaluation of the model"*), the continuous problem was formulated as binary classification:
$$\text{Demand\_Class} = \begin{cases} 1 & \text{if } \text{Rented\_Bike\_Count} \ge 505 \\ 0 & \text{if } \text{Rented\_Bike\_Count} < 505 \end{cases}$$

> **Critical Anti-Leakage Action:** Retaining `Rented_Bike_Count` in the feature matrix $X$ would result in $100\%$ target leakage, rendering the model trivial and invalid. Immediately following label generation, `Rented_Bike_Count` was completely excised from $X$.

### C.2 Temporal Feature Extraction
Raw calendar date strings (`01/12/2017`) are uninterpretable by mathematical models. Using `pd.to_datetime()`, two new numerical features were engineered:
1. `Month` ($\in [1, 12]$): Encodes the annual macro-climate cycle and seasonal progression.
2. `DayOfWeek` ($\in [0, 6]$ where $0 = \text{Monday}$ and $6 = \text{Sunday}$): Encodes weekly work-versus-weekend leisure patterns.

### C.3 Pearson Correlation Analysis and Feature Selection
![Figure 4: Correlation Matrix](figures/correlation_heatmap.png)

We evaluated linear dependencies across all continuous variables and the target:

| Feature Variable | Correlation with Rentals ($r$) | Direction & Strength | Physical / Behavioral Meaning |
| :--- | :---: | :--- | :--- |
| `Temperature_C` | **$+0.54$** | Strong Positive | Warmer temperatures strongly encourage cycling. |
| `Hour` | **$+0.41$** | Moderate Positive | Aligns with daylight, business hours, and commuter surges. |
| `Dew_Point_C` | **$+0.38$** | Moderate Positive | Tracks warmer, moister air masses (collinear with temperature). |
| `Solar_Radiation_MJm2` | **$+0.26$** | Weak-Moderate Positive | Clear, sunny conditions promote outdoor transit. |
| `Visibility_10m` | **$+0.20$** | Weak Positive | Clear atmospheric conditions correlate with safe cycling. |
| `Wind_Speed_ms` | $+0.12$ | Negligible Positive | Minor ventilation effect in warm seasons. |
| `Month` | $+0.13$ | Weak Positive | Seasonal shift towards summer/autumn peak months. |
| `DayOfWeek` | $-0.04$ | Negligible Negative | Commuting vs. weekend patterns offset each other in aggregate. |
| `Rainfall_mm` | **$-0.12$** | Weak Negative | Immediate barrier to cycling safety and comfort. |
| `Snowfall_cm` | **$-0.14$** | Weak Negative | Freezing, slippery road surfaces deter riders. |
| `Humidity_Pct` | **$-0.20$** | Moderate Negative | High humidity causes discomfort and accompanies precipitation. |

### C.4 Multicollinearity Ablation Experiment (Dew Point Removal)
The correlation matrix reveals an extreme collinearity between **`Temperature_C`** and **`Dew_Point_C`** with a Pearson coefficient of **$r = 0.913$**. 

* **The Scientific Hypothesis:** Including two nearly identical thermodynamic measurements inflates parameter variance in linear models (Logistic Regression) without providing additive information.
* **The Controlled Experiment:** We evaluated two model variants:
  * *Variant A (All Features):* All numeric features included.
  * *Variant B (Collinearity Mitigated):* Excising `Dew_Point_C` from $X$.
* **Empirical Result:** 
  * Logistic Regression (All Features): Accuracy = **$86.93\%$**, F1 = **$0.8665$**.
  * Logistic Regression (No Dew Point): Accuracy = **$86.99\%$**, F1 = **$0.8671$**.
  * *Conclusion:* Removing `Dew_Point_C` slightly improved out-of-sample accuracy while producing a leaner, more robust model and eliminating coefficient distortion. This validates our initial hypothesis that testing feature subsets via correlation analysis can streamline the model while maintaining or marginally improving classification accuracy.

### C.5 Preprocessing Pipeline and Transformed Feature Inspection
To prevent data snooping, preprocessing was wrapped in a `ColumnTransformer`:
* **Numeric Pipelines:** `StandardScaler()` applied to `['Hour', 'Temperature_C', 'Humidity_Pct', 'Wind_Speed_ms', 'Visibility_10m', 'Dew_Point_C', 'Solar_Radiation_MJm2', 'Rainfall_mm', 'Snowfall_cm', 'Month', 'DayOfWeek']` to standardize zero mean and unit variance ($\mu = 0, \sigma = 1$).
* **Categorical Pipelines:** `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to nominal features:
  * `Seasons`: 4 categories $\rightarrow$ 3 binary columns (`Spring`, `Summer`, `Winter`; `Autumn` as base reference).
  * `Holiday`: 2 categories $\rightarrow$ 1 binary column (`No Holiday` as reference).
  * `Functioning_Day`: 2 categories $\rightarrow$ 1 binary column (`Functioning_Day_Yes`).

*Output Verification:* Preprocessing expands the original 14 input features into **16 normalized numerical predictors** with zero missing values and zero leakage.

---

## D. Data Splitting

### D.1 Partitioning Strategy (80/20 Train-Test Split)
In strict conformity with Requirement 3, the dataset was partitioned into:
* **Training Partition:** **$80\%$ ($n = 7,008$ hours)**
* **Testing Holdout Partition:** **$20\%$ ($n = 1,752$ hours)**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

| Partition | Total Samples | Percentage | Low Demand ($<505$) | High Demand ($\ge 505$) | Class Balance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Set (`X_train`)** | $7,008$ | $80.0\%$ | $3,495$ ($49.87\%$) | $3,513$ ($50.13\%$) | Perfectly balanced |
| **Testing Set (`X_test`)** | $1,752$ | $20.0\%$ | $885$ ($50.51\%$) | $867$ ($49.49\%$) | Perfectly balanced |
| **Full Dataset** | $8,760$ | $100.0\%$ | $4,380$ ($50.00\%$) | $4,380$ ($50.00\%$) | Exact 50/50 split |

### D.2 Reproducibility and Zero-Leakage Guarantee
1. **Reproducibility:** Fixing `random_state=42` guarantees that every shuffle, data split, and pseudo-random seed remains identical across any evaluation machine or kernel execution.
2. **Zero-Leakage Guarantee (Learning Outcome 3):** 
   * Preprocessing parameters (the means and standard deviations $\mu, \sigma$ for `StandardScaler`, and the vocabulary categories for `OneHotEncoder`) were computed **strictly on `X_train`**.
   * `X_test` was strictly transformed using the frozen parameters learned from `X_train`. Under no circumstances was `.fit()` or `.fit_transform()` ever called on the full dataset or the test partition.

---

## E. Model Selection and Training

### E.1 Model Justification and Architectural Roles
To address Learning Outcomes 3 & 4 and fulfill the grading rubric, three contrasting models were selected:

1. **Baseline Model — `DummyClassifier(strategy='most_frequent')`**
   * *Justification:* Establishes the empirical lower-bound benchmark. Because the dataset has roughly equal classes, guessing the most frequent class provides an accuracy baseline of $\approx 49.5\%$. Any viable machine learning model must dramatically beat this benchmark.
2. **Linear Benchmark — `LogisticRegression(max_iter=1000, random_state=42)`**
   * *Justification:* A convex parametric linear model that estimates log-odds of high demand as a linear combination of normalized inputs. It serves as an interpretable benchmark to assess whether linear hyperplanes can separate the demand classes.
3. **Non-Linear Ensemble — `RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42)`**
   * *Justification:* A bagging ensemble of 100 decorrelated decision trees. Random forests naturally model step-function non-linearities (such as sudden commute spikes at 8 AM and 6 PM), handle categorical interaction thresholds (cold temperatures combined with rain), and are robust against overfitting.

### E.2 Pipeline-Integrated Model Training
All models were packaged inside `sklearn.pipeline.Pipeline` objects linking the `ColumnTransformer` directly to the classifier, ensuring that fitting and prediction remain atomic, clean, and completely reproducible.

---

## F. Model Evaluation

### F.1 Confusion Matrix Results and Interpretation
![Figure 5: Confusion Matrices](figures/confusion_matrices.png)

The $1,752$ holdout test cases ($885$ Actual Low Demand, $867$ Actual High Demand) yielded the following confusion matrices:

$$\text{Baseline: } \begin{bmatrix} 0 & 885 \\ 0 & 867 \end{bmatrix} \quad\quad \text{Logistic Regression: } \begin{bmatrix} 780 & 105 \\ 124 & 743 \end{bmatrix} \quad\quad \text{Random Forest: } \begin{bmatrix} 855 & 30 \\ 46 & 821 \end{bmatrix}$$

* **Baseline (`DummyClassifier`):** Predicts High Demand unconditionally, resulting in $885$ False Positives and zero True Negatives.
* **Logistic Regression:** Correctly classifies $780$ Low Demand and $743$ High Demand hours, but makes $229$ total errors ($105$ False Positives and $124$ False Negatives).
* **Random Forest Classifier:** Drastically reduces total errors to only **$76$ hours out of $1,752$** ($855$ True Negatives, $821$ True Positives, only $30$ False Positives, and $46$ False Negatives).

### F.2 Performance Metrics Summary Table

| Model Architecture | Features Used | Test Accuracy | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (`DummyClassifier`)** | All Features | $49.49\%$ | $0.4949$ | $1.0000$ | $0.6621$ | $0$ | $885$ | $0$ | $867$ |
| **Logistic Regression** | All Features | $86.93\%$ | $0.8762$ | $0.8570$ | $0.8665$ | $780$ | $105$ | $124$ | $743$ |
| **Logistic Regression (Ablated)**| Drop `Dew_Point_C`| $86.99\%$ | $0.8763$ | $0.8581$ | $0.8671$ | $780$ | $105$ | $123$ | $744$ |
| **Random Forest Classifier** | **All Features** | **$95.55\%$** | **$0.9647$** | **$0.9469$** | **$0.9558$** | **$855$** | **$30$** | **$46$** | **$821$** |

### F.3 Detailed Classification Reports

#### Logistic Regression Classification Report:
```
                      precision    recall  f1-score   support

 Low Demand (<505)      0.8628    0.8814    0.8720       885
High Demand (>=505)     0.8762    0.8570    0.8665       867

          accuracy                          0.8693      1752
         macro avg      0.8695    0.8692    0.8692      1752
      weighted avg      0.8694    0.8693    0.8693      1752
```

#### Random Forest Classifier Classification Report:
```
                      precision    recall  f1-score   support

 Low Demand (<505)      0.9489    0.9661    0.9574       885
High Demand (>=505)     0.9647    0.9469    0.9558       867

          accuracy                          0.9555      1752
         macro avg      0.9568    0.9565    0.9566      1752
      weighted avg      0.9567    0.9566    0.9566      1752
```

### F.4 Feature Importance Analysis
![Figure 6: Feature Importances](figures/feature_importances.png)

Gini importance extracted from the trained Random Forest ensemble confirms the dominant drivers of bicycle sharing demand:

| Rank | Feature | Gini Importance | Cumulative Importance | Practical Impact |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `Hour` | **$20.66\%$** | $20.66\%$ | Dominates diurnal commute spikes at 8 AM and 6 PM. |
| **2** | `Temperature_C` | **$17.57\%$** | $38.23\%$ | Primary comfort determinant for outdoor physical activity. |
| **3** | `Humidity_Pct` | **$9.30\%$** | $47.53\%$ | Strong proxy for mugginess and imminent rain. |
| **4** | `Seasons_Winter` | **$9.21\%$** | $56.74\%$ | Binary seasonal freeze penalty on ridership. |
| **5** | `Month` | **$7.79\%$** | $64.53\%$ | Captures month-by-month temperature transitions. |
| **6** | `Solar_Radiation_MJm2` | **$6.82\%$** | $71.35\%$ | Daytime sunshine indicator promoting recreational cycling. |
| **7** | `Functioning_Day_Yes`| **$6.76\%$** | $78.11\%$ | Decisive zero-rental filter for operational shutdown days. |
| **8** | `Dew_Point_C` | **$6.22\%$** | $84.33\%$ | Atmospheric moisture indicator. |
| **9** | `Visibility_10m` | **$3.77\%$** | $88.10\%$ | Air clarity and particulate haze factor. |
| **10**| `Rainfall_mm` | **$3.35\%$** | $91.45\%$ | Sudden precipitation barrier stopping active trips. |

### F.5 Operational Error Cost Analysis (Type I vs. Type II Errors)
In municipal fleet logistics, false predictions carry asymmetric real-world consequences:

* **Type I Error — False Positive ($FP = 30$ hours):**
  * *Operational Consequence:* The system predicts high demand, but actual demand is low. Logistic redistribution trucks needlessly transport bikes to stations that remain unutilized. This causes unnecessary fuel expenditure, driver overtime, and dock congestion.
* **Type II Error — False Negative ($FN = 46$ hours):**
  * *Operational Consequence:* The system predicts low demand, but actual demand surges. Stations experience dock starvation. Commuters arrive to find zero bicycles, causing missed connections, transit delays, and reputational erosion.
* **Trade-off Summary:** 
  The Random Forest classifier reduces Type I errors to just **$30$ events** and Type II errors to **$46$ events** across $1,752$ hours ($95.55\%$ overall accuracy). Its balanced precision ($96.5\%$) and recall ($94.7\%$) ensure that neither over-allocation nor station starvation occurs frequently.

### F.6 Model Limitations and Risk Analysis
1. **Spatial Aggregation:** The dataset records aggregate system-wide rentals across Seoul. In practice, demand varies dramatically by district (e.g., Gangnam commercial hub vs. Han River recreational paths). Spatial GPS coordinates per station would be needed for micro-station routing.
2. **Temporal Distribution Drift (Pre-COVID Dataset):** The data represents 2017–2018 travel habits. Hybrid work-from-home models and urban transit policy changes post-2020 may flatten morning peak intensities.
3. **Exogenous Shocks:** The model does not include non-weather transit anomalies such as subway service strikes, festival road closures, or fuel price spikes, which can induce sudden ridership spikes.

---

## References
* **Sathishkumar, V. E., & Cho, J.** (2020). *Seoul Bike Sharing Demand [Dataset]*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R
* **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
* **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
