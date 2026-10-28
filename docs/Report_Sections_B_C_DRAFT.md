# Report Draft: Sections B and C (Art)

Every number below comes from a fresh Restart & Run All of `seoul_bike_ml_project.ipynb` (8,760 rows, `random_state=42`). Notebook references are 0-based cell indices in the notebook as submitted; update them if Ian adds or moves cells.

---

## B. Data Collection and Preprocessing

### B.1 Data source

The data is the Seoul Bike Sharing Demand dataset, collected from the Seoul Open Data Plaza and archived in the UCI Machine Learning Repository (Sathishkumar & Cho, 2020). It has **8,760 hourly records** (365 days x 24 hours) from **1 December 2017 to 30 November 2018**, with **14 columns**. Each row is one hour of the Seoul public bike-sharing system, with the number of bikes rented and the weather and calendar conditions for that hour.

The CSV contains the degree symbol (°) in two headers, so it is read with `encoding="latin1"` (Notebook cell 3).

### B.2 Data dictionary

| Variable | Original header | Type | Range or values | Description |
| :-- | :-- | :-- | :-- | :-- |
| `Date` | Date | datetime | 01/12/2017 to 30/11/2018 | Day of observation (DD/MM/YYYY) |
| `Rented_Bike_Count` | Rented Bike Count | integer | 0 to 3,556 | Bikes rented in that hour (raw target) |
| `Hour` | Hour | integer | 0 to 23 | Hour of day |
| `Temperature_C` | Temperature(°C) | float | -17.8 to 39.4 | Air temperature |
| `Humidity_Pct` | Humidity(%) | integer | 0 to 98 | Relative humidity |
| `Wind_Speed_ms` | Wind speed (m/s) | float | 0.0 to 7.4 | Wind speed |
| `Visibility_10m` | Visibility (10m) | integer | 27 to 2,000 | Visibility in units of 10 m |
| `Dew_Point_C` | Dew point temperature(°C) | float | -30.6 to 27.2 | Dew point |
| `Solar_Radiation_MJm2` | Solar Radiation (MJ/m2) | float | 0.00 to 3.52 | Solar radiation |
| `Rainfall_mm` | Rainfall(mm) | float | 0.0 to 35.0 | Rain in that hour |
| `Snowfall_cm` | Snowfall (cm) | float | 0.0 to 8.8 | Snow in that hour |
| `Seasons` | Seasons | category | Winter, Spring, Summer, Autumn | Season |
| `Holiday` | Holiday | category | Holiday, No Holiday | Public holiday flag (432 holiday hours) |
| `Functioning_Day` | Functioning Day | category | Yes, No | Whether the system was operating (295 hours are "No") |

Two columns are derived later (Section C): `Demand_Class` (the target) and the calendar features `Month` and `DayOfWeek`.

### B.3 Cleaning log (before and after)

| Check | Before | Action | After | Why |
| :-- | :-- | :-- | :-- | :-- |
| Column names | Spaces, brackets and ° in 10 headers | Renamed to snake_case (`Temperature(°C)` to `Temperature_C`) | 14 clean names | Safe to use in code (cell 6) |
| `Date` type | `str` | `pd.to_datetime(format="%d/%m/%Y")` | `datetime64` | Lets us extract month and weekday (cells 5 and 7) |
| Missing values | `df.isna().sum()` = 0 | None needed | 0 | Nothing to impute (cell 6) |
| Duplicate rows | `df.duplicated().sum()` = 0 | None needed | 0 | Nothing to drop (cell 6) |
| Row count | 8,760 | No rows removed | 8,760 | Nothing invalid was found |
| Closed hours (`Functioning_Day = "No"`) | 295 hours, all with 0 rentals | **Kept** | 295 | The system really was closed, and the model should learn that closed means Low demand |

The data was already clean, so cleaning is limited to renaming and type conversion. No rows were deleted and no values were changed.

### B.4 Exploratory data analysis

**Figure 1. Distribution of hourly rentals with the classification cutoff** (`figures/eda_target_distribution.png`, Notebook cell 9).
Hourly rentals are right-skewed (skewness 1.15). The mean is 704.6 and the median is 504.5, and the highest hour reached 3,556. We use the median as the cutoff, so hours with 505 or more rentals are High and the rest are Low. That gives exactly **4,380 Low and 4,380 High** hours, so accuracy is meaningful and a model cannot score well by always guessing one class.

**Figure 2. Average hourly rentals by season** (`figures/eda_hourly_commute_by_season.png`, cell 10).
Demand follows a commuter pattern: an 8 AM peak (all-season mean about 1,016 bikes) and a larger 6 PM peak (about 1,503). In Spring, Summer and Autumn the 8 AM average is 1,014 to 1,419 and the 6 PM average is 1,653 to 2,135. In Winter (mean temperature -2.5 °C) demand stays flat and low, and no hour of the day averages above 438 rentals. Winter is the clearest case where the hour of day only matters together with the weather.

**Figure 3. Rentals by weather and service status** (`figures/eda_weather_impact.png`, cell 11).
The median is **606** rentals in clear hours (7,817 hours), **151** in snow-only hours (415 hours) and **67** in rain hours (528 hours). In the 295 hours when the system was not functioning, rentals are exactly 0.

Together the three figures show that hour, temperature, precipitation and service status carry most of the signal.

---

## C. Feature Engineering and Selection

### C.1 Target and leakage control

`Demand_Class = 1` if `Rented_Bike_Count >= 505`, else `0` (cell 14). Keeping `Rented_Bike_Count` in the features would let the model read the answer, so it is dropped from X together with the raw `Date`. X has **14 columns**.

### C.2 Calendar features

`Month` (1 to 12) and `DayOfWeek` (0 = Monday to 6 = Sunday) are extracted from `Date` (cell 14). A raw date string cannot be used by the models, but these two numbers capture the yearly and weekly cycles.

### C.3 Correlation and feature selection

**Figure 4. Pearson correlation matrix** (`figures/correlation_heatmap.png`, cell 16). Correlation with raw rental count:

| Feature | r |
| :-- | :-: |
| `Temperature_C` | +0.54 |
| `Hour` | +0.41 |
| `Dew_Point_C` | +0.38 |
| `Solar_Radiation_MJm2` | +0.26 |
| `Visibility_10m` | +0.20 |
| `Month` | +0.13 |
| `Wind_Speed_ms` | +0.12 |
| `DayOfWeek` | -0.03 |
| `Rainfall_mm` | -0.12 |
| `Snowfall_cm` | -0.14 |
| `Humidity_Pct` | -0.20 |

Warmer, later-in-the-day and sunnier hours go with more rentals. Humidity, snow and rain go with fewer. `DayOfWeek` is close to zero but is kept because the models can still use it.

### C.4 Multicollinearity check

`Temperature_C` and `Dew_Point_C` have r = **0.913**, so they carry almost the same information. We compared Logistic Regression with and without `Dew_Point_C` (cell 23): accuracy was **86.93%** with it and **86.99%** without it. That is a difference of one test hour out of 1,752, so dropping the column does not change the result. We keep all features in the main models and report the ablation as a check.

### C.5 Preprocessing pipeline

Preprocessing is a `ColumnTransformer` fitted on the training set only (cell 21):

- **Numeric (11 columns):** `StandardScaler` (mean 0, standard deviation 1): `Hour`, `Temperature_C`, `Humidity_Pct`, `Wind_Speed_ms`, `Visibility_10m`, `Dew_Point_C`, `Solar_Radiation_MJm2`, `Rainfall_mm`, `Snowfall_cm`, `Month`, `DayOfWeek`.
- **Categorical (3 columns):** `OneHotEncoder(drop="first")`. `Seasons` becomes 3 columns (Spring, Summer, Winter; Autumn is the reference). `Holiday` becomes 1 column (`Holiday_No Holiday`; the reference is Holiday). `Functioning_Day` becomes 1 column (`Functioning_Day_Yes`; the reference is No).

The 14 input features become **16 numeric predictors** (11 + 3 + 1 + 1). Scaling and encoding are learned from the 7,008 training hours and then applied unchanged to the 1,752 test hours.
