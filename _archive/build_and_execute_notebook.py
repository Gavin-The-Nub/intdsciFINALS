"""
Script to run the full Seoul Bike Machine Learning Pipeline
and generate all output plots, metrics, and the Jupyter Notebook (.ipynb).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Set visual style
sns.set_theme(style="whitegrid")


print("=== 1. LOADING DATASET ===")
data_path = "SeoulBikeData.csv"
df = pd.read_csv(data_path, encoding="latin1")

# Standardize column names
rename_map = {
    'Temperature(\xb0C)': 'Temperature_C',
    'Dew point temperature(\xb0C)': 'Dew_Point_C',
    'Humidity(%)': 'Humidity_Pct',
    'Wind speed (m/s)': 'Wind_Speed_ms',
    'Visibility (10m)': 'Visibility_10m',
    'Solar Radiation (MJ/m2)': 'Solar_Radiation_MJm2',
    'Rainfall(mm)': 'Rainfall_mm',
    'Snowfall (cm)': 'Snowfall_cm'
}
df.rename(columns=rename_map, inplace=True)

print("Shape:", df.shape)
print("Missing values total:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Create output folder for figures
os.makedirs("figures", exist_ok=True)

# === EDA Plot 1: Target Distribution & Median Threshold ===
print("\n=== GENERATING PLOT 1: TARGET DISTRIBUTION ===")
median_val = df['Rented Bike Count'].median()
mean_val = df['Rented Bike Count'].mean()

plt.figure(figsize=(9, 5), dpi=300)
sns.histplot(df['Rented Bike Count'], bins=50, kde=True, color='#1f77b4', edgecolor='white', alpha=0.6)
plt.axvline(median_val, color='#d62728', linestyle='--', linewidth=2, label=f'Median Threshold ({median_val:.1f} bikes/hr)')
plt.axvline(mean_val, color='#2ca02c', linestyle=':', linewidth=2, label=f'Mean ({mean_val:.1f} bikes/hr)')
plt.title("Distribution of Hourly Rented Bike Count with Classification Cutoff", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Rented Bike Count (Bikes / Hour)", fontsize=11)
plt.ylabel("Frequency (Hours)", fontsize=11)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("figures/eda_target_distribution.png")
plt.close()

# === EDA Plot 2: Hourly Commute Trends by Season ===
print("=== GENERATING PLOT 2: HOURLY COMMUTE CURVE BY SEASON ===")
hourly_season = df.groupby(['Hour', 'Seasons'])['Rented Bike Count'].mean().reset_index()

plt.figure(figsize=(10, 5), dpi=300)
palette = {'Winter': '#1f77b4', 'Spring': '#2ca02c', 'Summer': '#ff7f0e', 'Autumn': '#d62728'}
sns.lineplot(data=hourly_season, x='Hour', y='Rented Bike Count', hue='Seasons', palette=palette, marker='o', linewidth=2.2)
plt.title("Average Hourly Bike Rental Demand Across Four Seasons", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Hour of the Day (0 to 23)", fontsize=11)
plt.ylabel("Average Rented Bike Count", fontsize=11)
plt.xticks(range(0, 24))
plt.axvspan(7.5, 9, color='#ffbb78', alpha=0.25, label='Morning Rush (8 AM)')
plt.axvspan(17.5, 19, color='#aec7e8', alpha=0.25, label='Evening Rush (6 PM)')
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("figures/eda_hourly_commute_by_season.png")
plt.close()

# === EDA Plot 3: Weather & Precipitation Impact ===
print("=== GENERATING PLOT 3: WEATHER IMPACT ===")
df['Has_Rain'] = df['Rainfall_mm'].apply(lambda x: 'Rain (>0mm)' if x > 0 else 'No Rain (0mm)')
df['Has_Snow'] = df['Snowfall_cm'].apply(lambda x: 'Snow (>0cm)' if x > 0 else 'No Snow (0cm)')

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
sns.boxplot(data=df, x='Has_Rain', y='Rented Bike Count', ax=axes[0], palette=['#43a2ca', '#e0f3f8'])
axes[0].set_title("Rental Demand: Rain vs. No Rain", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Rain Condition", fontsize=10)
axes[0].set_ylabel("Rented Bike Count", fontsize=10)

sns.boxplot(data=df, x='Functioning Day', y='Rented Bike Count', ax=axes[1], palette=['#31a354', '#de2d26'])
axes[1].set_title("Rental Demand by Functioning Day Status", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Functioning Day", fontsize=10)
axes[1].set_ylabel("Rented Bike Count", fontsize=10)

plt.tight_layout()
plt.savefig("figures/eda_weather_impact.png")
plt.close()

# === TARGET CREATION & PREPARATION ===
# Classification target: 0 = Low Demand (<505), 1 = High Demand (>=505)
df['Demand_Class'] = (df['Rented Bike Count'] >= 505).astype(int)

# Extract Date Features
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Target separation and drop leaked raw target and original date
y = df['Demand_Class']
X = df.drop(columns=['Rented Bike Count', 'Demand_Class', 'Date', 'Has_Rain', 'Has_Snow'])

print("\n=== FEATURE MATRIX (X) & TARGET (y) ===")
print("Feature Columns:", list(X.columns))
print("Class Distribution:\n", y.value_counts(normalize=True))

# === CORRELATION MATRIX PLOT ===
numeric_cols = [
    'Hour', 'Temperature_C', 'Humidity_Pct', 'Wind_Speed_ms',
    'Visibility_10m', 'Dew_Point_C', 'Solar_Radiation_MJm2',
    'Rainfall_mm', 'Snowfall_cm', 'Month', 'DayOfWeek'
]
corr_df = df[numeric_cols + ['Rented Bike Count']].corr()

plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
plt.title("Correlation Matrix of Seoul Bike Numerical Variables", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png")
plt.close()

# === DATA SPLITTING (Strict 80-20 split with random_state=42) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Test class distribution:\n{y_test.value_counts()}")

# Categorical and Numerical column definition
categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Build Preprocessor
preprocessor_full = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# Reduced Preprocessor (Without Dew_Point_C to test collinearity ablation)
numerical_cols_reduced = [col for col in numerical_cols if col != 'Dew_Point_C']
preprocessor_reduced = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_reduced),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# === MODELS DEFINITION ===
models = {
    'Baseline (Most Frequent)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': Pipeline([
        ('prep', preprocessor_full),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Logistic Regression (No Dew Point)': Pipeline([
        ('prep', preprocessor_reduced),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest Classifier': Pipeline([
        ('prep', preprocessor_full),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1))
    ])
}

results = []
trained_models = {}
y_preds = {}

print("\n=== MODEL TRAINING AND EVALUATION ===")
for name, model in models.items():
    if name == 'Baseline (Most Frequent)':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    trained_models[name] = model
    y_preds[name] = y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'TP': cm[1, 1]
    })
    print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

results_df = pd.DataFrame(results)
print("\n=== SUMMARY EVALUATION TABLE ===")
print(results_df.to_string(index=False))

# === CONFUSION MATRIX PLOTS (1x3 subplots) ===
print("\n=== GENERATING CONFUSION MATRICES PLOT ===")
eval_models = [
    'Baseline (Most Frequent)',
    'Logistic Regression',
    'Random Forest Classifier'
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
for idx, m_name in enumerate(eval_models):
    cm = confusion_matrix(y_test, y_preds[m_name])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Low (<505)', 'High (>=505)']
    )
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"{m_name}\nAcc: {accuracy_score(y_test, y_preds[m_name]):.2%}", fontsize=11, fontweight='bold')
    axes[idx].grid(False)

plt.tight_layout()
plt.savefig("figures/confusion_matrices.png")
plt.close()

# === FEATURE IMPORTANCES FROM RANDOM FOREST ===
rf_pipeline = trained_models['Random Forest Classifier']
rf_clf = rf_pipeline.named_steps['clf']
feature_names = rf_pipeline.named_steps['prep'].get_feature_names_out()
# Clean names for readability
clean_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

importances = rf_clf.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': clean_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6), dpi=300)
sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("Top 10 Most Important Features in Random Forest Classifier", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Gini Feature Importance", fontsize=11)
plt.ylabel("Predictor Variable", fontsize=11)
plt.tight_layout()
plt.savefig("figures/feature_importances.png")
plt.close()

print("\nTop 10 Feature Importances:\n", feat_imp_df.head(10).to_string(index=False))
print("\nAll pipeline execution and plots generated successfully!")
