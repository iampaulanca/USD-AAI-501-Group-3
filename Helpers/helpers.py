# ============================================================
# Helper.py — Utility Functions for EDA and Model Evaluation
# ============================================================

# --- Visualization ---
import seaborn as sns
import matplotlib.pyplot as plt

# --- Math / Core ---
import math
import pandas as pd

# --- Statistical Tests ---
from scipy.stats import ttest_ind, f_oneway, chi2_contingency

# --- Regression Metrics ---
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --- Classification Metrics ---
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# Chi-Square Test + Visualization
# ============================================================

def run_chi_square_test(df, cat_col, target_col="GradeClass"):
    """
    Chi-Square EDA function for categorical vs categorical relationships.
    Outputs a contingency table, chi-square results, and a stacked bar chart.
    """

    print("=" * 80)
    print(f"Chi-Square EDA: {cat_col} vs {target_col}")
    print("=" * 80)

    # Contingency table
    table = pd.crosstab(df[cat_col], df[target_col])

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(table)

    print("\nChi-Square Test Results:")
    print(f"Chi² statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Significant association (reject H₀).")
    else:
        print("Result: No significant association (fail to reject H₀).")

    # Stacked bar plot
    plt.figure(figsize=(7, 5))
    table.plot(kind="bar", stacked=True, colormap="Pastel1")
    plt.title(f"{cat_col} vs {target_col} (Stacked Bar Chart)")
    plt.xlabel(cat_col)
    plt.ylabel("Frequency")
    plt.xticks(rotation=0)
    plt.legend(title=target_col)
    plt.tight_layout()
    plt.show()
    print("\n")


# ============================================================
# Regression Evaluation
# ============================================================

def evaluate_regression(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"--- {model_name} ---")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print()


# ============================================================
# Classification Evaluation
# ============================================================

def evaluate_classifier(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print()


# ============================================================
# EDA Tests (T-Test + ANOVA)
# ============================================================

def run_eda_test(df, category_col, numeric_col):
    """
    Unified EDA function for:
      - Binary categorical → t-test
      - Multi-category → ANOVA
    Includes boxplot visualization.
    """

    groups = df[category_col].dropna().unique()
    n_groups = len(groups)

    print("=" * 80)
    print(f"EDA Analysis: {numeric_col} by {category_col}")
    print("=" * 80)

    # --- Binary Categorical: T-Test ---
    if n_groups == 2:
        print("\nRunning Independent Samples T-Test...")

        g1, g2 = groups[0], groups[1]
        data1 = df[df[category_col] == g1][numeric_col]
        data2 = df[df[category_col] == g2][numeric_col]

        t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

        print(f"\nGroup {g1} mean: {data1.mean():.3f}")
        print(f"Group {g2} mean: {data2.mean():.3f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("Result: Significant difference (p < 0.05).")
        else:
            print("Result: No significant difference (p ≥ 0.05).")

        plt.figure(figsize=(7,5))
        sns.boxplot(data=df, x=category_col, y=numeric_col)
        plt.title(f"{numeric_col} by {category_col}")
        plt.show()

    # --- Multi-Category: ANOVA ---
    elif n_groups >= 3:
        print("\nRunning One-Way ANOVA...")

        samples = [df[df[category_col] == g][numeric_col] for g in groups]
        f_stat, p_value = f_oneway(*samples)

        print("\nGroup Means:")
        for g in groups:
            print(f"  {g}: {df[df[category_col] == g][numeric_col].mean():.3f}")

        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.6f}")

        if p_value < 0.05:
            print("Result: Significant differences (p < 0.05).")
        else:
            print("Result: No significant differences (p ≥ 0.05).")

        plt.figure(figsize=(8,5))
        sns.boxplot(data=df, x=category_col, y=numeric_col)
        plt.title(f"{numeric_col} by {category_col}")
        plt.show()

    else:
        raise ValueError(f"Column '{category_col}' has fewer than 2 groups.")

    print("\n")


# ============================================================
# Histogram / Countplot Grid
# ============================================================

def plot_multiple_hists(df, columns, label_maps=None, column_descriptions=None,
                        ncols=3, figsize=(15, 8)):
    """
    Plots histograms and countplots for multiple columns using subplots.
    Applies label maps and descriptive titles automatically.
    """

    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        col_data = df[col]
        desc = column_descriptions.get(col, col)

        if label_maps and col in label_maps:
            col_data = pd.to_numeric(col_data, errors="coerce").fillna(-1).astype(int)
            data = col_data.map(label_maps[col])
            plot_type = "categorical"
            order = [label_maps[col][k] for k in label_maps[col]]
        else:
            data = col_data
            plot_type = "numeric" if pd.api.types.is_numeric_dtype(col_data) else "categorical"
            order = sorted(data.dropna().unique()) if plot_type == "categorical" else None

        if plot_type == "numeric":
            sns.histplot(data.dropna(), bins=10, kde=True,
                         color="skyblue", edgecolor="black", ax=ax)
        else:
            sns.countplot(x=data, order=order, color="lightgreen", ax=ax)

        ax.set_title(f"{col} — {desc}", fontsize=10)
        ax.set_xlabel(col)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()