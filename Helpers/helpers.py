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
    Chi-Square EDA function for:
    - Categorical vs Categorical (e.g., Sports vs GradeClass)
    
    Performs:
        - Contingency table
        - Chi-square test of independence
        - Stacked bar chart visualization
      Uses letter grades (A–F) for readability, while keeping the
      underlying GradeClass numeric for modeling.
    """

    # Local mapping (does NOT alter the original dataset)
    mapping = {
        0.0: "A",
        1.0: "B",
        2.0: "C",
        3.0: "D",
        4.0: "F"
    }

    # Temporary mapped labels
    temp_target = df[target_col].map(mapping)

    print("=" * 80)
    print(f"Chi-Square EDA: {cat_col} vs {target_col}")
    print("=" * 80)

    # --------------------------------------------------
    # Build contingency table
    # --------------------------------------------------
    table = pd.crosstab(df[cat_col], temp_target)

    # --------------------------------------------------
    # Run Chi-Square Test
    # --------------------------------------------------
    chi2, p_value, dof, expected = chi2_contingency(table)

    print("\nChi-Square Test Results:")
    print(f"Chi² statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Significant association (reject H₀).")
    else:
        print("Result: No significant association (fail to reject H₀).")

    # --------------------------------------------------
    # Stacked Bar Chart
    # --------------------------------------------------
    table.plot(
        kind="bar",
        stacked=True,
        colormap="Pastel1",
        figsize=(7,5)
    )

    plt.title(f"{cat_col} vs GradeClass (Stacked Bar Chart)")
    plt.xlabel(cat_col)
    plt.ylabel("Frequency")
    plt.xticks(rotation=0)
    plt.legend(title="Letter Grade")

    plt.tight_layout()
    plt.show()

    print("\n")

def chisquare_test(table):
    stat, p_value, dof, expected = chi2_contingency(table, correction=False)
    # Interpret p-value
    alpha = 0.05
    print("The p-value is {}".format(p_value))
    if p_value <= alpha:
        print(f'{bold}YES{normal} Relationship between variables')
    else:
        print(f'{bold}NO{normal} relationship between variables.')

def plot_feature_importances(df, title, palette="viridis", top_n=15):
    """
    Plots the top N feature importances from a dataframe with
    columns ['Feature', 'Importance'].
    Compatible with Seaborn 0.14+ (requires hue assignment).
    """

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df.head(top_n),
        x="Importance",
        y="Feature",
        hue="Feature",
        palette=palette,
        legend=False
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

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

def run_eda_test(df, category_col, numeric_col):
    """
    Unified EDA function for:
    - Binary categorical vars (t-test + boxplot)
    - Multi-category vars (ANOVA + boxplot)

    Plots:
        Boxplot only (simple & professional)
    """

    groups = df[category_col].dropna().unique()
    n_groups = len(groups)

    print("=" * 80)
    print(f"EDA Analysis: {numeric_col} by {category_col}")
    print("=" * 80)

    # --------------------------------------------------
    #  Binary Category → T-Test + Boxplot
    # --------------------------------------------------
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
            print("Result: Significant difference between groups (p < 0.05).")
        else:
            print("Result: No significant difference between groups (p ≥ 0.05).")

        # ----- Boxplot -----
        plt.figure(figsize=(7,5))
        sns.boxplot(data=df, x=category_col, y=numeric_col)
        plt.title(f"{numeric_col} by {category_col}")
        plt.show()

    # --------------------------------------------------
    #  Multi-Category → ANOVA + Boxplot
    # --------------------------------------------------
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
            print("Result: Significant differences among groups (p < 0.05).")
        else:
            print("Result: No significant differences (p ≥ 0.05).")

        # ----- Boxplot -----
        plt.figure(figsize=(8,5))
        sns.boxplot(data=df, x=category_col, y=numeric_col)
        plt.title(f"{numeric_col} by {category_col}")
        plt.show()

    else:
        raise ValueError(f"Column '{category_col}' has fewer than 2 groups.")

    print("\n")

def plot_multiple_hists(df, columns, label_maps=None, column_descriptions=None,
                        ncols=3, figsize=(15, 8)):
    """
    Creates subplots of histograms / countplots side-by-side
    using label maps and descriptive titles.
    Automatically respects the order of label_maps for categorical columns.
    """
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        col_data = df[col]
        desc = column_descriptions.get(col, col)

        # Apply mapping if exists
        if label_maps and col in label_maps:
            # convert to int safely before mapping
            col_data = pd.to_numeric(col_data, errors='coerce').fillna(-1).astype(int)
            data = col_data.map(label_maps[col])
            plot_type = "categorical"

            # ✅ Use order from mapping keys
            order = [label_maps[col][k] for k in label_maps[col].keys()]
        else:
            data = col_data
            plot_type = "numeric" if pd.api.types.is_numeric_dtype(col_data) else "categorical"
            order = sorted(data.dropna().unique()) if plot_type == "categorical" else None

        # Plot
        if plot_type == "numeric":
            sns.histplot(data.dropna(), bins=10, kde=True,
                         color='skyblue', edgecolor='black', ax=ax)
            ax.set_ylabel("Frequency")
        else:
            sns.countplot(x=data, color='lightgreen', order=order, ax=ax)
            ax.set_ylabel("Count")

        ax.set_xlabel(col)
        ax.set_title(f"{col} — {desc}", fontsize=10)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()