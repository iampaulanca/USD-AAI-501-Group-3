import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import pandas as pd
import math

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