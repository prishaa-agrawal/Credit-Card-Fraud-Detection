"""
Credit Card Fraud Detection - Exploratory Data Analysis
========================================================
Run this script to visualize the dataset before training.
Outputs charts into the outputs/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")


# ── Load or generate data ──────────────────────────────────────────────────────
try:
    df = pd.read_csv("data/transactions.csv")
    print("Loaded data/transactions.csv")
except FileNotFoundError:
    print("Dataset not found. Run train.py first to generate it.")
    import sys; sys.exit(1)


print("\n── Dataset Overview ──────────────────────────────")
print(df.describe().round(2))
print(f"\nClass balance:\n{df['Class'].value_counts()}")


# ── Figure 1: Class Distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Credit Card Fraud — Class Distribution", fontsize=15, fontweight="bold")

counts = df["Class"].value_counts()
colors = ["#4CAF50", "#F44336"]

axes[0].bar(["Legitimate", "Fraud"], counts.values, color=colors, edgecolor="white", linewidth=1.2)
axes[0].set_title("Transaction Counts")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=["Legitimate", "Fraud"], colors=colors,
            autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"})
axes[1].set_title("Class Proportion")

plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150, bbox_inches="tight")
print("\n✅  Saved: outputs/class_distribution.png")
plt.close()


# ── Figure 2: Amount Distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Transaction Amount by Class", fontsize=15, fontweight="bold")

for ax, log_scale in zip(axes, [False, True]):
    for cls, color, label in [(0, "#4CAF50", "Legitimate"), (1, "#F44336", "Fraud")]:
        subset = df[df["Class"] == cls]["amount"]
        ax.hist(subset, bins=60, alpha=0.6, color=color, label=label, edgecolor="none")
    ax.set_xlabel("Transaction Amount ($)" + (" [log scale]" if log_scale else ""))
    ax.set_ylabel("Frequency")
    ax.legend()
    if log_scale:
        ax.set_yscale("log")
        ax.set_title("Log Scale")
    else:
        ax.set_title("Linear Scale")

plt.tight_layout()
plt.savefig("outputs/amount_distribution.png", dpi=150, bbox_inches="tight")
print("✅  Saved: outputs/amount_distribution.png")
plt.close()


# ── Figure 3: Transactions by Hour ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
for cls, color, label in [(0, "#4CAF50", "Legitimate"), (1, "#F44336", "Fraud")]:
    subset = df[df["Class"] == cls]["hour"].value_counts().sort_index()
    ax.plot(subset.index, subset.values, marker="o", color=color, label=label, linewidth=2)

ax.set_title("Transaction Volume by Hour of Day", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour (0 = Midnight)")
ax.set_ylabel("Number of Transactions")
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig("outputs/transactions_by_hour.png", dpi=150, bbox_inches="tight")
print("✅  Saved: outputs/transactions_by_hour.png")
plt.close()


# ── Figure 4: Feature Correlation Heatmap ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = ["amount", "hour", "v1", "v2", "v3", "v4", "v5", "n_transactions_last_hour", "Class"]
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
print("✅  Saved: outputs/correlation_heatmap.png")
plt.close()


# ── Figure 5: Boxplot — V features by class ───────────────────────────────────
v_features = ["v1", "v2", "v3", "v4", "v5"]
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("PCA Feature Distributions: Fraud vs Legitimate", fontsize=14, fontweight="bold")

for ax, feat in zip(axes, v_features):
    data = [df[df["Class"] == 0][feat], df[df["Class"] == 1][feat]]
    bp = ax.boxplot(data, patch_artist=True, labels=["Legit", "Fraud"])
    bp["boxes"][0].set_facecolor("#4CAF5080")
    bp["boxes"][1].set_facecolor("#F4433680")
    ax.set_title(feat.upper())
    ax.set_xlabel("Class")

plt.tight_layout()
plt.savefig("outputs/feature_boxplots.png", dpi=150, bbox_inches="tight")
print("✅  Saved: outputs/feature_boxplots.png")
plt.close()

print("\n📊  EDA complete! Check the outputs/ folder for all charts.\n")
