#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROPOSAL FIGURES FOR BAYESIAN ICE SHEET MODEL EVALUATION

Produces figures illustrating:

1. Posterior weight distribution
2. Ranked model weights
3. Cumulative probability curve
4. Parameter sensitivity
5. Posterior parameter distributions
6. Parameter-space heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# LOAD MODEL WEIGHTS
# ==========================================================

weights_table = pd.read_csv(
"/Users/sp53972/Documents/GitHub/PSUISM_HBM_V1/weights_params.csv"
)

weights = weights_table["weight"].values
weights = weights / weights.sum()

print(weights_table)

# ==========================================================
# FIGURE 1 — WEIGHT DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,5))

plt.hist(weights, bins=10, edgecolor="black")

plt.xlabel("Posterior Model Weight")
plt.ylabel("Number of Models")

plt.title("Distribution of Bayesian Model Weights")

plt.tight_layout()

plt.savefig("weight_distribution.png", dpi=300)

plt.show()


# ==========================================================
# FIGURE 2 — RANKED MODEL WEIGHTS
# ==========================================================

sorted_table = weights_table.sort_values("weight", ascending=False)

plt.figure(figsize=(8,5))

plt.bar(sorted_table["model_id"], sorted_table["weight"])

plt.xticks(rotation=90)

plt.ylabel("Posterior Weight")
plt.xlabel("Model Ensemble Member")

plt.title("Model Skill Ranking from Bayesian Evaluation")

plt.tight_layout()

plt.savefig("ranked_model_weights.png", dpi=300)

plt.show()


# ==========================================================
# FIGURE 3 — CUMULATIVE POSTERIOR PROBABILITY
# ==========================================================

w_sorted = np.sort(weights)[::-1]
cum_prob = np.cumsum(w_sorted)

plt.figure(figsize=(6,5))

plt.plot(cum_prob, marker="o")

plt.xlabel("Number of Models")
plt.ylabel("Cumulative Posterior Probability")

plt.title("Concentration of Posterior Probability")

plt.tight_layout()

plt.savefig("cumulative_probability.png", dpi=300)

plt.show()


# ==========================================================
# FIGURE 4 — PARAMETER SENSITIVITY
# ==========================================================

# Adjust column names if needed
parameters = ["OCFACMULT", "DTAUASTH", "CALVNICK"]

for p in parameters:

    if p in weights_table.columns:

        plt.figure(figsize=(6,5))

        plt.scatter(
            weights_table[p],
            weights_table["weight"],
            s=60
        )

        plt.xlabel(p)
        plt.ylabel("Posterior Weight")

        plt.title(f"Model Skill Sensitivity to {p}")

        plt.tight_layout()

        plt.savefig(f"sensitivity_{p}.png", dpi=300)

        plt.show()


# ==========================================================
# FIGURE 5 — POSTERIOR PARAMETER DISTRIBUTIONS
# ==========================================================

for p in parameters:

    if p in weights_table.columns:

        plt.figure(figsize=(6,5))

        plt.hist(
            weights_table[p],
            weights=weights,
            bins=10,
            edgecolor="black"
        )

        plt.xlabel(p)
        plt.ylabel("Posterior Probability")

        plt.title(f"Posterior Distribution of {p}")

        plt.tight_layout()

        plt.savefig(f"posterior_{p}.png", dpi=300)

        plt.show()


# ==========================================================
# FIGURE 6 — PARAMETER SPACE HEATMAP
# ==========================================================

if "OCFACMULT" in weights_table.columns and "DTAUASTH" in weights_table.columns:

    pivot = weights_table.pivot_table(
        values="weight",
        index="OCFACMULT",
        columns="DTAUASTH",
        aggfunc=np.mean
    )

    plt.figure(figsize=(7,6))

    plt.imshow(pivot, origin="lower", aspect="auto")

    plt.colorbar(label="Posterior Weight")

    plt.xlabel("DTAUASTH")
    plt.ylabel("OCFACMULT")

    plt.title("Posterior Probability in Parameter Space")

    plt.tight_layout()

    plt.savefig("parameter_space_heatmap.png", dpi=300)

    plt.show()


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

print("\n======================================")
print("ENSEMBLE SUMMARY")
print("======================================")

print("Number of models:", len(weights))

print("Max weight:", np.max(weights))
print("Min weight:", np.min(weights))

print("\nTop 5 models:")

print(sorted_table.head())

print("\nDone.")