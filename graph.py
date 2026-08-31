import matplotlib.pyplot as plt
import numpy as np

data = {
    "CIFAR-100": {
        "Fixed":    [-1.32, 0.52, -0.07, 0.37, 0.52, 0.60],
        "Adaptive": [-0.48, 0.81,  0.39, 0.29, 0.81, 1.23],
    },
    "TinyImageNet": {
        "Fixed":    [2.27, 3.12, 2.67, 2.14, 3.12, 3.61],
        "Adaptive": [2.96, 3.83, 3.07, 2.85, 3.83, 4.41],
    },
}

fixed_color = "#4C72B0"
adaptive_color = "#C44E52"

# gap between Local Epoch and Participation
x = np.array([0, 1, 2, 4, 5, 6])
width = 0.32

labels = ["1", "5", "10", "5%", "10%", "20%"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, dataset in zip(axes, ["CIFAR-100", "TinyImageNet"]):

    fixed = data[dataset]["Fixed"]
    adaptive = data[dataset]["Adaptive"]

    ax.bar(
        x - width/2,
        fixed,
        width,
        color=fixed_color,
        label=r"Fixed $\lambda=0.3$"
    )

    ax.bar(
        x + width/2,
        adaptive,
        width,
        color=adaptive_color,
        label="Adaptive KD"
    )

    # Plain baseline
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="gray",
        alpha=0.7
    )

    # Separate the two condition groups
    ax.axvline(
        3,
        linestyle=":",
        linewidth=1,
        color="gray",
        alpha=0.45
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_title(
        dataset,
        fontsize=14,
        fontweight="bold"
    )

    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)

    # Group labels
    ax.text(
        1, -0.13,
        "Local Epoch",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11
    )

    ax.text(
        5, -0.13,
        "Client Participation",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11
    )

axes[0].set_ylabel("Average Gain over Plain (pp)")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=2,
    frameon=True
)

fig.tight_layout(rect=[0, 0.06, 1, 0.92])

plt.savefig(
    "robustness_training_conditions_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()