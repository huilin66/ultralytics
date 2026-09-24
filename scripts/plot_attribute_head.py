"""Draw the MAYOLO multi-attribute detection head used by the final model.

The figure follows the current ``v10MDetect`` implementation for
``sep=False`` and ``gin_margin_residual``:

    neck feature -> 3x3 Conv -> 3x3 Conv -> 1x1 Conv -> 10x2 logits
    -> binary risk margins -> GIN graph propagation -> residual margins
    -> reconstructed logits -> per-attribute softmax

The one-to-many and one-to-one branches instantiate the same attribute-head
architecture with independent parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def rounded_box(ax, x, y, w, h, text, *, face, edge="#243447", size=10, weight="normal", color="#17212b"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, *, color="#566573", lw=1.8, style="-|>", connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )


def panel(ax, x, y, w, h, title, *, face="#f7f9fb", edge="#b8c4cf"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.18,
        y + h - 0.28,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#243447",
        zorder=3,
    )
    return patch


def draw(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(16.5, 7.6), dpi=220)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Palette aligned with the existing MAYOLO figures.
    blue = "#d9edf8"
    blue_dark = "#4b9bd3"
    yellow = "#fff0b8"
    yellow_dark = "#d19a00"
    purple = "#eadffc"
    purple_dark = "#7554a6"
    green = "#dff2e1"
    green_dark = "#4c956c"
    orange = "#fbe4d5"
    orange_dark = "#c86b35"
    gray = "#eef2f5"

    ax.text(
        8.25,
        7.38,
        "MAYOLO Attribute Detection Head",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color="#1c2833",
    )
    rounded_box(
        ax,
        4.0,
        6.72,
        8.5,
        0.43,
        "one-to-many and one-to-one branches: same attribute-head architecture, independent parameters",
        face="#edf4fa",
        edge=blue_dark,
        size=10,
        weight="bold",
    )

    panel(ax, 0.25, 0.55, 7.55, 5.85, "A. Per-scale attribute prediction", face="#f8fbfd", edge="#aac5d8")
    panel(ax, 8.0, 0.55, 8.25, 5.85, "B. Attribute-level graph refinement", face="#fcfbff", edge="#c8b9e0")

    # Multi-scale input.
    rounded_box(ax, 0.65, 4.98, 1.45, 0.62, r"Neck feature $\mathbf{F}_s$", face=blue, edge=blue_dark, size=11, weight="bold")
    ax.text(1.38, 4.72, r"$s\in\{P3,P4,P5\}$", ha="center", va="top", fontsize=9, color="#50606e")

    # Current sep=False attribute branch.
    rounded_box(ax, 2.35, 4.98, 1.25, 0.62, "3×3 Conv", face=orange, edge=orange_dark, size=10, weight="bold")
    rounded_box(ax, 3.92, 4.98, 1.25, 0.62, "3×3 Conv", face=orange, edge=orange_dark, size=10, weight="bold")
    rounded_box(ax, 5.49, 4.98, 1.25, 0.62, "1×1 Conv", face=orange, edge=orange_dark, size=10, weight="bold")
    arrow(ax, 2.1, 5.29, 2.35, 5.29)
    arrow(ax, 3.6, 5.29, 3.92, 5.29)
    arrow(ax, 5.17, 5.29, 5.49, 5.29)

    rounded_box(
        ax,
        2.2,
        3.73,
        4.8,
        0.72,
        r"Raw attribute logits $\mathbf{z}_s\in\mathbb{R}^{B\times(AK)\times H_s\times W_s}$"
        "\n"
        r"$A=10$ attributes, $K=2$ mutually exclusive risk levels",
        face=yellow,
        edge=yellow_dark,
        size=10,
        weight="bold",
    )
    arrow(ax, 6.12, 4.98, 6.12, 4.45)

    # Candidate alignment and branch context.
    rounded_box(
        ax,
        0.72,
        1.15,
        6.55,
        1.45,
        "Attribute logits are produced at the same candidate locations\n"
        "as the box-regression and object-class branches.\n"
        "Decoded boxes are not inputs to this head.",
        face=gray,
        edge="#9aa8b5",
        size=9.6,
    )
    ax.text(3.98, 2.82, "parallel extension of the detection head", ha="center", va="center", fontsize=10, color="#5d6b77")
    arrow(ax, 3.95, 3.72, 3.95, 2.62, color="#8293a1")

    # Right: margin -> GIN -> softmax pipeline.
    rounded_box(
        ax,
        8.42,
        4.55,
        1.72,
        0.92,
        "Risk margin\n" r"$m_a=z_{a,1}-z_{a,0}$",
        face=yellow,
        edge=yellow_dark,
        size=10,
        weight="bold",
    )
    rounded_box(
        ax,
        10.52,
        4.36,
        2.35,
        1.30,
        "GIN graph propagation\n" r"$\mathcal{N}(a)$ aggregation" "\n"
        r"$(1+\epsilon)m_a+\sum_{j\in\mathcal{N}(a)}m_j$" "\nsmall MLP",
        face=purple,
        edge=purple_dark,
        size=9.5,
        weight="bold",
    )
    rounded_box(
        ax,
        13.28,
        4.55,
        1.65,
        0.92,
        "Residual update\n" r"$m'_a=m_a+\gamma\,\Phi_a$",
        face=green,
        edge=green_dark,
        size=9.5,
        weight="bold",
    )
    rounded_box(
        ax,
        15.25,
        4.55,
        0.75,
        0.92,
        "Softmax",
        face=orange,
        edge=orange_dark,
        size=9,
        weight="bold",
    )
    arrow(ax, 8.08, 4.09, 8.42, 4.98)
    arrow(ax, 10.14, 5.01, 10.52, 5.01)
    arrow(ax, 12.87, 5.01, 13.28, 5.01)
    arrow(ax, 14.93, 5.01, 15.25, 5.01)

    # Fixed category matrix and derived graph support.
    rounded_box(
        ax,
        8.35,
        1.78,
        1.82,
        0.78,
        "Fixed category matrix\n" r"$P$" "\n(training-set co-occurrence)",
        face=blue,
        edge=blue_dark,
        size=8.4,
        weight="bold",
    )
    rounded_box(
        ax,
        10.42,
        1.78,
        2.18,
        0.78,
        "Binary graph support " r"$A(P)$" "\n" r"$A_{ij}=1[P_{ij}>0],\ i\neq j$",
        face=blue,
        edge=blue_dark,
        size=8.4,
        weight="bold",
    )
    rounded_box(
        ax,
        12.92,
        1.78,
        3.05,
        0.78,
        "Non-trainable buffer\nused by the GIN operator",
        face=gray,
        edge="#8293a1",
        size=8.8,
        weight="bold",
    )
    arrow(ax, 10.17, 2.17, 10.42, 2.17)
    arrow(ax, 12.6, 2.17, 12.92, 2.17)
    arrow(ax, 11.51, 2.56, 11.65, 4.34, color=blue_dark, connectionstyle="arc3,rad=-0.18")

    rounded_box(
        ax,
        8.42,
        3.0,
        7.82,
        0.62,
        r"The graph acts on attribute channels at each candidate location, not across image locations.",
        face="#f0f6fb",
        edge="#a8c5d9",
        size=10,
    )
    arrow(ax, 9.38, 4.54, 9.38, 3.64, color="#8293a1")

    rounded_box(
        ax,
        8.35,
        0.78,
        7.62,
        0.62,
        "Output: 10 attributes × 2 risk probabilities\nreturned with each retained detection candidate",
        face=yellow,
        edge=yellow_dark,
        size=9.2,
        weight="bold",
    )
    arrow(ax, 15.62, 4.54, 15.62, 1.42, color="#8293a1", connectionstyle="arc3,rad=-0.15")

    fig.savefig(output, dpi=240, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\repository\academic research\mayolo\MAYOLO_ASOC\figs\attribute_head.png"),
    )
    args = parser.parse_args()
    draw(args.output)
    print(f"generated: {args.output}")
    print(f"generated: {args.output.with_suffix('.pdf')}")
    print(f"generated: {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
