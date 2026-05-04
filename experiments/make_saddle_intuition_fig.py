"""
Generate the "what does stuck at a saddle look like" intuition figure for
the presentation.

Geometry we want:
  high loss  ->  descent #1  ->  LONG plateau  ->  descent #2  ->  minimum
                                       ^                              ^
                                  stuck here                  true minimum
                                  (||grad f|| ~= 0)            (well below
                                                                the plateau)

Built as f(theta) = A*sigmoid(-(theta - a1))  +  B*sigmoid(-(theta - a2)).
The first sigmoid handles the high-loss -> plateau drop.
The second sigmoid handles the plateau -> minimum drop.
Between the two transitions, the loss is genuinely flat (gradient ~= 0).

Output:
  figures/intuition_saddle_plateau.png  (300 dpi, palette matched to the
                                         Beamer slate-blue + accent-red).

Run from the project root:
  uv run python experiments/make_saddle_intuition_fig.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Palette matches the Beamer presentation (slateblue + slateaccent).
SLATE  = np.array([40, 55, 90])  / 255.0
ACCENT = np.array([180, 40, 55]) / 255.0
LIGHT  = np.array([235, 238, 245]) / 255.0
TEXT   = np.array([35, 35, 35])  / 255.0


def loss_curve(theta: np.ndarray) -> np.ndarray:
    """Two stacked sigmoid drops separated by a plateau.

    Drop 1 (high loss -> plateau) at theta = -1, height A.
    Drop 2 (plateau -> minimum)   at theta =  4, height B.
    Between ~0 and ~3 the loss sits at ~B (the plateau).
    Past theta = 4 the loss falls to ~0 (true minimum).

    Sigmoids are steep (k=4) so the inter-drop region has gradient ~0.
    """
    A, a1, k1 = 3.5, -1.0, 4.0
    B, a2, k2 = 1.5,  4.0, 4.0
    drop1 = A / (1.0 + np.exp( k1 * (theta - a1)))
    drop2 = B / (1.0 + np.exp( k2 * (theta - a2)))
    return drop1 + drop2


def grad_norm(theta: np.ndarray) -> np.ndarray:
    """|f'(theta)| computed numerically -- robust to any tweak of f."""
    return np.abs(np.gradient(loss_curve(theta), theta))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    figures_dir  = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    theta = np.linspace(-5.0, 8.0, 1300)
    f     = loss_curve(theta)
    g     = grad_norm(theta)

    # The plateau lives between the two sigmoid transitions: roughly
    # theta in [0, 3].  We try to detect it from the gradient (more
    # robust to function tweaks) but fall back to the analytic interval
    # if the threshold is too tight to catch any samples.
    eps_plateau    = 0.10
    detection_zone = (theta > 0.0) & (theta < 3.0)
    plateau_mask   = (g < eps_plateau) & detection_zone
    if plateau_mask.any():
        plateau_theta = theta[plateau_mask]
        plateau_lo, plateau_hi = float(plateau_theta.min()), float(plateau_theta.max())
    else:
        # Fallback: use the analytic plateau bounds. Should not normally fire.
        plateau_lo, plateau_hi = 0.5, 2.5

    # Stuck point: middle of the plateau.
    stuck_theta = 0.5 * (plateau_lo + plateau_hi)
    stuck_f     = float(loss_curve(np.array([stuck_theta]))[0])

    # True minimum: the rightmost low-loss region (past the second drop).
    min_idx     = np.argmin(f)
    min_theta   = float(theta[min_idx])
    min_f       = float(f[min_idx])

    # ---------- figure layout --------------------------------------------
    fig, (ax_loss, ax_grad) = plt.subplots(
        2, 1, figsize=(8.6, 5.2),
        gridspec_kw={"height_ratios": [3.0, 1.1], "hspace": 0.22},
        sharex=True,
    )

    # ===== top panel: the loss curve =====================================
    ax = ax_loss
    ax.plot(theta, f, color=SLATE, lw=2.8, zorder=3)

    # Shade the plateau region so the eye latches onto it
    ax.axvspan(plateau_lo, plateau_hi, color=ACCENT, alpha=0.10, zorder=1)

    # Plateau label (placed above the curve so it doesn't collide with markers)
    ax.text(0.5 * (plateau_lo + plateau_hi),
            stuck_f + 1.05,
            "plateau (saddle-like region)",
            ha="center", va="bottom", fontsize=10.5,
            color=ACCENT, alpha=0.9, fontweight="bold")

    # The stuck ball (on the plateau)
    ax.scatter([stuck_theta], [stuck_f],
               s=170, color=ACCENT, edgecolors="black",
               linewidths=1.0, zorder=6)
    ax.annotate(
        "stuck:  $\\|\\nabla f\\|\\!\\approx\\!0$",
        xy=(stuck_theta, stuck_f),
        xytext=(stuck_theta - 1.6, stuck_f + 0.55),
        ha="center", va="bottom",
        fontsize=11, color=ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                        shrinkA=0, shrinkB=4),
        zorder=7,
    )

    # The true minimum (well past the plateau)
    ax.scatter([min_theta], [min_f],
               s=160, color=SLATE, edgecolors="black",
               linewidths=1.0, marker="*", zorder=6)
    ax.annotate(
        "true minimum",
        xy=(min_theta, min_f),
        xytext=(min_theta - 1.4, min_f + 0.55),
        ha="center", va="bottom",
        fontsize=11, color=SLATE, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.3,
                        shrinkA=0, shrinkB=4),
        zorder=7,
    )

    # Curved arrow from stuck point to a downhill location past the plateau
    descent_x = plateau_hi + 0.7
    descent_y = float(loss_curve(np.array([descent_x]))[0])
    ax.annotate(
        "",
        xy=(descent_x, descent_y + 0.05),
        xytext=(stuck_theta + 0.15, stuck_f - 0.05),
        arrowprops=dict(
            arrowstyle="-|>", color=ACCENT, lw=1.7,
            connectionstyle="arc3,rad=-0.35", shrinkA=4, shrinkB=4,
        ),
        zorder=5,
    )
    ax.text(plateau_hi + 0.05, stuck_f - 0.42,
            "descent direction\nexists past the plateau",
            ha="left", va="top", fontsize=10,
            color=ACCENT, fontstyle="italic")

    ax.set_ylabel(r"loss  $f(\theta)$", fontsize=12)
    ax.set_ylim(-0.4, max(f) + 1.2)
    ax.set_title(
        "A 1-D loss with a flat plateau between two descents",
        fontsize=12, color=SLATE, pad=8, loc="left",
    )

    # ===== bottom panel: gradient norm ===================================
    ax = ax_grad
    ax.plot(theta, g, color=SLATE, lw=2.0, zorder=3)
    ax.axvspan(plateau_lo, plateau_hi, color=ACCENT, alpha=0.10, zorder=1)
    ax.axhline(eps_plateau, color=ACCENT, ls="--", lw=1.0, alpha=0.8, zorder=2)
    ax.text(theta.min() + 0.2, eps_plateau + 0.06,
            r"stagnation threshold $\varepsilon$",
            color=ACCENT, fontsize=9.5, va="bottom")
    ax.set_xlabel(r"parameter  $\theta$", fontsize=12)
    ax.set_ylabel(r"$\|\nabla f\|$", fontsize=12)
    ax.set_ylim(-0.04, max(g) * 1.12)

    # ===== global polish =================================================
    for a in (ax_loss, ax_grad):
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
        a.spines["left"].set_color(TEXT)
        a.spines["bottom"].set_color(TEXT)
        a.tick_params(colors=TEXT, labelsize=10)
        a.set_xlim(theta.min(), theta.max())
        a.grid(True, alpha=0.16, linestyle="-", linewidth=0.5, zorder=0)

    fig.suptitle(
        "What does “stuck at a saddle” look like?",
        fontsize=14, color=SLATE, y=0.998, fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.11)

    out = figures_dir / "intuition_saddle_plateau.png"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
