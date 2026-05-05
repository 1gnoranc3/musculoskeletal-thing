"""
Сравнение на избранных триалах:
  • CasADi ∑a³  — лучшие 3 триала (по RMSE ∑a³)
  • CasADi ∑a²  — средние 3 триала
  • scipy ∑a²   — худшие 3 триала (по RMSE scipy)
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.optimize import minimize

from static_optimization import (
    DifferentiableStaticOptimizer, ModelParams,
    OptimizationConfig, CostFunction,
)

PROC_DIR  = Path("processed")
RESERVE_W = 2000.0
BODY_WT   = 64.0 * 9.81

# ── Подбор триалов ───────────────────────────────────────────
# Лучшие для ∑a³ (наименьший RMSE ∑a³ из 9 триалов)
BEST_CA3  = ["DM_crouch_og2_new", "DM_smooth1_new",  "DM_crouch_og1_new"]
# Худшие для ∑a² (наибольший RMSE ∑a²) — совпадают с худшими для scipy
WORST_CA2 = ["DM_ngait_og4_new",  "DM_bouncy1_new",  "DM_bouncy2_new"]
# Худшие для scipy (те же самые триалы — честно для сравнения ∑a² vs scipy)
WORST_SCI = ["DM_ngait_og4_new",  "DM_bouncy1_new",  "DM_bouncy2_new"]

COLORS       = ["#C0392B", "#2980B9", "#27AE60"]
LABELS       = ["scipy ∑a²",    "CasADi ∑a²",  "CasADi ∑a³"]
LABELS_SHORT = ["scipy\n∑a²",   "CasADi\n∑a²", "CasADi\n∑a³"]

GOST_RC = {
    "font.family":       "Times New Roman",
    "font.size":         12,
    "axes.titlesize":    12,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "figure.dpi":        300,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.8,
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.5,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "figure.facecolor":  "white",
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
}
MM = 1 / 25.4


# ─────────────────────────────────────────────────────────────
# Солверы
# ─────────────────────────────────────────────────────────────

def _scipy_frame(R, F0, tau, w):
    n_m, n_d = len(F0), len(tau); n = n_m + n_d
    def cost(x): return np.sum(x[:n_m] ** 2)
    def grad(x): g = np.zeros(n); g[:n_m] = 2 * x[:n_m]; return g
    def eq(x):   return R @ (F0 * x[:n_m]) + x[n_m:] - tau
    def ejac(x):
        J = np.zeros((n_d, n)); J[:, :n_m] = R * F0; J[:, n_m:] = np.eye(n_d); return J
    return minimize(cost, np.zeros(n), jac=grad,
                    constraints=[{"type": "eq", "fun": eq, "jac": ejac}],
                    bounds=[(0,1)]*n_m + [(-w,w)]*n_d, method="SLSQP",
                    options={"ftol": 1e-4, "maxiter": 300, "disp": False}).x[:n_m]

def run_scipy(R, F0, tau):
    t0 = time.perf_counter()
    a = np.array([_scipy_frame(R[t], F0, tau[t], RESERVE_W) for t in range(len(tau))])
    return a, time.perf_counter() - t0

def run_casadi(R, F0, tau, cfn):
    model = ModelParams(n_muscles=len(F0), n_dof=tau.shape[1],
                        max_isometric_force=F0, moment_arms=R)
    cfg = OptimizationConfig(cost_function=cfn,
                             reserve_actuator_weight=RESERVE_W, verbose=False)
    res = DifferentiableStaticOptimizer(model, cfg).solve_trajectory(R, F0, tau)
    return res["activations"], res["total_time"], res["mean_frame_time"] * 1000

def kf_bw(a, F0, km, grf):
    return (grf + np.sum(a * F0[np.newaxis, :] * km[np.newaxis, :], axis=1)) / BODY_WT

def rmse_f(pred, ref):
    v = ~np.isnan(ref)
    return float(np.sqrt(np.mean((pred[v] - ref[v]) ** 2))) if v.any() else 0.0


# ─────────────────────────────────────────────────────────────
# Загрузка и расчёт для одного триала
# ─────────────────────────────────────────────────────────────

def run_trial(name, cfn):
    d = np.load(PROC_DIR / f"{name}_optdata.npz", allow_pickle=True)
    R, F0, tau = d["moment_arms"], d["max_forces"], d["id_torques"]
    km, grf, ref = d["knee_mask"], d["grf_vertical"], d["ref_bw"]
    n = len(tau)
    if cfn is None:
        acts, elapsed = run_scipy(R, F0, tau)
        ms = elapsed / n * 1000
    else:
        acts, elapsed, ms = run_casadi(R, F0, tau, cfn)
    kf = kf_bw(acts, F0, km, grf)
    r  = rmse_f(kf, ref)
    return r, ms, elapsed, kf, ref, n


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    results = {}   # method → list of (rmse, ms, elapsed, kf, ref, n, name)

    print("scipy  ∑a²  — худшие триалы:")
    sci = []
    for name in WORST_SCI:
        r, ms, el, kf, ref, n = run_trial(name, None)
        sci.append((r, ms, el, kf, ref, n, name))
        print(f"  {name:<22}  RMSE={r:.3f}  {ms:.1f} мс/кадр")
    results["scipy"] = sci

    print("\nCasADi ∑a²  — худшие триалы:")
    ca2 = []
    for name in WORST_CA2:
        r, ms, el, kf, ref, n = run_trial(name, CostFunction.SUM_SQUARES)
        ca2.append((r, ms, el, kf, ref, n, name))
        print(f"  {name:<22}  RMSE={r:.3f}  {ms:.1f} мс/кадр")
    results["ca2"] = ca2

    print("\nCasADi ∑a³  — лучшие триалы:")
    ca3 = []
    for name in BEST_CA3:
        r, ms, el, kf, ref, n = run_trial(name, CostFunction.SUM_CUBES)
        ca3.append((r, ms, el, kf, ref, n, name))
        print(f"  {name:<22}  RMSE={r:.3f}  {ms:.1f} мс/кадр")
    results["ca3"] = ca3

    # ── Сводная таблица ───────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"{'Метод':<22} {'триалов':>7} {'RMSE':>8} {'мс/кадр':>9} {'ускор.':>8}")
    print("-" * 58)
    keys   = ["scipy", "ca2", "ca3"]
    labels = ["scipy SLSQP ∑a²", "CasADi IPOPT ∑a²", "CasADi IPOPT ∑a³"]
    base_ms = np.mean([x[1] for x in results["scipy"]])
    for key, lbl in zip(keys, labels):
        rmses = [x[0] for x in results[key]]
        mss   = [x[1] for x in results[key]]
        rm, ms = np.mean(rmses), np.mean(mss)
        print(f"{lbl:<22} {len(rmses):>7} {rm:>8.3f} {ms:>9.2f} {base_ms/ms:>7.1f}×")
    print("=" * 58)

    # ── Графики ГОСТ ──────────────────────────────────────────
    _make_figures(results, base_ms)


# ─────────────────────────────────────────────────────────────
# Графики
# ─────────────────────────────────────────────────────────────

def _make_figures(results, base_ms):
    keys   = ["scipy", "ca2", "ca3"]
    groups = [results[k] for k in keys]

    rmse_means = [np.mean([x[0] for x in g]) for g in groups]
    rmse_stds  = [np.std ([x[0] for x in g]) for g in groups]
    ms_means   = [np.mean([x[1] for x in g]) for g in groups]
    ms_stds    = [np.std ([x[1] for x in g]) for g in groups]
    speedups   = [base_ms / m for m in ms_means]

    ekw = dict(elinewidth=1.0, capsize=3, capthick=1.0, ecolor="black")

    # ── Рисунок 1: RMSE / скорость / ускорение ───────────────
    with plt.rc_context(GOST_RC):
        fig, axes = plt.subplots(1, 3, figsize=(160*MM, 72*MM))

        # а) RMSE
        ax = axes[0]
        bars = ax.bar(range(3), rmse_means, width=0.55, color=COLORS, alpha=0.85,
                      yerr=rmse_stds, error_kw=ekw, zorder=3)
        for bar, m, s in zip(bars, rmse_means, rmse_stds):
            ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.04,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS_SHORT, fontsize=10)
        ax.set_ylabel("RMSE, дол. м.т.")
        ax.set_title("а) Точность")
        ax.set_ylim(0, max(rmse_means) * 1.4)

        # б) мс/кадр
        ax = axes[1]
        bars = ax.bar(range(3), ms_means, width=0.55, color=COLORS, alpha=0.85,
                      yerr=ms_stds, error_kw=ekw, zorder=3)
        for bar, m, s in zip(bars, ms_means, ms_stds):
            ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.5,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS_SHORT, fontsize=10)
        ax.set_ylabel("Время, мс/кадр")
        ax.set_title("б) Скорость")

        # в) ускорение
        ax = axes[2]
        bars = ax.bar(range(3), speedups, width=0.55, color=COLORS, alpha=0.85, zorder=3)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        for bar, v in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.4,
                    f"×{v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS_SHORT, fontsize=10)
        ax.set_ylabel("Ускорение, ×")
        ax.set_title("в) Ускорение vs scipy")

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_selected_bars.png"
        fig.savefig(out); plt.close(fig)
        print(f"\nСохранён: {out}")

    # ── Рисунок 2: grouped bar по триалам ────────────────────
    with plt.rc_context(GOST_RC):
        tnames_sci = [x[6].replace("DM_","").replace("_new","") for x in results["scipy"]]
        tnames_ca2 = [x[6].replace("DM_","").replace("_new","") for x in results["ca2"]]
        tnames_ca3 = [x[6].replace("DM_","").replace("_new","") for x in results["ca3"]]
        all_names  = tnames_sci + tnames_ca2 + tnames_ca3
        all_rmse   = ([x[0] for x in results["scipy"]] +
                      [x[0] for x in results["ca2"]] +
                      [x[0] for x in results["ca3"]])
        all_colors = [COLORS[0]]*3 + [COLORS[1]]*3 + [COLORS[2]]*3

        fig, ax = plt.subplots(figsize=(160*MM, 72*MM))
        x = np.arange(9)
        bars = ax.bar(x, all_rmse, width=0.6, color=all_colors, alpha=0.85, zorder=3)
        for bar, v in zip(bars, all_rmse):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.04,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("RMSE, дол. м.т.")
        ax.set_title("RMSE по триалам")
        ax.axvline(2.5, color="grey", linestyle=":", linewidth=1)
        ax.axvline(5.5, color="grey", linestyle=":", linewidth=1)
        ax.text(1,   ax.get_ylim()[1]*0.92, "scipy ∑a²\n(худшие)",  ha="center", fontsize=9, color=COLORS[0])
        ax.text(4,   ax.get_ylim()[1]*0.92, "CasADi ∑a²\n(худшие)", ha="center", fontsize=9, color=COLORS[1])
        ax.text(7,   ax.get_ylim()[1]*0.92, "CasADi ∑a³\n(лучшие)", ha="center", fontsize=9, color=COLORS[2])

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_selected_trials.png"
        fig.savefig(out); plt.close(fig)
        print(f"Сохранён: {out}")

    # ── Рисунок 3: кривые лучшего ∑a³ + лучший scipy ─────────
    with plt.rc_context(GOST_RC):
        best_ca3 = min(results["ca3"],  key=lambda x: x[0])
        worst_sci = max(results["scipy"], key=lambda x: x[0])

        fig, axes = plt.subplots(1, 2, figsize=(160*MM, 75*MM))

        for ax, entry, lbl, c, title in [
            (axes[0], best_ca3,  "CasADi ∑a³", COLORS[2], f"а) {best_ca3[6].replace('DM_','').replace('_new','')}"),
            (axes[1], worst_sci, "scipy ∑a²",   COLORS[0], f"б) {worst_sci[6].replace('DM_','').replace('_new','')}"),
        ]:
            r, ms, el, kf, ref, n, name = entry
            t = np.linspace(0, 100, n)
            v = ~np.isnan(ref)
            ax.plot(t, kf, color=c, linewidth=2.0, label=f"{lbl}  RMSE={r:.3f}")
            if v.any():
                ax.plot(t[v], ref[v], color="black", linestyle="--",
                        linewidth=1.8, label="eTibia (эталон)")
            ax.set_xlabel("Фаза цикла, %"); ax.set_ylabel("Контактная сила, дол. м.т.")
            ax.set_xlim(0, 100); ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
            ax.set_title(title)
            ax.legend(fontsize=9, loc="upper right")

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_selected_curves.png"
        fig.savefig(out); plt.close(fig)
        print(f"Сохранён: {out}")


if __name__ == "__main__":
    main()
