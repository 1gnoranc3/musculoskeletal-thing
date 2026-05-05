"""
Графики по ГОСТ 7.32 для ВКР.

Запуск: python benchmark_gost.py
Выход:  processed/fig_speed_rmse.png
        processed/fig_knee_force.png
        processed/fig_boxplot.png
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

# ─────────────────────────────────────────────────────────────
# ГОСТ-настройки matplotlib
# ─────────────────────────────────────────────────────────────

GOST_RC = {
    "font.family":        "Times New Roman",
    "font.size":          12,
    "axes.titlesize":     12,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.dpi":         300,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "lines.linewidth":    1.5,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.5,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.facecolor":   "white",
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
}

# Размеры для А4: текстовое поле 160×240 мм
MM = 1 / 25.4   # мм → дюймы

PROC_DIR  = Path("processed")
RESERVE_W = 2000.0
BODY_WT   = 64.0 * 9.81
TRIALS = [
    "DM_bouncy1_new", "DM_bouncy2_new",
    "DM_crouch_og1_new", "DM_crouch_og2_new", "DM_crouch_og3_new",
    "DM_ngait_og2_new",  "DM_ngait_og4_new",
    "DM_smooth1_new",    "DM_smooth3_new",
]
# Основное сравнение: scipy + ∑a²  vs  предлагаемый алгоритм (CasADi + ∑a²)
# ∑a³ включён как «альтернативная целевая функция»
COLORS       = ["#C0392B", "#27AE60", "#2980B9"]
LABELS       = ["scipy SLSQP ∑a²\n(базовый метод)",
                "CasADi IPOPT ∑a²\n(предлагаемый)",
                "CasADi IPOPT ∑a³\n(альтернативный)"]
LABELS_SHORT = ["scipy\n∑a²", "CasADi\n∑a²\n(предл.)", "CasADi\n∑a³\n(альт.)"]


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
    res = minimize(cost, np.zeros(n), jac=grad,
                   constraints=[{"type": "eq", "fun": eq, "jac": ejac}],
                   bounds=[(0,1)]*n_m + [(-w,w)]*n_d, method="SLSQP",
                   options={"ftol": 1e-4, "maxiter": 300, "disp": False})
    return res.x[:n_m]

def run_scipy(R, F0, tau):
    t0 = time.perf_counter()
    acts = np.array([_scipy_frame(R[t], F0, tau[t], RESERVE_W) for t in range(len(tau))])
    return acts, time.perf_counter() - t0

def run_casadi(R, F0, tau, cfn):
    model = ModelParams(n_muscles=len(F0), n_dof=tau.shape[1],
                        max_isometric_force=F0, moment_arms=R)
    cfg = OptimizationConfig(cost_function=cfn, reserve_actuator_weight=RESERVE_W, verbose=False)
    res = DifferentiableStaticOptimizer(model, cfg).solve_trajectory(R, F0, tau)
    return res["activations"], res["total_time"], res["mean_frame_time"] * 1000

def kf_bw(acts, F0, km, grf):
    return (grf + np.sum(acts * F0[np.newaxis, :] * km[np.newaxis, :], axis=1)) / BODY_WT

def rmse_fn(pred, ref):
    v = ~np.isnan(ref)
    return float(np.sqrt(np.mean((pred[v] - ref[v]) ** 2))) if v.any() else None


# ─────────────────────────────────────────────────────────────
# Сбор данных
# ─────────────────────────────────────────────────────────────

def collect():
    ms_all   = [[] for _ in LABELS]
    rmse_all = [[] for _ in LABELS]
    time_all = [[] for _ in LABELS]
    best     = {}
    trial_names = []

    for trial in TRIALS:
        npz = PROC_DIR / f"{trial}_optdata.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        R, F0, tau = d["moment_arms"], d["max_forces"], d["id_torques"]
        km, grf, ref = d["knee_mask"], d["grf_vertical"], d["ref_bw"]
        n = len(tau)
        print(f"  {trial}  ({n} кадров)", flush=True)

        a0, t0     = run_scipy(R, F0, tau)
        a1, t1, m1 = run_casadi(R, F0, tau, CostFunction.SUM_SQUARES)
        a2, t2, m2 = run_casadi(R, F0, tau, CostFunction.SUM_CUBES)

        kfs = [kf_bw(a, F0, km, grf) for a in (a0, a1, a2)]
        rs  = [rmse_fn(k, ref) for k in kfs]
        mss = [t0 / n * 1000, m1, m2]
        ts  = [t0, t1, t2]

        for i in range(3):
            ms_all[i].append(mss[i])
            time_all[i].append(ts[i])
            if rs[i] is not None:
                rmse_all[i].append(rs[i])

        tname = trial.replace("DM_", "").replace("_new", "")
        trial_names.append(tname)

        r2 = rs[2]
        if r2 is not None and (not best or r2 < best["r2"]):
            best = dict(name=tname, r2=r2, ref=ref,
                        kfs=kfs, acts=[a0, a1, a2], n=n,
                        valid=~np.isnan(ref))

    return ms_all, rmse_all, time_all, best, trial_names


# ─────────────────────────────────────────────────────────────
# Рисунок 1 — Скорость и точность (3 столбчатых + scatter)
# ─────────────────────────────────────────────────────────────

def fig_speed_rmse(ms_all, rmse_all, time_all):
    with plt.rc_context(GOST_RC):
        fig, axes = plt.subplots(1, 3, figsize=(160*MM, 70*MM))

        means_ms  = [np.mean(m) for m in ms_all]
        stds_ms   = [np.std(m)  for m in ms_all]
        means_rm  = [np.mean(r) if r else 0 for r in rmse_all]
        stds_rm   = [np.std(r)  if r else 0 for r in rmse_all]
        speedups  = [means_ms[0] / m for m in means_ms]

        ekw = dict(elinewidth=1.0, capsize=3, capthick=1.0, ecolor="black")

        # --- а) RMSE ---
        ax = axes[0]
        bars = ax.bar(range(3), means_rm, width=0.55, color=COLORS, alpha=0.85,
                      yerr=stds_rm, error_kw=ekw, zorder=3)
        for bar, h in zip(bars, [m+s+0.02 for m,s in zip(means_rm, stds_rm)]):
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS_SHORT, fontsize=10)
        ax.set_ylabel("RMSE, дол. м.т.")
        ax.set_title("а) Точность")
        ax.set_ylim(0, max(means_rm) * 1.35)

        # --- б) Скорость ---
        ax = axes[1]
        bars = ax.bar(range(3), means_ms, width=0.55, color=COLORS, alpha=0.85,
                      yerr=stds_ms, error_kw=ekw, zorder=3)
        for bar, h in zip(bars, [m+s+0.5 for m,s in zip(means_ms, stds_ms)]):
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS_SHORT, fontsize=10)
        ax.set_ylabel("Время, мс/кадр")
        ax.set_title("б) Скорость")

        # --- в) Ускорение ---
        ax = axes[2]
        bars = ax.bar(range(3), speedups, width=0.55, color=COLORS, alpha=0.85, zorder=3)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        for bar, v in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"×{v:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS_SHORT, fontsize=10)
        ax.set_ylabel("Ускорение, ×")
        ax.set_title("в) Ускорение vs scipy")

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_speed_rmse.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Сохранён: {out}")


# ─────────────────────────────────────────────────────────────
# Рисунок 2 — Кривые контактной силы колена
# ─────────────────────────────────────────────────────────────

def fig_knee_force(best):
    if not best:
        return
    with plt.rc_context(GOST_RC):
        fig, ax = plt.subplots(figsize=(160*MM, 80*MM))

        t   = np.linspace(0, 100, best["n"])
        v   = best["valid"]
        ref = best["ref"]

        ls_list = ["-", "--", ":"]
        for kf, lbl, c, ls, r in zip(
                best["kfs"], LABELS, COLORS, ls_list,
                [rmse_fn(kf, ref) for kf in best["kfs"]]):
            r_str = f"{r:.3f}" if r else "—"
            ax.plot(t, kf, color=c, linestyle=ls, linewidth=1.8,
                    label=f"{lbl}  (RMSE = {r_str} дол. м.т.)")

        if v.any():
            ax.plot(t[v], ref[v], color="black", linestyle="-",
                    linewidth=2.0, label="eTibia (эталонное значение)")

        ax.set_xlabel("Фаза цикла, %")
        ax.set_ylabel("Контактная сила, дол. м.т.")
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.legend(loc="upper right", framealpha=0.9)

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_knee_force.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Сохранён: {out}")


# ─────────────────────────────────────────────────────────────
# Рисунок 3 — Box-plot RMSE по триалам
# ─────────────────────────────────────────────────────────────

def fig_boxplot(rmse_all, trial_names):
    with plt.rc_context(GOST_RC):
        fig, axes = plt.subplots(1, 2, figsize=(160*MM, 75*MM))

        # --- а) Box-plot по методам ---
        ax = axes[0]
        bp = ax.boxplot(rmse_all, patch_artist=True, widths=0.45,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=4, alpha=0.5))
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_xticklabels(LABELS_SHORT, fontsize=10)
        ax.set_ylabel("RMSE, дол. м.т.")
        ax.set_title("а) Разброс RMSE по триалам")

        # --- б) Grouped bar по триалам ---
        ax = axes[1]
        n  = len(trial_names)
        x  = np.arange(n)
        w  = 0.28
        for i in range(3):
            vals = rmse_all[i] if len(rmse_all[i]) == n else [np.nan] * n
            ax.bar(x + (i-1)*w, vals, w, color=COLORS[i],
                   alpha=0.82, label=LABELS[i])
        ax.set_xticks(x)
        ax.set_xticklabels(trial_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMSE, дол. м.т.")
        ax.set_title("б) RMSE по триалам")
        ax.legend(fontsize=8, loc="upper right")

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_boxplot.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Сохранён: {out}")


# ─────────────────────────────────────────────────────────────
# Рисунок 4 — Активации мышц
# ─────────────────────────────────────────────────────────────

def fig_activations(best):
    if not best:
        return
    with plt.rc_context(GOST_RC):
        fig, axes = plt.subplots(1, 2, figsize=(160*MM, 70*MM))

        t = np.linspace(0, 100, best["n"])

        # --- а) Средняя активация по кадру ---
        ax = axes[0]
        ls_list = ["-", "--", ":"]
        for acts, lbl, c, ls in zip(best["acts"], LABELS, COLORS, ls_list):
            ax.plot(t, acts.mean(axis=1), color=c, linestyle=ls,
                    linewidth=1.8, label=lbl)
        ax.set_xlabel("Фаза цикла, %")
        ax.set_ylabel("Средняя активация")
        ax.set_xlim(0, 100)
        ax.set_ylim(0)
        ax.set_title("а) Средняя активация мышц")
        ax.legend(fontsize=9)

        # --- б) Макс. активация по кадру ---
        ax = axes[1]
        for acts, lbl, c, ls in zip(best["acts"], LABELS, COLORS, ls_list):
            ax.plot(t, acts.max(axis=1), color=c, linestyle=ls,
                    linewidth=1.8, label=lbl)
        ax.set_xlabel("Фаза цикла, %")
        ax.set_ylabel("Макс. активация")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        ax.set_title("б) Максимальная активация мышц")
        ax.legend(fontsize=9)

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_activations.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Сохранён: {out}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    print("Сбор данных...\n")
    ms_all, rmse_all, time_all, best, trial_names = collect()

    base = np.mean(ms_all[0])
    print(f"\n{'Метод':<28} {'мс/кадр':>9} {'RMSE':>8} {'Ускор.':>8}")
    print("-" * 58)
    rows = ["scipy SLSQP ∑a²  (базовый)",
            "CasADi IPOPT ∑a² (предлагаемый)",
            "CasADi IPOPT ∑a³ (альтернативный)"]
    for i, lbl in enumerate(rows):
        ms = np.mean(ms_all[i]); rm = np.mean(rmse_all[i]) if rmse_all[i] else 0
        print(f"{lbl:<28} {ms:>9.2f} {rm:>8.3f} {base/ms:>7.1f}×")

    print("\nГенерация графиков ГОСТ...")
    fig_speed_rmse(ms_all, rmse_all, time_all)
    fig_knee_force(best)
    fig_boxplot(rmse_all, trial_names)
    fig_activations(best)
    print("\nВсе графики сохранены в processed/")


if __name__ == "__main__":
    main()
