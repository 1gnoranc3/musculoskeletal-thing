"""
Финальные публикационные графики по ГОСТ 7.32 для ВКР.
Запуск: python make_final_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.optimize import minimize
import time as time_module

from static_optimization import (
    DifferentiableStaticOptimizer, ModelParams,
    OptimizationConfig, CostFunction,
)

PROC_DIR = Path("processed")
BW       = 64.0 * 9.81
W        = 2000.0
MM       = 1 / 25.4

GOST = {
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
    "grid.alpha":        0.45,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "figure.facecolor":  "white",
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
}

C_SCI  = "#C0392B"   # красный   — scipy
C_CA2  = "#27AE60"   # зелёный   — CasADi ∑a² (предлагаемый)
C_CA3  = "#2980B9"   # синий     — CasADi ∑a³
C_REF  = "#2C3E50"   # почти чёрный — референс


# ─────────────────────────────────────────────────────────────
# Солверы
# ─────────────────────────────────────────────────────────────

def run_scipy(R, F0, tau):
    def frame(R, F0, tau):
        n_m, n_d = len(F0), len(tau); n = n_m + n_d
        res = minimize(
            lambda x: np.sum(x[:n_m]**2),
            np.zeros(n),
            jac=lambda x: np.r_[2*x[:n_m], np.zeros(n_d)],
            constraints=[{"type": "eq",
                          "fun":  lambda x: R@(F0*x[:n_m])+x[n_m:]-tau,
                          "jac":  lambda x: np.c_[R*F0, np.eye(n_d)]}],
            bounds=[(0,1)]*n_m + [(-W,W)]*n_d,
            method="SLSQP",
            options={"ftol": 1e-4, "maxiter": 300, "disp": False})
        return res.x[:n_m]
    t0 = time_module.perf_counter()
    acts = np.array([frame(R[i], F0, tau[i]) for i in range(len(tau))])
    return acts, (time_module.perf_counter()-t0)/len(tau)*1000

def run_ca(R, F0, tau, t, cfn, lam=0.0, dyn=False):
    model = ModelParams(len(F0), tau.shape[1], F0, R)
    cfg = OptimizationConfig(
        cost_function=cfn, reserve_actuator_weight=W, verbose=False,
        activation_dynamics=dyn, tau_act=0.01, tau_deact=0.04,
        temporal_smoothing=lam)
    res = DifferentiableStaticOptimizer(model, cfg).solve_trajectory(R, F0, tau, t)
    return res["activations"], res["mean_frame_time"]*1000

def kf(a, F0, km, grf):
    return (grf + np.sum(a * F0[np.newaxis,:] * km[np.newaxis,:], axis=1)) / BW

def rmse(p, ref):
    v = ~np.isnan(ref)
    return float(np.sqrt(np.mean((p[v]-ref[v])**2)))


# ─────────────────────────────────────────────────────────────
# Данные для «кривого» триала
# ─────────────────────────────────────────────────────────────

def load_best():
    d = np.load(PROC_DIR / "DM_crouch_og2_new_optdata.npz", allow_pickle=True)
    R, F0, tau = d["moment_arms"], d["max_forces"], d["id_torques"]
    km, grf, ref, t = d["knee_mask"], d["grf_vertical"], d["ref_bw"], d["time"]

    a_sp,  ms_sp  = run_scipy(R, F0, tau)
    a_ca2, ms_ca2 = run_ca(R, F0, tau, t, CostFunction.SUM_SQUARES)
    a_ca2d,ms_ca2d= run_ca(R, F0, tau, t, CostFunction.SUM_SQUARES, dyn=True)
    a_ca3, ms_ca3 = run_ca(R, F0, tau, t, CostFunction.SUM_CUBES)

    curves = {
        "scipy":   (kf(a_sp,   F0, km, grf), ms_sp,   a_sp),
        "ca2":     (kf(a_ca2,  F0, km, grf), ms_ca2,  a_ca2),
        "ca2d":    (kf(a_ca2d, F0, km, grf), ms_ca2d, a_ca2d),
        "ca3":     (kf(a_ca3,  F0, km, grf), ms_ca3,  a_ca3),
    }
    return curves, ref, t, a_sp, a_ca2, a_ca2d, F0, km

# ─────────────────────────────────────────────────────────────
# Данные по всем 9 триалам (для сводных баров)
# ─────────────────────────────────────────────────────────────

TRIALS = ["DM_bouncy1_new","DM_bouncy2_new","DM_crouch_og1_new","DM_crouch_og2_new",
          "DM_crouch_og3_new","DM_ngait_og2_new","DM_ngait_og4_new",
          "DM_smooth1_new","DM_smooth3_new"]

def load_all():
    ms_list  = [[], [], []]
    rm_list  = [[], [], []]
    for trial in TRIALS:
        d = np.load(PROC_DIR/f"{trial}_optdata.npz", allow_pickle=True)
        R,F0,tau,t = d["moment_arms"],d["max_forces"],d["id_torques"],d["time"]
        km,grf,ref = d["knee_mask"],d["grf_vertical"],d["ref_bw"]
        a0,ms0 = run_scipy(R,F0,tau)
        a1,ms1 = run_ca(R,F0,tau,t,CostFunction.SUM_SQUARES)
        a2,ms2 = run_ca(R,F0,tau,t,CostFunction.SUM_SQUARES,dyn=True)
        for i,(a,ms) in enumerate([(a0,ms0),(a1,ms1),(a2,ms2)]):
            ms_list[i].append(ms)
            r = rmse(kf(a,F0,km,grf),ref)
            if r: rm_list[i].append(r)
    return ms_list, rm_list


# ─────────────────────────────────────────────────────────────
# Рисунок 1 — Главный: кривые + zoom
# ─────────────────────────────────────────────────────────────

def fig_main_curves(curves, ref, t_arr):
    pct = np.linspace(0, 100, len(ref))

    with plt.rc_context(GOST):
        fig = plt.figure(figsize=(160*MM, 90*MM))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # ── Левая: все кривые ───────────────────────────────
        kf_sp  = curves["scipy"][0]
        kf_ca2 = curves["ca2"][0]
        kf_ca2d= curves["ca2d"][0]

        # scipy - показываем но обрезаем ось чтобы не уходил в бесконечность
        ax1.plot(pct, np.clip(kf_sp, 0, 2.5), color=C_SCI,
                 lw=1.5, ls="--", alpha=0.7,
                 label=f"scipy SLSQP ∑a²  (RMSE={rmse(kf_sp,ref):.2f})")
        ax1.plot(pct, kf_ca2,  color=C_CA2, lw=2.2,
                 label=f"CasADi ∑a²  (RMSE={rmse(kf_ca2,ref):.3f})")
        ax1.plot(pct, kf_ca2d, color=C_CA2, lw=1.5, ls=":",
                 label=f"CasADi ∑a² + ActDyn  (RMSE={rmse(kf_ca2d,ref):.3f})")
        ax1.plot(pct, ref, color=C_REF, lw=2.0, ls="-",
                 label="eTibia (эталон)")

        # Стрелка на выброс scipy
        peak_idx = np.argmax(kf_sp)
        if kf_sp[peak_idx] > 2.0:
            ax1.annotate(f"scipy: {kf_sp[peak_idx]:.1f} BW →",
                         xy=(pct[peak_idx], 2.45),
                         xytext=(pct[peak_idx]-18, 2.2),
                         fontsize=9, color=C_SCI,
                         arrowprops=dict(arrowstyle="->", color=C_SCI, lw=1))

        ax1.set_xlabel("Фаза движения, %")
        ax1.set_ylabel("Контактная сила, дол. м.т.")
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 2.6)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax1.legend(fontsize=8, loc="upper left",
                   framealpha=0.9, edgecolor="lightgrey")
        ax1.set_title("а) Контактная сила коленного сустава")

        # ── Правая: zoom на CasADi vs референс ─────────────
        ax2.fill_between(pct, ref - 0.05, ref + 0.05,
                         color=C_REF, alpha=0.12, label="±0.05 BW")
        ax2.plot(pct, ref,    color=C_REF, lw=2.0, label="eTibia (эталон)")
        ax2.plot(pct, kf_ca2, color=C_CA2, lw=2.0, ls="--",
                 label=f"CasADi ∑a²\nRMSE={rmse(kf_ca2,ref):.3f} BW")

        ax2.set_xlabel("Фаза движения, %")
        ax2.set_ylabel("Контактная сила, дол. м.т.")
        ax2.set_xlim(0, 100)
        # Tight ylim to show closeness
        ylo = min(ref.min(), kf_ca2.min()) - 0.05
        yhi = max(ref.max(), kf_ca2.max()) + 0.05
        ax2.set_ylim(ylo, yhi)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(25))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax2.set_title("б) Увеличенный масштаб")

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_curves_main.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Сохранён: {out}")


# ─────────────────────────────────────────────────────────────
# Рисунок 2 — Сводные метрики (9 триалов)
# ─────────────────────────────────────────────────────────────

def fig_metrics(ms_list, rm_list):
    labels = ["scipy\n∑a²", "CasADi\n∑a²\n(предл.)", "CasADi ∑a²\n+ ActDyn"]
    colors = [C_SCI, C_CA2, "#1A8A4C"]
    ekw    = dict(elinewidth=1.0, capsize=3, ecolor="black")

    with plt.rc_context(GOST):
        fig, axes = plt.subplots(1, 3, figsize=(160*MM, 68*MM))

        # а) RMSE
        ax = axes[0]
        means = [np.mean(r) for r in rm_list]
        stds  = [np.std(r)  for r in rm_list]
        bars  = ax.bar(range(3), means, 0.55, color=colors, alpha=0.85,
                       yerr=stds, error_kw=ekw, zorder=3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.06,
                    f"{m:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("RMSE, дол. м.т.")
        ax.set_title("а) Точность")
        ax.set_ylim(0, max(means)*1.4)

        # б) Скорость
        ax = axes[1]
        ms_m = [np.mean(m) for m in ms_list]
        ms_s = [np.std(m)  for m in ms_list]
        bars = ax.bar(range(3), ms_m, 0.55, color=colors, alpha=0.85,
                      yerr=ms_s, error_kw=ekw, zorder=3)
        for bar, m, s in zip(bars, ms_m, ms_s):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.5,
                    f"{m:.1f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Время расчёта, мс/кадр")
        ax.set_title("б) Скорость")

        # в) Ускорение
        ax = axes[2]
        base = ms_m[0]
        spd  = [base/m for m in ms_m]
        bars = ax.bar(range(3), spd, 0.55, color=colors, alpha=0.85, zorder=3)
        ax.axhline(1, color="grey", ls="--", lw=0.8)
        for bar, v in zip(bars, spd):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.4,
                    f"×{v:.0f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Ускорение, ×")
        ax.set_title("в) Ускорение vs scipy")

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_metrics.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Сохранён: {out}")


# ─────────────────────────────────────────────────────────────
# Рисунок 3 — Активации мышц
# ─────────────────────────────────────────────────────────────

def fig_activations(a_sp, a_ca2, a_ca2d, F0, km):
    pct = np.linspace(0, 100, len(a_ca2))

    with plt.rc_context(GOST):
        fig, axes = plt.subplots(1, 2, figsize=(160*MM, 68*MM))

        # а) Средняя активация
        ax = axes[0]
        ax.plot(pct, np.clip(a_sp,  0,1).mean(axis=1), color=C_SCI, lw=1.5,
                ls="--", alpha=0.8, label="scipy ∑a²")
        ax.plot(pct, a_ca2.mean(axis=1),  color=C_CA2, lw=2.0,
                label="CasADi ∑a² (предл.)")
        ax.plot(pct, a_ca2d.mean(axis=1), color="#1A8A4C", lw=1.5, ls=":",
                label="CasADi ∑a² + ActDyn")
        ax.set_xlabel("Фаза движения, %"); ax.set_ylabel("Средняя активация")
        ax.set_xlim(0,100); ax.set_ylim(bottom=0)
        ax.set_title("а) Средняя активация мышц")
        ax.legend(fontsize=8)

        # б) Макс. активация
        ax = axes[1]
        ax.plot(pct, np.clip(a_sp,  0,1).max(axis=1), color=C_SCI, lw=1.5,
                ls="--", alpha=0.8, label="scipy ∑a²")
        ax.plot(pct, a_ca2.max(axis=1),  color=C_CA2, lw=2.0,
                label="CasADi ∑a² (предл.)")
        ax.plot(pct, a_ca2d.max(axis=1), color="#1A8A4C", lw=1.5, ls=":",
                label="CasADi ∑a² + ActDyn")
        ax.set_xlabel("Фаза движения, %"); ax.set_ylabel("Макс. активация")
        ax.set_xlim(0,100); ax.set_ylim(0, 1.05)
        ax.set_title("б) Максимальная активация мышц")
        ax.legend(fontsize=8)

        fig.tight_layout(pad=1.2)
        out = PROC_DIR / "fig_activations_final.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Сохранён: {out}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    print("Загрузка лучшего триала (DM_crouch_og2_new)...")
    curves, ref, t_arr, a_sp, a_ca2, a_ca2d, F0, km = load_best()

    print("\nРезультаты на лучшем триале:")
    print(f"  scipy  ∑a²:            RMSE={rmse(curves['scipy'][0],ref):.3f} BW  "
          f"{curves['scipy'][1]:.1f} мс/кадр")
    print(f"  CasADi ∑a²:            RMSE={rmse(curves['ca2'][0],ref):.3f} BW  "
          f"{curves['ca2'][1]:.1f} мс/кадр")
    print(f"  CasADi ∑a² + ActDyn:   RMSE={rmse(curves['ca2d'][0],ref):.3f} BW  "
          f"{curves['ca2d'][1]:.1f} мс/кадр")
    sp_ms = curves["scipy"][1]; ca_ms = curves["ca2"][1]
    print(f"  Ускорение: ×{sp_ms/ca_ms:.0f}")

    print("\nЗагрузка всех 9 триалов...")
    ms_list, rm_list = load_all()
    print(f"  scipy:           RMSE {np.mean(rm_list[0]):.3f} ± {np.std(rm_list[0]):.3f}  "
          f"мс {np.mean(ms_list[0]):.1f}")
    print(f"  CasADi ∑a²:      RMSE {np.mean(rm_list[1]):.3f} ± {np.std(rm_list[1]):.3f}  "
          f"мс {np.mean(ms_list[1]):.1f}")
    print(f"  CasADi + ActDyn: RMSE {np.mean(rm_list[2]):.3f} ± {np.std(rm_list[2]):.3f}  "
          f"мс {np.mean(ms_list[2]):.1f}")

    print("\nГенерация графиков...")
    fig_main_curves(curves, ref, t_arr)
    fig_metrics(ms_list, rm_list)
    fig_activations(a_sp, a_ca2, a_ca2d, F0, km)
    print("\nВсе графики сохранены в processed/")


if __name__ == "__main__":
    main()
