
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

from static_optimization import (
    DifferentiableStaticOptimizer, ModelParams,
    OptimizationConfig, CostFunction,
)

PROC_DIR  = Path("processed")
RESERVE_W = 2000.0
BODY_WT   = 64.0 * 9.81

TRIALS = [
    "DM_bouncy1_new", "DM_bouncy2_new",
    "DM_crouch_og1_new", "DM_crouch_og2_new", "DM_crouch_og3_new",
    "DM_ngait_og2_new",  "DM_ngait_og4_new",
    "DM_smooth1_new",    "DM_smooth3_new",
]

LABELS = ["scipy\n∑a²", "CasADi\n∑a²", "CasADi\n∑a³"]
COLORS = ["#e07b54", "#5b8db8", "#2e7d32"]

def scipy_frame(R, F0, tau, w):
    n_m, n_d = len(F0), len(tau)
    n = n_m + n_d
    def cost(x): return np.sum(x[:n_m] ** 2)
    def grad(x): g = np.zeros(n); g[:n_m] = 2 * x[:n_m]; return g
    def eq(x):   return R @ (F0 * x[:n_m]) + x[n_m:] - tau
    def eq_jac(x):
        J = np.zeros((n_d, n)); J[:, :n_m] = R * F0; J[:, n_m:] = np.eye(n_d); return J
    res = minimize(cost, np.zeros(n), jac=grad,
                   constraints=[{"type": "eq", "fun": eq, "jac": eq_jac}],
                   bounds=[(0, 1)] * n_m + [(-w, w)] * n_d,
                   method="SLSQP",
                   options={"ftol": 1e-4, "maxiter": 300, "disp": False})
    return res.x[:n_m]

def run_scipy(R, F0, tau):
    t0 = time.perf_counter()
    acts = np.array([scipy_frame(R[t], F0, tau[t], RESERVE_W)
                     for t in range(len(tau))])
    return acts, time.perf_counter() - t0

def run_casadi(R, F0, tau, cost_fn):
    n_m, n_d = len(F0), tau.shape[1]
    model = ModelParams(n_muscles=n_m, n_dof=n_d,
                        max_isometric_force=F0, moment_arms=R)
    cfg = OptimizationConfig(cost_function=cost_fn,
                             reserve_actuator_weight=RESERVE_W,
                             verbose=False)
    opt = DifferentiableStaticOptimizer(model, cfg)
    res = opt.solve_trajectory(R, F0, tau)
    return res["activations"], res["total_time"], res["mean_frame_time"] * 1000

def knee_force_bw(acts, F0, km, grf):
    return (grf + np.sum(acts * F0[np.newaxis, :] * km[np.newaxis, :], axis=1)) / BODY_WT

def rmse(pred, ref):
    v = ~np.isnan(ref)
    return float(np.sqrt(np.mean((pred[v] - ref[v]) ** 2))) if v.any() else None

def main():
    npz_files = [PROC_DIR / f"{t}_optdata.npz" for t in TRIALS
                 if (PROC_DIR / f"{t}_optdata.npz").exists()]
    if not npz_files:
        print("[!] Нет *_optdata.npz — запустите prepare_data.py")
        return

    print(f"Бенчмарк на {len(npz_files)} триалах\n")

    ms_all   = [[] for _ in LABELS]
    rmse_all = [[] for _ in LABELS]
    time_all = [[] for _ in LABELS]

    best = {}   # лучший триал по RMSE ∑a³ для кривой

    for npz in npz_files:
        d = np.load(npz, allow_pickle=True)
        R, F0, tau = d["moment_arms"], d["max_forces"], d["id_torques"]
        km, grf, ref = d["knee_mask"], d["grf_vertical"], d["ref_bw"]
        n   = len(tau)
        name = npz.stem.replace("_optdata", "")
        print(f"  {name}  ({n} кадров)", flush=True)

        a0, t0     = run_scipy(R, F0, tau)
        a1, t1, m1 = run_casadi(R, F0, tau, CostFunction.SUM_SQUARES)
        a2, t2, m2 = run_casadi(R, F0, tau, CostFunction.SUM_CUBES)

        kf = [knee_force_bw(a, F0, km, grf) for a in (a0, a1, a2)]
        rs = [rmse(k, ref) for k in kf]
        ms = [t0 / n * 1000, m1, m2]
        ts = [t0, t1, t2]

        for i in range(3):
            ms_all[i].append(ms[i])
            time_all[i].append(ts[i])
            if rs[i] is not None:
                rmse_all[i].append(rs[i])

        r2 = rs[2]
        if r2 is not None and (not best or r2 < best["r2"]):
            best = dict(name=name, r2=r2, ref=ref,
                        kf0=kf[0], kf1=kf[1], kf2=kf[2],
                        acts0=a0, acts1=a1, acts2=a2,
                        n=n, valid=~np.isnan(ref))

        r_str = [f"{r:.3f}" if r else "n/a" for r in rs]
        print(f"    scipy={ms[0]:.1f}ms r={r_str[0]}  "
              f"ca2={ms[1]:.1f}ms r={r_str[1]}  "
              f"ca3={ms[2]:.1f}ms r={r_str[2]}")

    print("\n" + "=" * 62)
    print(f"{'Вариант':<24} {'мс/кадр':>9} {'RMSE BW':>9} {'Ускор.':>8}")
    print("-" * 62)
    base_ms = np.mean(ms_all[0])
    names_tbl = ["scipy SLSQP  ∑a²", "CasADi IPOPT ∑a²", "CasADi IPOPT ∑a³"]
    for i, lbl in enumerate(names_tbl):
        ms  = np.mean(ms_all[i])
        rm  = np.mean(rmse_all[i]) if rmse_all[i] else float("nan")
        spd = base_ms / ms
        print(f"{lbl:<24} {ms:>9.2f} {rm:>9.3f} {spd:>7.1f}×")
    print("=" * 62)

    _make_figure(ms_all, rmse_all, time_all, best, npz_files)

def _make_figure(ms_all, rmse_all, time_all, best, npz_files):
    n_trials = len(npz_files)
    trial_names = [p.stem.replace("_optdata", "").replace("DM_", "") for p in npz_files]
    base_ms = np.mean(ms_all[0])

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Сравнение методов статической оптимизации\n"
        "Grand Challenge (субъект DM, 80 мышц, 12 DOF)",
        fontsize=13, fontweight="bold", y=0.98
    )

    ax1 = fig.add_subplot(3, 3, 1)
    x  = np.arange(n_trials)
    w  = 0.26
    for i in range(3):
        vals = rmse_all[i] if len(rmse_all[i]) == n_trials else [np.nan] * n_trials
        ax1.bar(x + (i - 1) * w, vals, w, color=COLORS[i], alpha=0.85,
                label=LABELS[i].replace("\n", " "))
    ax1.set_xticks(x)
    ax1.set_xticklabels(trial_names, rotation=40, ha="right", fontsize=7)
    ax1.set_ylabel("RMSE [BW]")
    ax1.set_title("RMSE по триалам")
    ax1.legend(fontsize=7, loc="upper right")

    ax2 = fig.add_subplot(3, 3, 2)
    means = [np.mean(r) if r else 0 for r in rmse_all]
    stds  = [np.std(r)  if r else 0 for r in rmse_all]
    bars  = ax2.bar(range(3), means, color=COLORS, alpha=0.85, width=0.5,
                    yerr=stds, capsize=5, error_kw=dict(elinewidth=1.5))
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(LABELS, fontsize=9)
    ax2.set_ylabel("RMSE [BW]")
    ax2.set_title("Среднее RMSE ± std")
    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 m + s + 0.02, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax3 = fig.add_subplot(3, 3, 3)
    ms_means = [np.mean(m) for m in ms_all]
    ms_stds  = [np.std(m)  for m in ms_all]
    bars3 = ax3.bar(range(3), ms_means, color=COLORS, alpha=0.85, width=0.5,
                    yerr=ms_stds, capsize=5, error_kw=dict(elinewidth=1.5))
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(LABELS, fontsize=9)
    ax3.set_ylabel("мс / кадр")
    ax3.set_title("Скорость расчёта")
    for bar, m in zip(bars3, ms_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, m + 1,
                 f"{m:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax4 = fig.add_subplot(3, 3, 4)
    speedups = [base_ms / np.mean(m) for m in ms_all]
    bars4 = ax4.bar(range(3), speedups, color=COLORS, alpha=0.85, width=0.5)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(LABELS, fontsize=9)
    ax4.set_ylabel("Ускорение (×)")
    ax4.set_title("Ускорение vs scipy baseline")
    ax4.axhline(1, color="grey", linestyle="--", linewidth=1)
    for bar, v in zip(bars4, speedups):
        ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                 f"×{v:.1f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax5 = fig.add_subplot(3, 3, 5)
    for i, (lbl, c) in enumerate(zip(LABELS, COLORS)):
        n = min(len(ms_all[i]), len(rmse_all[i]))
        ax5.scatter(ms_all[i][:n], rmse_all[i][:n],
                    color=c, s=50, alpha=0.75, label=lbl.replace("\n", " "),
                    edgecolors="white", linewidths=0.5)
    ax5.set_xlabel("мс / кадр")
    ax5.set_ylabel("RMSE [BW]")
    ax5.set_title("Скорость–Точность (по триалам)")
    ax5.legend(fontsize=7)
    ax5.set_xscale("log")

    ax6 = fig.add_subplot(3, 3, 6)
    if best:
        t = np.linspace(0, 100, best["n"])
        ax6.plot(t, best["kf0"], color=COLORS[0], lw=1.5, alpha=0.75,
                 label=f"scipy ∑a²  (RMSE={rmse(best['kf0'], best['ref']):.2f})")
        ax6.plot(t, best["kf1"], color=COLORS[1], lw=1.5, alpha=0.75,
                 label=f"CasADi ∑a² (RMSE={rmse(best['kf1'], best['ref']):.2f})")
        ax6.plot(t, best["kf2"], color=COLORS[2], lw=2.0,
                 label=f"CasADi ∑a³ (RMSE={best['r2']:.2f})")
        v = best["valid"]
        if v.any():
            ax6.plot(t[v], best["ref"][v], "k--", lw=2, label="eTibia (эталон)")
        ax6.set_xlabel("% цикла")
        ax6.set_ylabel("Сила [BW]")
        ax6.set_title(f"Контактная сила колена\n{best['name']}")
        ax6.legend(fontsize=7)
        ax6.set_ylim(bottom=0)

    ax7 = fig.add_subplot(3, 3, 7)
    bp = ax7.boxplot([r if r else [0] for r in rmse_all],
                     patch_artist=True, widths=0.45,
                     medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], COLORS):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax7.set_xticklabels(LABELS, fontsize=9)
    ax7.set_ylabel("RMSE [BW]")
    ax7.set_title("Распределение RMSE (box)")

    ax8 = fig.add_subplot(3, 3, 8)
    if best:
        for i, (acts, lbl, c) in enumerate(zip(
                [best["acts0"], best["acts1"], best["acts2"]], LABELS, COLORS)):
            mean_a = acts.mean(axis=1)
            t = np.linspace(0, 100, len(mean_a))
            ax8.plot(t, mean_a, color=c, lw=1.5, alpha=0.85,
                     label=lbl.replace("\n", " "))
        ax8.set_xlabel("% цикла")
        ax8.set_ylabel("Среднее по мышцам")
        ax8.set_title(f"Активации мышц\n{best['name']}")
        ax8.legend(fontsize=7)

    ax9 = fig.add_subplot(3, 3, 9)
    t_means = [np.mean(t) for t in time_all]
    t_stds  = [np.std(t)  for t in time_all]
    bars9 = ax9.bar(range(3), t_means, color=COLORS, alpha=0.85, width=0.5,
                    yerr=t_stds, capsize=5, error_kw=dict(elinewidth=1.5))
    ax9.set_xticks(range(3))
    ax9.set_xticklabels(LABELS, fontsize=9)
    ax9.set_ylabel("Секунды")
    ax9.set_title("Суммарное время на триал")
    ax9.axhline(20, color="red", linestyle="--", lw=1.5, label="≤ 20 с")
    ax9.legend(fontsize=8)
    for bar, m in zip(bars9, t_means):
        ax9.text(bar.get_x() + bar.get_width() / 2, m + 0.2,
                 f"{m:.1f}с", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = PROC_DIR / "benchmark_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nГрафик сохранён: {out}")
    plt.close()

if __name__ == "__main__":
    main()
