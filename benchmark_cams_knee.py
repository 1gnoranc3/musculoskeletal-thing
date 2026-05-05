"""
benchmark_cams_knee.py
─────────────────────────────────────────────────────────────────
Сравнение вариантов статической оптимизации на данных CAMS-Knee
(субъект K8L, тибиальный имплант, ходьба + присед).

Момент-плечи и ID-моменты берутся из Grand Challenge (DM, нормальная
ходьба) и масштабируются под K8L (масса 78.8 кг). Это стандартная
практика когда нет OpenSim IK/ID для нового субъекта — сравнение
вариантов целевой функции остаётся корректным.

Варианты:
  1. scipy SLSQP  ∑a²            — baseline
  2. CasADi IPOPT ∑a²            — autodiff
  3. CasADi IPOPT ∑a³ + резервы  — целевой
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

from static_optimization import (
    DifferentiableStaticOptimizer, ModelParams,
    OptimizationConfig, CostFunction,
)

# ─────────────────────────────────────────────────────────────
# Параметры
# ─────────────────────────────────────────────────────────────
PROC_DIR   = Path("processed")
CAMS_GAIT  = Path("CAMS_Knee_Sample/K8L/gait/export_proc/K8L_CAMS_Knee_gait_sample.csv")
CAMS_SQUAT = Path("CAMS_Knee_Sample/K8L/squat/export_proc/K8L_CAMS_Knee_squat_sample.csv")

BODY_MASS_K8L = 773.0 / 9.81   # кг (из CSV: Bodyweight = 773 Н)
BODY_WT_K8L   = 773.0           # Н
BODY_MASS_DM  = 64.0            # кг (Grand Challenge DM)
BODY_WT_DM    = 64.0 * 9.81     # Н
MASS_RATIO    = BODY_MASS_K8L / BODY_MASS_DM   # ~1.23

RESERVE_W = 2000.0

# Триалы нормальной ходьбы из GC — усредним момент-плечи и ID-моменты
GC_GAIT_TRIALS = [
    "DM_ngait_og1_new", "DM_ngait_og2_new", "DM_ngait_og3_new",
    "DM_ngait_og4_new", "DM_ngait_og5_new", "DM_ngait_og6_new",
    "DM_smooth1_new",   "DM_smooth3_new",   "DM_smooth4_new",
]


# ─────────────────────────────────────────────────────────────
# Загрузка и усреднение GC данных (момент-плечи + ID)
# ─────────────────────────────────────────────────────────────

def load_gc_template(n_target: int) -> dict:
    """
    Загрузить GC нормальные ходьбовые триалы, интерполировать все на
    n_target кадров (0–100 % цикла), усреднить.
    Масштабировать под K8L.
    """
    pct_target = np.linspace(0, 1, n_target)
    R_list, tau_list = [], []
    F0_ref, km_ref = None, None

    loaded = 0
    for name in GC_GAIT_TRIALS:
        p = PROC_DIR / f"{name}_optdata.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        n = d["id_torques"].shape[0]
        pct = np.linspace(0, 1, n)

        # Интерполяция момент-плеч (n_frames, n_dof, n_muscles) → n_target
        R_i = d["moment_arms"]
        R_new = np.stack([
            np.array([
                np.interp(pct_target, pct, R_i[:, dof, m])
                for m in range(R_i.shape[2])
            ]).T
            for dof in range(R_i.shape[1])
        ], axis=1)  # (n_target, n_dof, n_muscles)
        R_list.append(R_new)

        # Интерполяция ID-моментов (n_frames, n_dof) → n_target
        tau_i = d["id_torques"]
        tau_new = np.column_stack([
            np.interp(pct_target, pct, tau_i[:, dof])
            for dof in range(tau_i.shape[1])
        ])
        tau_list.append(tau_new)

        if F0_ref is None:
            F0_ref    = d["max_forces"]
            km_ref    = d["knee_mask"]
        loaded += 1

    if not loaded:
        raise RuntimeError("Нет GC optdata файлов в processed/")

    print(f"  Усреднено {loaded} GC триалов → {n_target} кадров")
    R_mean   = np.mean(R_list, axis=0)
    tau_mean = np.mean(tau_list, axis=0)

    # Масштабировать ID-моменты под K8L
    tau_scaled = tau_mean * MASS_RATIO

    return {
        "moment_arms":  R_mean,
        "id_torques":   tau_scaled,
        "max_forces":   F0_ref,
        "knee_mask":    km_ref,
    }


# ─────────────────────────────────────────────────────────────
# Парсинг CAMS-Knee CSV
# ─────────────────────────────────────────────────────────────

def parse_cams_csv(csv_path: Path):
    with open(csv_path) as f:
        lines = f.readlines()

    meta = {"bw_n": BODY_WT_K8L, "events": []}
    for line in lines[:22]:
        s = line.strip()
        if s.startswith("Bodyweight"):
            try:
                meta["bw_n"] = float(s.split(",")[1])
            except (ValueError, IndexError):
                pass
        elif "Foot Strike" in s or "Foot Off" in s:
            parts = s.split(",")
            try:
                meta["events"].append({"event": parts[0].strip(),
                                       "time":  float(parts[1])})
            except (ValueError, IndexError):
                pass

    df = pd.read_csv(csv_path, skiprows=23, header=0, low_memory=False)
    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return meta, df


def extract_gait_cycle(df: pd.DataFrame, events: list, bw: float):
    """Взять первый полный правый цикл (RFS → RFS)."""
    rhs = sorted(e["time"] for e in events if e["event"] == "Right Foot Strike")
    if len(rhs) < 2:
        return None

    t0, t1 = rhs[0], rhs[1]
    seg = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
    seg = seg[["time", "Fres"] + [f"Fz_lab_{j}" for j in range(1, 9)
                                   if f"Fz_lab_{j}" in df.columns]].dropna(subset=["time", "Fres"])
    if len(seg) < 10:
        return None

    grf_cols = [c for c in seg.columns if c.startswith("Fz_lab_")]
    grf_v    = seg[grf_cols].abs().sum(axis=1).values
    ref_bw   = seg["Fres"].values / bw
    time     = seg["time"].values

    return {"time": time, "grf_v": grf_v, "ref_bw": ref_bw,
            "n_frames": len(time), "t0": t0, "t1": t1}


def extract_squat(df: pd.DataFrame, bw: float):
    valid = df[["time", "Fres"]].dropna()
    if valid.empty:
        return None
    grf_cols = [c for c in df.columns if c.startswith("Fz_lab_")]
    grf_v    = df.loc[valid.index, grf_cols].abs().sum(axis=1).values if grf_cols else np.zeros(len(valid))
    ref_bw   = valid["Fres"].values / bw
    return {"time": valid["time"].values, "grf_v": grf_v,
            "ref_bw": ref_bw, "n_frames": len(valid)}


# ─────────────────────────────────────────────────────────────
# scipy baseline
# ─────────────────────────────────────────────────────────────

def scipy_frame(R, F0, tau, w_res):
    n_m, n_d = len(F0), len(tau)
    n = n_m + n_d

    def cost(x):  return np.sum(x[:n_m] ** 2)
    def grad(x):
        g = np.zeros(n); g[:n_m] = 2 * x[:n_m]; return g
    def eq(x):    return R @ (F0 * x[:n_m]) + x[n_m:] - tau
    def eq_jac(x):
        J = np.zeros((n_d, n))
        J[:, :n_m] = R * F0; J[:, n_m:] = np.eye(n_d)
        return J

    res = minimize(cost, np.zeros(n), jac=grad,
                   constraints=[{"type": "eq", "fun": eq, "jac": eq_jac}],
                   bounds=[(0, 1)] * n_m + [(-w_res, w_res)] * n_d,
                   method="SLSQP",
                   options={"ftol": 1e-4, "maxiter": 300, "disp": False})
    return res.x[:n_m]


def run_scipy(R, F0, tau):
    t0 = time.perf_counter()
    acts = np.array([scipy_frame(R[i], F0, tau[i], RESERVE_W)
                     for i in range(len(tau))])
    return acts, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────
# CasADi варианты
# ─────────────────────────────────────────────────────────────

def run_casadi(R, F0, tau, cost_fn, reserve):
    n_m, n_d = len(F0), tau.shape[1]
    model = ModelParams(n_muscles=n_m, n_dof=n_d,
                        max_isometric_force=F0, moment_arms=R)
    cfg = OptimizationConfig(cost_function=cost_fn,
                             reserve_actuator_weight=reserve,
                             verbose=False)
    opt = DifferentiableStaticOptimizer(model, cfg)
    res = opt.solve_trajectory(R, F0, tau)
    return res["activations"], res["total_time"], res["mean_frame_time"] * 1000


# ─────────────────────────────────────────────────────────────
# Расчёт контактной силы колена
# ─────────────────────────────────────────────────────────────

def knee_force_bw(acts, F0, knee_mask, grf_v, bw):
    mf   = acts * F0[np.newaxis, :]
    knee = np.sum(mf[:, knee_mask], axis=1)
    return (grf_v + knee) / bw


def rmse(pred, ref):
    valid = ~np.isnan(ref)
    if not valid.any():
        return None
    return float(np.sqrt(np.mean((pred[valid] - ref[valid]) ** 2)))


# ─────────────────────────────────────────────────────────────
# Прогон бенчмарка на одном датасете
# ─────────────────────────────────────────────────────────────

def run_benchmark(name: str, data: dict, template: dict):
    n = data["n_frames"]
    R   = template["moment_arms"]
    tau = template["id_torques"]
    F0  = template["max_forces"]
    km  = template["knee_mask"]
    grf = data["grf_v"]
    ref = data["ref_bw"]

    # Привести GRF к длине n (уже должно совпадать, но на всякий случай)
    if len(grf) != n:
        grf = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(grf)), grf)

    print(f"\n  [{name}]  {n} кадров  {R.shape[1]} DOF  {R.shape[2]} мышц")

    labels = ["scipy ∑a²", "CasADi ∑a²", "CasADi ∑a³+res"]
    times_ms, rmses, kf_curves = [], [], []

    # 1. scipy
    print("    ▸ scipy ...", end=" ", flush=True)
    acts0, t0 = run_scipy(R, F0, tau)
    kf0 = knee_force_bw(acts0, F0, km, grf, BODY_WT_K8L)
    r0  = rmse(kf0, ref)
    ms0 = t0 / n * 1000
    r0_str = f"{r0:.3f}" if r0 is not None else "n/a"
    print(f"{ms0:.1f} мс/кадр  RMSE={r0_str} BW")
    times_ms.append(ms0); rmses.append(r0); kf_curves.append(kf0)

    # 2. CasADi ∑a²
    print("    ▸ CasADi ∑a² ...", end=" ", flush=True)
    acts1, t1, ms1 = run_casadi(R, F0, tau, CostFunction.SUM_SQUARES, RESERVE_W)
    kf1 = knee_force_bw(acts1, F0, km, grf, BODY_WT_K8L)
    r1  = rmse(kf1, ref)
    r1_str = f"{r1:.3f}" if r1 is not None else "n/a"
    print(f"{ms1:.1f} мс/кадр  RMSE={r1_str} BW")
    times_ms.append(ms1); rmses.append(r1); kf_curves.append(kf1)

    # 3. CasADi ∑a³
    print("    ▸ CasADi ∑a³ ...", end=" ", flush=True)
    acts2, t2, ms2 = run_casadi(R, F0, tau, CostFunction.SUM_CUBES, RESERVE_W)
    kf2 = knee_force_bw(acts2, F0, km, grf, BODY_WT_K8L)
    r2  = rmse(kf2, ref)
    r2_str = f"{r2:.3f}" if r2 is not None else "n/a"
    print(f"{ms2:.1f} мс/кадр  RMSE={r2_str} BW")
    times_ms.append(ms2); rmses.append(r2); kf_curves.append(kf2)

    return {
        "name":     name,
        "labels":   labels,
        "times_ms": times_ms,
        "rmses":    rmses,
        "kf_curves": kf_curves,
        "ref_bw":   ref,
        "n":        n,
        "speedup":  [times_ms[0] / ms for ms in times_ms],
    }


# ─────────────────────────────────────────────────────────────
# Визуализация
# ─────────────────────────────────────────────────────────────

def make_figure(results: list[dict]):
    n_datasets = len(results)
    colors = ["#e07b54", "#5b8db8", "#2e7d32"]
    fig = plt.figure(figsize=(15, 5 * n_datasets))
    fig.suptitle(
        "Сравнение вариантов статической оптимизации — CAMS-Knee K8L\n"
        "(момент-плечи: среднее GC нормальная ходьба, масштаб ×{:.2f})".format(MASS_RATIO),
        fontsize=12, fontweight="bold"
    )

    cols = 4
    for row, res in enumerate(results):
        base = row * cols + 1
        pct  = np.linspace(0, 100, res["n"])
        labels_short = ["scipy\n∑a²", "CasADi\n∑a²", "CasADi\n∑a³"]

        # 1. Кривые контактной силы
        ax1 = fig.add_subplot(n_datasets, cols, base)
        for i, (kf, lbl, c) in enumerate(zip(res["kf_curves"], res["labels"], colors)):
            ax1.plot(pct, kf, color=c, linewidth=1.8 + 0.2 * i,
                     alpha=0.85, label=lbl)
        ax1.plot(pct, res["ref_bw"], "k--", linewidth=2, label="Имплант (эталон)")
        ax1.set_xlabel("% движения")
        ax1.set_ylabel("Контактная сила [BW]")
        ax1.set_title(f"{res['name']}: кривые силы колена")
        ax1.legend(fontsize=7)
        ax1.set_ylim(bottom=0)

        # 2. RMSE bar
        ax2 = fig.add_subplot(n_datasets, cols, base + 1)
        rmse_vals = [r if r is not None else 0 for r in res["rmses"]]
        bars = ax2.bar(range(3), rmse_vals, color=colors, alpha=0.85, width=0.55)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels_short, fontsize=8)
        ax2.set_ylabel("RMSE [BW]")
        ax2.set_title("Точность (RMSE)")
        for bar, v in zip(bars, rmse_vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # 3. Скорость мс/кадр
        ax3 = fig.add_subplot(n_datasets, cols, base + 2)
        bars3 = ax3.bar(range(3), res["times_ms"], color=colors, alpha=0.85, width=0.55)
        ax3.set_xticks(range(3))
        ax3.set_xticklabels(labels_short, fontsize=8)
        ax3.set_ylabel("мс / кадр")
        ax3.set_title("Скорость расчёта")
        for bar, v in zip(bars3, res["times_ms"]):
            ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        # 4. Ускорение
        ax4 = fig.add_subplot(n_datasets, cols, base + 3)
        bars4 = ax4.bar(range(3), res["speedup"], color=colors, alpha=0.85, width=0.55)
        ax4.set_xticks(range(3))
        ax4.set_xticklabels(labels_short, fontsize=8)
        ax4.set_ylabel("Ускорение (×)")
        ax4.set_title("Ускорение vs scipy")
        ax4.axhline(1, color="grey", linestyle="--", linewidth=1)
        for bar, v in zip(bars4, res["speedup"]):
            ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"×{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = PROC_DIR / "benchmark_cams_knee.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nГрафик сохранён: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Сводная таблица
# ─────────────────────────────────────────────────────────────

def print_table(results: list[dict]):
    print("\n" + "=" * 70)
    print(f"{'Датасет / Вариант':<30} {'мс/кадр':>9} {'RMSE BW':>9} {'Ускор.':>8}")
    print("-" * 70)
    for res in results:
        print(f"  {res['name']}")
        for lbl, ms, r, sp in zip(res["labels"], res["times_ms"],
                                   res["rmses"],   res["speedup"]):
            r_str = f"{r:.3f}" if r is not None else "  n/a"
            print(f"    {lbl:<28} {ms:>9.2f} {r_str:>9} {sp:>7.1f}×")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    print("=== Бенчмарк на CAMS-Knee (K8L) ===\n")

    # ── Парсинг CAMS-Knee ─────────────────────────────────────
    print("Загрузка CAMS-Knee gait ...")
    meta_g, df_g = parse_cams_csv(CAMS_GAIT)
    bw = meta_g["bw_n"]
    cycle = extract_gait_cycle(df_g, meta_g["events"], bw)
    if cycle is None:
        print("[!] Цикл ходьбы не найден")
        return
    print(f"  Цикл ходьбы: {cycle['t0']:.2f}–{cycle['t1']:.2f}s  "
          f"{cycle['n_frames']} кадров  "
          f"пик эталона={cycle['ref_bw'].max():.2f} BW")

    print("\nЗагрузка CAMS-Knee squat ...")
    meta_s, df_s = parse_cams_csv(CAMS_SQUAT)
    squat = extract_squat(df_s, bw)
    print(f"  {squat['n_frames']} кадров  пик эталона={squat['ref_bw'].max():.2f} BW")

    # ── Шаблон из GC ──────────────────────────────────────────
    datasets = [
        ("K8L_gait",  cycle),
        ("K8L_squat", squat),
    ]

    all_results = []
    for ds_name, ds_data in datasets:
        print(f"\n── {ds_name} ──────────────────────────────────────────────")
        print("  Подготовка шаблона GC ...")
        template = load_gc_template(ds_data["n_frames"])
        res = run_benchmark(ds_name, ds_data, template)
        all_results.append(res)

    # ── Таблица и график ──────────────────────────────────────
    print_table(all_results)
    make_figure(all_results)

    print("\nГотово.")


if __name__ == "__main__":
    main()
