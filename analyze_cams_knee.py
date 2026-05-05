"""
Анализ данных CAMS-Knee (K8L) и сравнение с Grand Challenge (DM).

Таблица 3 ВКР — характеристики in-vivo датасетов:
  • Grand Challenge (DM, eTibia-датчик) — из processed/*_optdata.npz
  • CAMS-Knee (K8L, имплантный сенсор) — gait + squat

Запуск:
  python analyze_cams_knee.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


CAMS_GAIT  = Path("CAMS_Knee_Sample/K8L/gait/export_proc/K8L_CAMS_Knee_gait_sample.csv")
CAMS_SQUAT = Path("CAMS_Knee_Sample/K8L/squat/export_proc/K8L_CAMS_Knee_squat_sample.csv")
PROC_DIR   = Path("processed")
OUT_DIR    = PROC_DIR


# ─────────────────────────────────────────────────────────────
# Парсер CAMS-Knee CSV
# ─────────────────────────────────────────────────────────────

def parse_cams_csv(csv_path: Path) -> tuple[dict, pd.DataFrame]:
    with open(csv_path, "r") as f:
        lines = f.readlines()

    meta = {"bw_n": 773.0, "activity": "unknown", "patient": "K8L"}
    events = []

    for line in lines[:22]:
        s = line.strip()
        if s.startswith("Bodyweight"):
            meta["bw_n"] = float(s.split(",")[1])
        elif s.startswith("Activity"):
            meta["activity"] = s.split(",")[1].strip()
        elif s.startswith("Patient"):
            meta["patient"] = s.split(",")[1].strip()
        elif "Foot Strike" in s or "Foot Off" in s:
            parts = s.split(",")
            try:
                events.append({"event": parts[0].strip(), "time": float(parts[1])})
            except (ValueError, IndexError):
                pass

    meta["events"] = events

    df = pd.read_csv(csv_path, skiprows=23, header=0, low_memory=False)
    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return meta, df


# ─────────────────────────────────────────────────────────────
# Нарезка на циклы ходьбы по событиям
# ─────────────────────────────────────────────────────────────

def extract_gait_cycles(df: pd.DataFrame, events: list, bw: float) -> list[dict]:
    """Извлечь циклы правой ноги (Right Foot Strike → Right Foot Strike)."""
    rhs_times = sorted(
        e["time"] for e in events if e["event"] == "Right Foot Strike"
    )
    if len(rhs_times) < 2:
        return []

    cycles = []
    for i in range(len(rhs_times) - 1):
        t0, t1 = rhs_times[i], rhs_times[i + 1]
        seg = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
        seg_valid = seg[["time", "Fx", "Fy", "Fz", "Fres"]].dropna()
        if len(seg_valid) < 5:
            continue

        # Вертикальная GRF (Fz_lab активных платформ)
        grf_cols = [f"Fz_lab_{j}" for j in range(1, 9) if f"Fz_lab_{j}" in df.columns]
        grf_seg  = seg[["time"] + grf_cols].copy()
        grf_total = grf_seg[grf_cols].abs().sum(axis=1)
        grf_peak  = grf_total.max()

        cycles.append({
            "t_start":    t0,
            "t_end":      t1,
            "duration_s": t1 - t0,
            "n_frames":   len(seg_valid),
            "fres_peak_n":  seg_valid["Fres"].max(),
            "fres_mean_n":  seg_valid["Fres"].mean(),
            "fz_peak_n":    seg_valid["Fz"].abs().max(),
            "fres_peak_bw": seg_valid["Fres"].max() / bw,
            "fres_mean_bw": seg_valid["Fres"].mean() / bw,
            "fz_peak_bw":   seg_valid["Fz"].abs().max() / bw,
            "grf_peak_bw":  grf_peak / bw,
            "fres_arr":     seg_valid["Fres"].values / bw,
            "time_arr":     seg_valid["time"].values,
        })

    return cycles


def extract_squat_stats(df: pd.DataFrame, bw: float) -> dict:
    valid = df[["time", "Fx", "Fy", "Fz", "Fres"]].dropna()
    if valid.empty:
        return {}
    return {
        "n_frames":     len(valid),
        "fres_peak_bw": valid["Fres"].max() / bw,
        "fz_peak_bw":   valid["Fz"].abs().max() / bw,
        "fres_arr":     valid["Fres"].values / bw,
        "time_arr":     valid["time"].values,
    }


# ─────────────────────────────────────────────────────────────
# Grand Challenge eTibia данные из *_optdata.npz
# ─────────────────────────────────────────────────────────────

def load_gc_reference() -> list[dict]:
    results = []
    for npz in sorted(PROC_DIR.glob("*_optdata.npz")):
        d = np.load(npz, allow_pickle=True)
        ref = d["ref_bw"]
        valid = ~np.isnan(ref)
        if not valid.any():
            continue
        results.append({
            "name":         npz.stem.replace("_optdata", ""),
            "n_frames":     int(valid.sum()),
            "fres_peak_bw": float(ref[valid].max()),
            "fres_mean_bw": float(ref[valid].mean()),
            "ref_arr":      ref[valid],
        })
    return results


# ─────────────────────────────────────────────────────────────
# Сводная таблица
# ─────────────────────────────────────────────────────────────

def print_summary_table(
    gc_trials: list[dict],
    cams_gait_cycles: list[dict],
    cams_squat: dict,
    bw_cams: float,
):
    print("\n" + "=" * 72)
    print(f"{'Характеристика':<28} {'Grand Challenge (DM)':<22} {'CAMS-Knee (K8L)'}")
    print("─" * 72)

    rows = [
        ("Субъект",             "DM (анонимн.)",       "K8L"),
        ("Масса тела [кг]",     "64.0",                f"{bw_cams/9.81:.1f}"),
        ("Вес тела [Н]",        "627.8",               f"{bw_cams:.0f}"),
        ("Имплантный сенсор",   "eTibia (Fz, осевая)", "3D (Fx/Fy/Fz + моменты)"),
        ("Активностей",         "5 (ходьба, крауч…)",  "2 (ходьба + присед)"),
    ]

    # Ходьба: GC
    gc_peaks   = [t["fres_peak_bw"] for t in gc_trials]
    gc_means   = [t["fres_mean_bw"] for t in gc_trials]
    gc_n       = sum(t["n_frames"] for t in gc_trials)

    # Ходьба: CAMS
    if cams_gait_cycles:
        cams_g_peaks = [c["fres_peak_bw"] for c in cams_gait_cycles]
        cams_g_means = [c["fres_mean_bw"] for c in cams_gait_cycles]
        cams_g_dur   = [c["duration_s"] for c in cams_gait_cycles]

    rows += [
        ("── Ходьба ──",        "─" * 20,              "─" * 18),
        ("Триалов / циклов",
            f"{len(gc_trials)}",
            f"{len(cams_gait_cycles)} цикла" if cams_gait_cycles else "─"),
        ("Кадров с эталоном",
            f"{gc_n}",
            f"{sum(c['n_frames'] for c in cams_gait_cycles)}" if cams_gait_cycles else "─"),
        ("Пик Fres [BW]",
            f"{np.mean(gc_peaks):.2f} ± {np.std(gc_peaks):.2f}",
            f"{np.mean(cams_g_peaks):.2f} ± {np.std(cams_g_peaks):.2f}" if cams_gait_cycles else "─"),
        ("Среднее Fres [BW]",
            f"{np.mean(gc_means):.2f} ± {np.std(gc_means):.2f}",
            f"{np.mean(cams_g_means):.2f} ± {np.std(cams_g_means):.2f}" if cams_gait_cycles else "─"),
        ("Длительность цикла",
            "~1.0–1.2 с",
            f"{np.mean(cams_g_dur):.2f} ± {np.std(cams_g_dur):.2f} с" if cams_gait_cycles else "─"),
    ]

    if cams_squat:
        rows += [
            ("── Присед ──",    "─" * 20,              "─" * 18),
            ("Пик Fres [BW]",   "—",                   f"{cams_squat['fres_peak_bw']:.2f}"),
            ("Кадров",          "—",                   f"{cams_squat['n_frames']}"),
        ]

    for label, gc_val, cams_val in rows:
        print(f"  {label:<26} {gc_val:<22} {cams_val}")

    print("=" * 72)


# ─────────────────────────────────────────────────────────────
# Визуализация
# ─────────────────────────────────────────────────────────────

def make_figure(
    gc_trials: list[dict],
    cams_gait_cycles: list[dict],
    cams_squat: dict,
):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Сравнение in-vivo эталонных данных\n"
        "Grand Challenge (DM, eTibia) vs CAMS-Knee (K8L, имплантный сенсор)",
        fontsize=12, fontweight="bold",
    )

    gc_peaks  = [t["fres_peak_bw"] for t in gc_trials]
    gc_means  = [t["fres_mean_bw"] for t in gc_trials]
    gc_names  = [t["name"] for t in gc_trials]

    colors_gc   = "#5b8db8"
    colors_cams = "#e07b54"
    colors_squat = "#9c27b0"

    # ── 1. Пиковые силы по триалам/циклам ──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.bar(range(len(gc_peaks)), gc_peaks,
            color=colors_gc, alpha=0.85, label="GC (DM)", width=0.6)
    if cams_gait_cycles:
        cams_g_peaks = [c["fres_peak_bw"] for c in cams_gait_cycles]
        ax1.bar(range(len(gc_peaks), len(gc_peaks) + len(cams_g_peaks)),
                cams_g_peaks, color=colors_cams, alpha=0.85,
                label="CAMS-Knee (ходьба)", width=0.6)
    ax1.axhline(np.mean(gc_peaks), color=colors_gc, linestyle="--",
                linewidth=1.5, alpha=0.6)
    if cams_gait_cycles:
        ax1.axhline(np.mean([c["fres_peak_bw"] for c in cams_gait_cycles]),
                    color=colors_cams, linestyle="--", linewidth=1.5, alpha=0.6)
    ax1.set_ylabel("Пиковая Fres [BW]")
    ax1.set_title("Пиковые контактные силы")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", labelbottom=False)

    # ── 2. Box: средние силы ──
    ax2 = fig.add_subplot(2, 3, 2)
    box_data  = [gc_means]
    box_labels = ["GC (DM)\neTibia"]
    box_colors = [colors_gc]
    if cams_gait_cycles:
        box_data.append([c["fres_mean_bw"] for c in cams_gait_cycles])
        box_labels.append("CAMS-Knee\n(ходьба)")
        box_colors.append(colors_cams)
    if cams_squat:
        box_data.append([cams_squat["fres_peak_bw"]])
        box_labels.append("CAMS-Knee\n(присед)")
        box_colors.append(colors_squat)
    bp = ax2.boxplot(box_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax2.set_xticklabels(box_labels, fontsize=8)
    ax2.set_ylabel("Среднее Fres [BW]")
    ax2.set_title("Распределение средних сил")

    # ── 3. Кривые CAMS-Knee ходьба ──
    ax3 = fig.add_subplot(2, 3, 3)
    if cams_gait_cycles:
        for i, cyc in enumerate(cams_gait_cycles):
            pct = np.linspace(0, 100, len(cyc["fres_arr"]))
            ax3.plot(pct, cyc["fres_arr"],
                     color=colors_cams, alpha=0.75, linewidth=1.8,
                     label=f"Цикл {i+1} (пик {cyc['fres_peak_bw']:.2f} BW)")
        ax3.set_xlabel("% цикла ходьбы")
        ax3.set_ylabel("Fres [BW]")
        ax3.set_title("CAMS-Knee: циклы ходьбы\n(тибиальный имплант)")
        ax3.legend(fontsize=7)
        ax3.set_ylim(bottom=0)

    # ── 4. Кривая CAMS-Knee присед ──
    ax4 = fig.add_subplot(2, 3, 4)
    if cams_squat:
        pct = np.linspace(0, 100, len(cams_squat["fres_arr"]))
        ax4.plot(pct, cams_squat["fres_arr"],
                 color=colors_squat, linewidth=2.0)
        ax4.axhline(cams_squat["fres_peak_bw"], color="grey",
                    linestyle="--", linewidth=1, alpha=0.6,
                    label=f'Пик {cams_squat["fres_peak_bw"]:.2f} BW')
        ax4.set_xlabel("% движения")
        ax4.set_ylabel("Fres [BW]")
        ax4.set_title("CAMS-Knee: присед")
        ax4.legend(fontsize=8)
        ax4.set_ylim(bottom=0)

    # ── 5. Grand Challenge кривые eTibia ──
    ax5 = fig.add_subplot(2, 3, 5)
    for trial in gc_trials[:12]:
        pct = np.linspace(0, 100, len(trial["ref_arr"]))
        ax5.plot(pct, trial["ref_arr"], color=colors_gc, alpha=0.4, linewidth=1)
    if gc_trials:
        # Среднее по всем триалам (интерполяция на 100 точек)
        common = np.linspace(0, 100, 100)
        stack  = np.array([
            np.interp(common, np.linspace(0, 100, len(t["ref_arr"])), t["ref_arr"])
            for t in gc_trials
        ])
        ax5.plot(common, stack.mean(axis=0), color=colors_gc,
                 linewidth=2.5, label=f"Среднее (n={len(gc_trials)})")
    ax5.set_xlabel("% триала")
    ax5.set_ylabel("Fres/|Fz| [BW]")
    ax5.set_title("Grand Challenge: eTibia (DM)")
    ax5.legend(fontsize=8)
    ax5.set_ylim(bottom=0)

    # ── 6. Сравнительный bar: пики ──
    ax6 = fig.add_subplot(2, 3, 6)
    labels6  = ["GC (DM)\nходьба", "CAMS-Knee\nходьба", "CAMS-Knee\nприсед"]
    values6  = [np.mean(gc_peaks), 0.0, 0.0]
    colors6  = [colors_gc, colors_cams, colors_squat]

    if cams_gait_cycles:
        values6[1] = np.mean([c["fres_peak_bw"] for c in cams_gait_cycles])
    if cams_squat:
        values6[2] = cams_squat["fres_peak_bw"]

    bars = ax6.bar(range(3), values6, color=colors6, alpha=0.85, width=0.5)
    ax6.set_xticks(range(3))
    ax6.set_xticklabels(labels6, fontsize=8)
    ax6.set_ylabel("Средний пик Fres [BW]")
    ax6.set_title("Средний пик по активностям")
    for bar, v in zip(bars, values6):
        if v > 0:
            ax6.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "cams_knee_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nГрафик сохранён: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# CSV-таблица для ВКР
# ─────────────────────────────────────────────────────────────

def save_csv_table(
    gc_trials: list[dict],
    cams_gait_cycles: list[dict],
    cams_squat: dict,
    bw_cams: float,
):
    rows = []

    for t in gc_trials:
        rows.append({
            "Датасет":         "Grand Challenge",
            "Субъект":         "DM",
            "Активность":      t["name"].split("_")[1] if "_" in t["name"] else "gait",
            "Триал/цикл":      t["name"],
            "Кадров":          t["n_frames"],
            "Пик_Fres_BW":     round(t["fres_peak_bw"], 3),
            "Среднее_Fres_BW": round(t["fres_mean_bw"], 3),
            "GRF_пик_BW":      "—",
            "Сенсор":          "eTibia (Fz)",
        })

    for i, c in enumerate(cams_gait_cycles):
        rows.append({
            "Датасет":         "CAMS-Knee",
            "Субъект":         "K8L",
            "Активность":      "gait",
            "Триал/цикл":      f"gait_cycle_{i+1}",
            "Кадров":          c["n_frames"],
            "Пик_Fres_BW":     round(c["fres_peak_bw"], 3),
            "Среднее_Fres_BW": round(c["fres_mean_bw"], 3),
            "GRF_пик_BW":      round(c["grf_peak_bw"], 3),
            "Сенсор":          "Tibial implant 3D",
        })

    if cams_squat:
        rows.append({
            "Датасет":         "CAMS-Knee",
            "Субъект":         "K8L",
            "Активность":      "squat",
            "Триал/цикл":      "squat_sample",
            "Кадров":          cams_squat["n_frames"],
            "Пик_Fres_BW":     round(cams_squat["fres_peak_bw"], 3),
            "Среднее_Fres_BW": "—",
            "GRF_пик_BW":      "—",
            "Сенсор":          "Tibial implant 3D",
        })

    df_out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "cams_knee_table.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Таблица CSV сохранена: {csv_path}")
    return df_out


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    print("=== Анализ CAMS-Knee (K8L) ===\n")

    # 1. CAMS gait
    print(f"Загрузка: {CAMS_GAIT.name}")
    meta_g, df_g = parse_cams_csv(CAMS_GAIT)
    bw = meta_g["bw_n"]
    print(f"  Субъект: {meta_g['patient']}, Вес: {bw} Н, Активность: {meta_g['activity']}")
    print(f"  События: {len(meta_g['events'])}")
    cycles = extract_gait_cycles(df_g, meta_g["events"], bw)
    print(f"  Циклов ходьбы: {len(cycles)}")
    for i, c in enumerate(cycles):
        print(f"    Цикл {i+1}: {c['t_start']:.2f}–{c['t_end']:.2f}s  "
              f"пик={c['fres_peak_bw']:.2f} BW  GRF_пик={c['grf_peak_bw']:.2f} BW")

    # 2. CAMS squat
    print(f"\nЗагрузка: {CAMS_SQUAT.name}")
    meta_s, df_s = parse_cams_csv(CAMS_SQUAT)
    squat_stats = extract_squat_stats(df_s, bw)
    print(f"  Кадров: {squat_stats.get('n_frames', 0)}, "
          f"пик Fres={squat_stats.get('fres_peak_bw', 0):.2f} BW")

    # 3. Grand Challenge
    print(f"\nЗагрузка Grand Challenge из {PROC_DIR}/ ...")
    gc_trials = load_gc_reference()
    gc_with_ref = [t for t in gc_trials]
    print(f"  Триалов с eTibia эталоном: {len(gc_with_ref)}")
    if gc_with_ref:
        peaks = [t["fres_peak_bw"] for t in gc_with_ref]
        print(f"  Пик Fres: {np.mean(peaks):.2f} ± {np.std(peaks):.2f} BW "
              f"[{np.min(peaks):.2f} – {np.max(peaks):.2f}]")

    # 4. Таблица
    print_summary_table(gc_with_ref, cycles, squat_stats, bw)

    # 5. CSV
    df_table = save_csv_table(gc_with_ref, cycles, squat_stats, bw)
    print("\nСтроки таблицы:")
    print(df_table.to_string(index=False))

    # 6. График
    make_figure(gc_with_ref, cycles, squat_stats)

    print("\nГотово. Следующий шаг: добавить K8L в статическую оптимизацию "
          "(требует OpenSim IK/ID для CAMS-Knee .c3d данных).")


if __name__ == "__main__":
    main()
