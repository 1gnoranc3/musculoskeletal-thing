import sys
import json
import time
import numpy as np
from pathlib import Path

from static_optimization import (
    DifferentiableStaticOptimizer, ModelParams, OptimizationConfig, CostFunction
)

PROC_DIR = Path("/Users/ignorance/PycharmProjects/diplom4ik/processed")

VARIANTS = [
    (CostFunction.SUM_SQUARES,    "∑a²      "),
    (CostFunction.SUM_CUBES,      "∑a³      "),
    (CostFunction.POLYNOMIAL_3,   "poly-3   "),
    (CostFunction.METABOLIC_PROXY,"metabolic"),
]


def compute_knee_contact_force_bw(
    activations: np.ndarray,
    max_forces: np.ndarray,
    knee_mask: np.ndarray,
    grf_vertical: np.ndarray,
    body_weight_n: float,
    muscle_names: np.ndarray | None = None,
) -> np.ndarray:
    muscle_forces = activations * max_forces[np.newaxis, :]
    if muscle_names is not None:
        right_mask = knee_mask & np.array(['_r' in str(n).lower() for n in muscle_names])
    else:
        right_mask = knee_mask.copy()
        right_indices = np.where(knee_mask)[0]
        right_mask[right_indices[len(right_indices) // 2:]] = False
    contact_n = grf_vertical + np.sum(muscle_forces[:, right_mask], axis=1)
    return contact_n / body_weight_n


def process_trial(npz_path: Path) -> dict:
    name = npz_path.stem.replace('_optdata', '')
    print(f"\n══ {name} ══")

    data            = np.load(npz_path, allow_pickle=True)
    moment_arms     = data['moment_arms']
    max_forces      = data['max_forces']
    id_torques      = data['id_torques']
    knee_mask       = data['knee_mask']
    grf_vertical    = data['grf_vertical']
    ref_bw          = data['ref_bw']
    muscle_names    = data['muscle_names'] if 'muscle_names' in data else None
    body_weight_n   = float(data['subject_mass_kg']) * 9.81

    n_frames, n_dof = id_torques.shape
    n_muscles       = len(max_forces)
    print(f"  frames={n_frames}, dof={n_dof}, muscles={n_muscles}")

    model     = ModelParams(n_muscles=n_muscles, n_dof=n_dof,
                            max_isometric_force=max_forces, moment_arms=moment_arms)
    valid_ref = ~np.isnan(ref_bw)
    results   = []

    for cost_fn, label in VARIANTS:
        cfg = OptimizationConfig(cost_function=cost_fn,
                                 reserve_actuator_weight=1000.0, verbose=False)
        res         = DifferentiableStaticOptimizer(model, cfg).solve_trajectory(moment_arms, max_forces, id_torques)
        knee_bw     = compute_knee_contact_force_bw(res['activations'], max_forces, knee_mask,
                                                    grf_vertical, body_weight_n, muscle_names)
        peak_bw     = float(np.max(knee_bw))
        rmse_bw     = float(np.sqrt(np.mean((knee_bw[valid_ref] - ref_bw[valid_ref])**2))) if valid_ref.any() else None
        results.append({'cost_fn': cost_fn, 'label': label,
                        'total_time_s': res['total_time'],
                        'mean_frame_ms': res['mean_frame_time'] * 1000,
                        'mean_iter': res['mean_iter_count'],
                        'peak_bw': peak_bw, 'rmse_bw': rmse_bw})
        print(f"  {label}: {res['total_time']:.2f}s  {res['mean_frame_time']*1000:.1f}ms/frame  "
              f"iter={res['mean_iter_count']:.1f}  peak={peak_bw:.2f}BW  "
              f"RMSE={'—' if rmse_bw is None else f'{rmse_bw:.3f}'}BW")

    best  = min((v for v in results if v['rmse_bw'] is not None),
                key=lambda v: v['rmse_bw'], default=results[0])
    acts  = DifferentiableStaticOptimizer(
                model, OptimizationConfig(cost_function=best['cost_fn'],
                                          reserve_actuator_weight=1000.0, verbose=False)
            ).solve_trajectory(moment_arms, max_forces, id_torques)['activations']
    knee_bw = compute_knee_contact_force_bw(acts, max_forces, knee_mask, grf_vertical, body_weight_n)

    np.save(PROC_DIR / f"{name}_activations.npy", acts)
    np.save(PROC_DIR / f"{name}_knee_force_bw.npy", knee_bw)

    metrics = {'trial': name, 'n_frames': int(n_frames), 'n_muscles': int(n_muscles),
               'n_dof': int(n_dof), 'variants': results, 'best_cost_fn': best['label'].strip(),
               'peak_knee_bw': best['peak_bw'],
               'reference_peak_bw': float(np.max(ref_bw[valid_ref])) if valid_ref.any() else None,
               'rmse_bw': best['rmse_bw']}
    with open(PROC_DIR / f"{name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics


def main():
    npz_files = ([PROC_DIR / f"{sys.argv[1]}_optdata.npz"] if len(sys.argv) > 1
                 else sorted(PROC_DIR.glob("*_optdata.npz")))
    if not npz_files:
        print(f"[!] No *_optdata.npz in {PROC_DIR}")
        print("    Run: conda activate opensim && python prepare_data.py")
        return

    print(f"Processing {len(npz_files)} trials\n")
    all_metrics = []
    for npz in npz_files:
        if not npz.exists():
            print(f"[!] {npz.name} not found, skipping")
            continue
        try:
            all_metrics.append(process_trial(npz))
        except Exception as e:
            print(f"  [!] Error: {e}")
            import traceback; traceback.print_exc()

    if not all_metrics:
        return

    variant_labels = [label.strip() for _, label in VARIANTS]
    print("\n" + "═" * 80)
    print(f"{'':16}", end="")
    for lbl in variant_labels:
        print(f"  {lbl:>12}", end="")
    print()
    print("─" * 80)
    for metric_key, unit, fmt in [("mean_frame_ms", "ms/frame", ".2f"),
                                   ("mean_iter", "iterations", ".1f"),
                                   ("rmse_bw", "RMSE BW", ".3f")]:
        print(f"{unit:<16}", end="")
        for lbl in variant_labels:
            vals = [v[metric_key] for m in all_metrics for v in m.get('variants', [])
                    if v['label'].strip() == lbl and v[metric_key] is not None]
            print(f"  {format(np.mean(vals), fmt) if vals else '—':>12}", end="")
        print()
    print("═" * 80)

    rmses = [m['rmse_bw'] for m in all_metrics if m.get('rmse_bw') is not None]
    if rmses:
        print(f"\nBest variant (mean RMSE over {len(rmses)} trials):")
        for _, lbl in VARIANTS:
            lbl = lbl.strip()
            vals = [v['rmse_bw'] for m in all_metrics for v in m.get('variants', [])
                    if v['label'].strip() == lbl and v['rmse_bw'] is not None]
            if vals:
                ms_vals = [v['mean_frame_ms'] for m in all_metrics for v in m.get('variants', [])
                           if v['label'].strip() == lbl]
                print(f"  {lbl:<12}: RMSE={np.mean(vals):.3f}±{np.std(vals):.3f} BW  "
                      f"speed={np.mean(ms_vals):.2f} ms/frame")


if __name__ == "__main__":
    main()
