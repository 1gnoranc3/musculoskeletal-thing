"""
run_real_data.py
─────────────────────────────────────────────────────────────────
Запуск дифференцируемой статической оптимизации на реальных данных
Grand Challenge (DM субъект) после прогона run_pipeline.py.

Вход:
  processed/{trial}_id.sto              — суставные моменты (Inverse Dynamics)
  processed/{trial}_grf.sto             — силы реакции опоры
  Synchronized Motion Data/.../eTibia Data/{trial}_knee_forces.csv  — эталон

Выход:
  processed/{trial}_activations.npy     — активации мышц (n_frames, n_muscles)
  processed/{trial}_metrics.json        — пик BW, RMSE, время

Запуск:
  conda activate opensim
  python run_real_data.py                   # все триалы
  python run_real_data.py DM_ngait_og1_new  # один триал
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import opensim as osim
    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False

from static_optimization import (
    DifferentiableStaticOptimizer, ModelParams, OptimizationConfig, CostFunction
)
from joint_reaction import JointReactionAnalysis
from data_loader import load_opensim_sto, load_etibia_reference

# ─────────────────────────────────────────────────────────────
# Настройки путей (синхронизированы с run_pipeline.py)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "/Users/ignorance/Documents/OpenSim/4.5/Models/Rajagopal/Rajagopal2016.osim"
DATA_DIR   = Path("/Users/ignorance/PycharmProjects/diplom4ik/Synchronized Motion Data/Overground Gait Trials")
PROC_DIR   = Path("/Users/ignorance/PycharmProjects/diplom4ik/processed")
ETIBIA_DIR = DATA_DIR / "eTibia Data"

SUBJECT_MASS_KG = 64.0           # DM субъект
BODY_WEIGHT_N   = SUBJECT_MASS_KG * 9.81

# DOFs нижних конечностей которые берём из ID
LOWER_LIMB_DOFS = [
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
]

# ─────────────────────────────────────────────────────────────
# Извлечение моментных плеч и max isometric force из .osim модели
# ─────────────────────────────────────────────────────────────

def extract_model_data(model_path: str, dof_names: list[str], q_traj: np.ndarray,
                       q_columns: list[str]):
    """
    Достать из OpenSim модели:
      - max_forces (n_muscles,)         — F0 для каждой мышцы
      - moment_arms (n_frames, n_dof, n_muscles) — на каждом кадре
      - muscle_names (n_muscles,)       — имена мышц
      - knee_mask (n_muscles,) bool     — флаг "пересекает колено"

    q_traj: (n_frames, n_dof_in_mot) значения координат из .mot
    q_columns: имена колонок в q_traj (без 'time')
    dof_names: какие DOF мы хотим использовать (порядок результата)
    """
    if not OPENSIM_AVAILABLE:
        raise RuntimeError("opensim не установлен — этот шаг требует OpenSim")

    model = osim.Model(model_path)
    state = model.initSystem()

    muscle_set = model.getMuscles()
    n_muscles  = muscle_set.getSize()
    muscle_names = [muscle_set.get(i).getName() for i in range(n_muscles)]

    # max isometric force
    max_forces = np.array([
        muscle_set.get(i).getMaxIsometricForce() for i in range(n_muscles)
    ])

    # Маска: пересекает ли мышца коленный сустав
    # Эвристика по имени: содержит quad/vas/rect/bf/sem/gas (квадрицепс,
    # хамстринги, икроножная — основные мышцы пересекающие колено)
    knee_keywords = ['vas', 'rect', 'bflh', 'bfsh', 'semimem', 'semiten',
                     'grac', 'sart', 'gas', 'tfl']
    knee_mask = np.array([
        any(kw in name.lower() for kw in knee_keywords)
        for name in muscle_names
    ])
    print(f"  Мышц всего: {n_muscles}, пересекают колено: {knee_mask.sum()}")

    # ── Моментные плечи на каждом кадре ────────────────────────────────
    coord_set = model.getCoordinateSet()

    # Найти OpenSim coordinate объекты для нужных DOF
    target_coords = []
    for dof in dof_names:
        try:
            target_coords.append(coord_set.get(dof))
        except Exception:
            print(f"  [!] DOF '{dof}' не найден в модели, пропускаем")
            target_coords.append(None)

    # Индексы колонок в q_traj для нужных DOF
    dof_to_col_idx = {}
    for dof in dof_names:
        if dof in q_columns:
            dof_to_col_idx[dof] = q_columns.index(dof)

    n_frames = q_traj.shape[0]
    n_dof    = len(dof_names)
    moment_arms = np.zeros((n_frames, n_dof, n_muscles))

    print(f"  Расчёт моментных плеч: {n_frames} кадров × {n_dof} DOF × {n_muscles} мышц...")
    t0 = time.perf_counter()

    # Фильтр: для каждого кадра выставить координаты, посчитать moment_arm
    # Берём шаг в 5 кадров для скорости и интерполируем (моментные плечи
    # медленно меняются)
    sample_step = 5
    sample_idx  = list(range(0, n_frames, sample_step))
    if sample_idx[-1] != n_frames - 1:
        sample_idx.append(n_frames - 1)

    moment_arms_sub = np.zeros((len(sample_idx), n_dof, n_muscles))

    for k, frame_idx in enumerate(sample_idx):
        # Установить позы суставов
        for dof, col_idx in dof_to_col_idx.items():
            try:
                coord_set.get(dof).setValue(state, q_traj[frame_idx, col_idx], False)
            except Exception:
                pass

        model.assemble(state)
        model.realizePosition(state)

        # Для каждой пары (DOF, мышца) — moment arm
        for d, coord in enumerate(target_coords):
            if coord is None:
                continue
            for m in range(n_muscles):
                try:
                    moment_arms_sub[k, d, m] = muscle_set.get(m).computeMomentArm(state, coord)
                except Exception:
                    pass

    # Линейная интерполяция на все кадры
    if len(sample_idx) < n_frames:
        for d in range(n_dof):
            for m in range(n_muscles):
                moment_arms[:, d, m] = np.interp(
                    np.arange(n_frames), sample_idx, moment_arms_sub[:, d, m]
                )
    else:
        moment_arms = moment_arms_sub

    print(f"  Моментные плечи: {time.perf_counter() - t0:.1f}s")
    return max_forces, moment_arms, muscle_names, knee_mask


# ─────────────────────────────────────────────────────────────
# Загрузка реальных данных одного триала
# ─────────────────────────────────────────────────────────────

def load_trial(trial_name: str) -> Optional[dict]:
    """
    Загрузить данные одного триала: ID моменты, IK координаты, GRF, эталон.
    """
    id_sto  = PROC_DIR / f"{trial_name}_id.sto"
    ik_mot  = PROC_DIR / f"{trial_name}_ik.mot"
    grf_sto = PROC_DIR / f"{trial_name.replace('_new', '')}_grf.sto"

    if not id_sto.exists():
        print(f"  [!] {id_sto.name} не найден — сначала запустите run_pipeline.py")
        return None
    if not ik_mot.exists():
        print(f"  [!] {ik_mot.name} не найден")
        return None

    # ID моменты
    df_id = load_opensim_sto(id_sto)
    df_id.columns = df_id.columns.str.strip()

    # Имена в .sto: hip_flexion_r_moment, knee_angle_r_moment, ...
    # Маппим на наши LOWER_LIMB_DOFS
    id_cols_present = []
    for dof in LOWER_LIMB_DOFS:
        col = f'{dof}_moment'
        if col in df_id.columns:
            id_cols_present.append(col)
        else:
            print(f"  [!] {col} нет в ID, пропускаем DOF")

    if not id_cols_present:
        print(f"  [!] Нет ни одного нужного DOF в {id_sto.name}")
        return None

    id_torques = df_id[id_cols_present].values
    time_id    = df_id['time'].values

    # IK координаты для расчёта моментных плеч
    df_ik = load_opensim_sto(ik_mot)
    df_ik.columns = df_ik.columns.str.strip()
    q_columns = [c for c in df_ik.columns if c != 'time']
    q_traj    = df_ik[q_columns].values
    time_ik   = df_ik['time'].values

    # Интерполировать IK на сетку ID (на случай разных частот)
    if not np.allclose(time_ik, time_id):
        q_aligned = np.column_stack([
            np.interp(time_id, time_ik, q_traj[:, i]) for i in range(q_traj.shape[1])
        ])
        q_traj = q_aligned

    # GRF (вертикальная составляющая для упрощённого JRA)
    grf_vertical = np.zeros(len(time_id))
    if grf_sto.exists():
        df_grf = load_opensim_sto(grf_sto)
        df_grf.columns = df_grf.columns.str.strip()
        # Колонки: 1_ground_force_vy, 2_ground_force_vy
        time_grf = df_grf[df_grf.columns[0]].values
        vy_cols  = [c for c in df_grf.columns if 'vy' in c.lower()]
        if vy_cols:
            total_vy = df_grf[vy_cols].sum(axis=1).values
            grf_vertical = np.interp(time_id, time_grf, total_vy)

    # Эталон (eTibia) — может отсутствовать у некоторых триалов
    base = trial_name.replace('_new', '')
    knee_csv = ETIBIA_DIR / f"{base}_knee_forces.csv"
    ref_bw = None
    if knee_csv.exists():
        df_ref = pd.read_csv(knee_csv, index_col=False)
        # Полный модуль контактной силы: sqrt(Fx² + Fy² + Fz²)
        total_f = np.sqrt(df_ref['Fx']**2 + df_ref['Fy']**2 + df_ref['Fz']**2)
        time_ref = df_ref['Time(sec)'].values
        # Нормировать на BW и интерполировать на сетку ID
        ref_bw_raw = total_f.values / BODY_WEIGHT_N
        # Только в пересечении временных диапазонов
        mask = (time_id >= time_ref.min()) & (time_id <= time_ref.max())
        if mask.sum() > 10:
            ref_bw = np.interp(time_id, time_ref, ref_bw_raw)

    return {
        'trial':        trial_name,
        'time':         time_id,
        'id_torques':   id_torques,
        'q_traj':       q_traj,
        'q_columns':    q_columns,
        'dof_names':    [c.replace('_moment', '') for c in id_cols_present],
        'grf_vertical': grf_vertical,
        'ref_bw':       ref_bw,
    }


# ─────────────────────────────────────────────────────────────
# Прогон всех 3 вариантов оптимизации (таблица 2 ВКР)
# ─────────────────────────────────────────────────────────────

def process_trial(trial_data: dict):
    name = trial_data['trial']
    print(f"\n══ Триал: {name} ══")
    print(f"  Кадров: {len(trial_data['time'])}, "
          f"DOF: {len(trial_data['dof_names'])}")

    # Извлечь моментные плечи и max forces из модели
    max_forces, moment_arms, muscle_names, knee_mask = extract_model_data(
        MODEL_PATH,
        trial_data['dof_names'],
        trial_data['q_traj'],
        trial_data['q_columns'],
    )

    n_frames  = trial_data['id_torques'].shape[0]
    n_muscles = len(max_forces)
    n_dof     = len(trial_data['dof_names'])

    model = ModelParams(
        n_muscles=n_muscles,
        n_dof=n_dof,
        max_isometric_force=max_forces,
        moment_arms=moment_arms,
    )

    results = {}

    # ── Вариант 3 (целевой): CasADi + ∑a³ + резервы 20 Н·м ─────
    print("\n  ▸ CasADi + ∑a³ + резервы 20 Н·м (целевой вариант)")
    cfg = OptimizationConfig(
        cost_function=CostFunction.SUM_CUBES,
        reserve_actuator_weight=20.0,
        verbose=False,
    )
    opt = DifferentiableStaticOptimizer(model, cfg)
    res = opt.solve_trajectory(moment_arms, max_forces, trial_data['id_torques'])

    activations = res['activations']
    print(f"    Время: {res['total_time']:.2f}s  "
          f"Среднее на кадр: {res['mean_frame_time']*1000:.1f}ms")
    print(f"    Среднее число итераций IPOPT: {res['mean_iter_count']:.1f}")

    # JRA + сравнение с эталоном
    jra = JointReactionAnalysis(body_mass_kg=SUBJECT_MASS_KG)
    jra_res = jra.compute_knee_contact_force(
        activations, max_forces, knee_mask, trial_data['grf_vertical']
    )
    peak_bw = float(np.max(jra_res.contact_force_bw))
    print(f"    Пик силы колена: {peak_bw:.2f} BW")

    rmse_bw = None
    if trial_data['ref_bw'] is not None:
        rmse_bw = JointReactionAnalysis.compute_rmse_bw(
            jra_res.contact_force_bw, trial_data['ref_bw']
        )
        peak_ref = float(np.max(trial_data['ref_bw']))
        print(f"    Эталон пик: {peak_ref:.2f} BW   RMSE: {rmse_bw:.3f} BW")

    # Сохранить
    np.save(PROC_DIR / f"{name}_activations.npy", activations)
    np.save(PROC_DIR / f"{name}_knee_force_bw.npy", jra_res.contact_force_bw)

    metrics = {
        'trial':              name,
        'n_frames':           n_frames,
        'n_muscles':          n_muscles,
        'n_dof':              n_dof,
        'total_time_s':       float(res['total_time']),
        'mean_frame_time_ms': float(res['mean_frame_time'] * 1000),
        'mean_iter_count':    float(res['mean_iter_count']),
        'peak_knee_bw':       peak_bw,
        'reference_peak_bw':  float(np.max(trial_data['ref_bw'])) if trial_data['ref_bw'] is not None else None,
        'rmse_bw':            rmse_bw,
    }

    with open(PROC_DIR / f"{name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    if not OPENSIM_AVAILABLE:
        print("❌ OpenSim не установлен. conda install -c opensim-org opensim")
        return

    # Список триалов: либо из аргумента, либо все *_id.sto
    if len(sys.argv) > 1:
        trial_names = [sys.argv[1]]
    else:
        id_files = sorted(PROC_DIR.glob("*_id.sto"))
        trial_names = [p.stem.replace('_id', '') for p in id_files]

    if not trial_names:
        print(f"[!] В {PROC_DIR} нет *_id.sto файлов")
        print(f"    Сначала запустите: python run_pipeline.py")
        return

    print(f"К обработке: {len(trial_names)} триалов\n")
    all_metrics = []

    for name in trial_names:
        trial_data = load_trial(name)
        if trial_data is None:
            continue
        try:
            m = process_trial(trial_data)
            all_metrics.append(m)
        except Exception as e:
            print(f"  [!] Ошибка: {e}")
            import traceback
            traceback.print_exc()

    # Сводная таблица
    if all_metrics:
        print("\n" + "═" * 80)
        print(f"{'Триал':<25} {'Кадров':>8} {'Время,с':>8} "
              f"{'Пик BW':>8} {'Эталон':>8} {'RMSE':>8}")
        print("─" * 80)
        for m in all_metrics:
            ref  = f"{m['reference_peak_bw']:.2f}" if m['reference_peak_bw'] is not None else "—"
            rmse = f"{m['rmse_bw']:.3f}"           if m['rmse_bw']           is not None else "—"
            print(f"{m['trial']:<25} {m['n_frames']:>8} "
                  f"{m['total_time_s']:>8.2f} {m['peak_knee_bw']:>8.2f} "
                  f"{ref:>8} {rmse:>8}")
        print("═" * 80)

        # Усреднение для табл. 2 ВКР
        rmses = [m['rmse_bw'] for m in all_metrics if m['rmse_bw'] is not None]
        times = [m['total_time_s'] for m in all_metrics]
        if rmses:
            print(f"\nСреднее по {len(rmses)} триалам с эталоном:")
            print(f"  RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f} BW")
        print(f"  Среднее время: {np.mean(times):.2f} ± {np.std(times):.2f} с")


if __name__ == "__main__":
    main()