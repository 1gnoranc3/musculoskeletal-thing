
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import opensim as osim
except ImportError:
    print("❌ opensim не установлен. conda install -c opensim-org opensim")
    sys.exit(1)

MODEL_PATH = "/Users/ignorance/PycharmProjects/diplom4ik/Rajagopal_DM_scaled.osim"
DATA_DIR   = Path("/Users/ignorance/PycharmProjects/diplom4ik/Synchronized Motion Data/Overground Gait Trials")
PROC_DIR   = Path("/Users/ignorance/PycharmProjects/diplom4ik/processed")
ETIBIA_DIR = DATA_DIR / "eTibia Data"

SUBJECT_MASS_KG = 64.0
BODY_WEIGHT_N   = SUBJECT_MASS_KG * 9.81

LOWER_LIMB_DOFS = [
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
    'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
]

def load_opensim_sto(filepath: Path) -> pd.DataFrame:
    """Загрузить .sto или .mot файл в DataFrame."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            header_end = i + 1
            break
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=header_end)
    return df

def extract_model_data(dof_names: list, q_traj: np.ndarray, q_columns: list):
    """
    Достать из OpenSim модели:
      max_forces (n_muscles,)
      moment_arms (n_frames, n_dof, n_muscles)
      muscle_names list
      knee_mask (n_muscles,) bool
    """
    model = osim.Model(MODEL_PATH)
    state = model.initSystem()

    muscle_set   = model.getMuscles()
    n_muscles    = muscle_set.getSize()
    muscle_names = [muscle_set.get(i).getName() for i in range(n_muscles)]
    max_forces   = np.array([
        muscle_set.get(i).getMaxIsometricForce() for i in range(n_muscles)
    ])

    knee_keywords = ['vas', 'rect', 'bflh', 'bfsh', 'semimem', 'semiten',
                     'grac', 'sart', 'gas', 'tfl']
    knee_mask = np.array([
        any(kw in name.lower() for kw in knee_keywords)
        for name in muscle_names
    ])
    print(f"    Мышц всего: {n_muscles}, пересекают колено: {knee_mask.sum()}")

    coord_set = model.getCoordinateSet()
    target_coords = []
    for dof in dof_names:
        try:
            target_coords.append(coord_set.get(dof))
        except Exception:
            target_coords.append(None)

    dof_to_col_idx = {}
    for dof in dof_names:
        if dof in q_columns:
            dof_to_col_idx[dof] = q_columns.index(dof)

    n_frames = q_traj.shape[0]
    n_dof    = len(dof_names)

    sample_step = 5
    sample_idx  = list(range(0, n_frames, sample_step))
    if sample_idx[-1] != n_frames - 1:
        sample_idx.append(n_frames - 1)

    moment_arms_sub = np.zeros((len(sample_idx), n_dof, n_muscles))
    print(f"    Расчёт моментных плеч: {len(sample_idx)} опорных кадров...")
    t0 = time.perf_counter()

    for k, frame_idx in enumerate(sample_idx):
        for dof, col_idx in dof_to_col_idx.items():
            try:
                coord_set.get(dof).setValue(state, q_traj[frame_idx, col_idx], False)
            except Exception:
                pass
        model.assemble(state)
        model.realizePosition(state)

        for d, coord in enumerate(target_coords):
            if coord is None:
                continue
            for m in range(n_muscles):
                try:
                    moment_arms_sub[k, d, m] = muscle_set.get(m).computeMomentArm(state, coord)
                except Exception:
                    pass

    moment_arms = np.zeros((n_frames, n_dof, n_muscles))
    if len(sample_idx) < n_frames:
        for d in range(n_dof):
            for m in range(n_muscles):
                moment_arms[:, d, m] = np.interp(
                    np.arange(n_frames), sample_idx, moment_arms_sub[:, d, m]
                )
    else:
        moment_arms = moment_arms_sub

    print(f"    Готово за {time.perf_counter() - t0:.1f}s")
    return max_forces, moment_arms, muscle_names, knee_mask

def prepare_trial(trial_name: str) -> bool:
    print(f"\n══ {trial_name} ══")

    id_sto  = PROC_DIR / f"{trial_name}_id.sto"
    ik_mot  = PROC_DIR / f"{trial_name}_ik.mot"
    base    = trial_name.replace('_new', '')
    grf_sto = PROC_DIR / f"{base}_grf.sto"

    if not id_sto.exists():
        print(f"    [!] {id_sto.name} не найден — сначала run_pipeline.py")
        return False
    if not ik_mot.exists():
        print(f"    [!] {ik_mot.name} не найден")
        return False

    df_id = load_opensim_sto(id_sto)
    df_id.columns = df_id.columns.str.strip()
    id_cols = [f'{dof}_moment' for dof in LOWER_LIMB_DOFS if f'{dof}_moment' in df_id.columns]
    if not id_cols:
        print(f"    [!] Нет нужных DOF в ID")
        return False
    id_torques = df_id[id_cols].values
    time_id    = df_id['time'].values
    dof_names  = [c.replace('_moment', '') for c in id_cols]
    print(f"    Кадров: {len(time_id)}, DOF: {len(dof_names)}")

    df_ik = load_opensim_sto(ik_mot)
    df_ik.columns = df_ik.columns.str.strip()
    q_columns = [c for c in df_ik.columns if c != 'time']
    q_traj    = df_ik[q_columns].values
    time_ik   = df_ik['time'].values

    if not np.allclose(time_ik, time_id):
        q_traj = np.column_stack([
            np.interp(time_id, time_ik, q_traj[:, i]) for i in range(q_traj.shape[1])
        ])

    grf_vertical = np.zeros(len(time_id))
    if grf_sto.exists():
        df_grf = load_opensim_sto(grf_sto)
        df_grf.columns = df_grf.columns.str.strip()
        time_grf = df_grf[df_grf.columns[0]].values
        vy_cols  = [c for c in df_grf.columns if 'vy' in c.lower()]
        if vy_cols:
            total = df_grf[vy_cols].sum(axis=1).values
            grf_vertical = np.interp(time_id, time_grf, total)

    knee_csv = ETIBIA_DIR / f"{base}_knee_forces.csv"
    ref_bw = np.full(len(time_id), np.nan)
    if knee_csv.exists():
        df_ref = pd.read_csv(knee_csv, index_col=False)
        total_f = np.sqrt(df_ref['Fx']**2 + df_ref['Fy']**2 + df_ref['Fz']**2).values
        time_ref = df_ref['Time(sec)'].values
        ref_bw_raw = total_f / BODY_WEIGHT_N
        if time_ref.max() > time_id.min() and time_ref.min() < time_id.max():
            ref_bw = np.interp(time_id, time_ref, ref_bw_raw)
            print(f"    Эталон загружен (пик: {ref_bw[~np.isnan(ref_bw)].max():.2f} BW)")
        else:
            print(f"    Эталон не пересекается по времени")
    else:
        print(f"    Эталон отсутствует")

    max_forces, moment_arms, muscle_names, knee_mask = extract_model_data(
        dof_names, q_traj, q_columns
    )

    out_path = PROC_DIR / f"{trial_name}_optdata.npz"
    np.savez_compressed(
        out_path,
        time=time_id,
        id_torques=id_torques,
        moment_arms=moment_arms,
        max_forces=max_forces,
        knee_mask=knee_mask,
        grf_vertical=grf_vertical,
        ref_bw=ref_bw,
        muscle_names=np.array(muscle_names),
        dof_names=np.array(dof_names),
        subject_mass_kg=SUBJECT_MASS_KG,
    )
    print(f"    ✓ {out_path.name}")
    return True

def main():
    if len(sys.argv) > 1:
        trial_names = [sys.argv[1]]
    else:
        id_files = sorted(PROC_DIR.glob("*_id.sto"))
        trial_names = [p.stem.replace('_id', '') for p in id_files]

    if not trial_names:
        print(f"[!] В {PROC_DIR} нет *_id.sto — сначала run_pipeline.py")
        return

    print(f"К подготовке: {len(trial_names)} триалов")
    ok = 0
    for name in trial_names:
        try:
            if prepare_trial(name):
                ok += 1
        except Exception as e:
            print(f"    [!] Ошибка: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✓ Подготовлено: {ok} / {len(trial_names)}")
    print("Дальше: conda activate opt && python solve_optimization.py")

if __name__ == "__main__":
    main()