"""
Запуск оптимизации на данных из оригинальной статьи Rajagopal 2016.

Данные: ExpData/motion_capture_walk.trc + grf_walk.mot
Модель: Rajagopal_subject_walk.osim (масштабированная под субъекта статьи)
Нет eTibia → сравнение формы кривой + физиологичность результата.

conda activate opensim && python run_rajagopal_walk.py
"""

import tempfile, time
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import opensim as osim
    OPENSIM = True
except ImportError:
    OPENSIM = False

RAJAGOPAL_DIR = Path("/Users/ignorance/Documents/OpenSim/4.5/Models/Rajagopal")
MODEL_PATH    = "/Users/ignorance/PycharmProjects/diplom4ik/Rajagopal_subject_walk.osim"
TRC_PATH      = RAJAGOPAL_DIR / "ExpData/motion_capture_walk.trc"
GRF_PATH      = RAJAGOPAL_DIR / "ExpData/grf_walk.mot"
OUT_DIR       = Path("/Users/ignorance/PycharmProjects/diplom4ik/processed_walk")
OUT_DIR.mkdir(exist_ok=True)

BODY_MASS = 75.337  # кг (субъект из статьи Rajagopal)

# Маппинг маркёров: TRC → Модель
MARKER_MAP = {
    'R.Shoulder': 'RACR',   'L.Shoulder': 'LACR',
    'R.ASIS': 'RASI',       'L.ASIS': 'LASI',
    'R.PSIS': 'RPSI',       'L.PSIS': 'LPSI',
    'R.Knee': 'RLFC',       'L.Knee': 'LLFC',
    'R.Ankle': 'RLMAL',     'L.Ankle': 'LLMAL',
    'R.Heel': 'RCAL',       'L.Heel': 'LCAL',
    'R.MT5': 'RMT5',        'L.MT5': 'LMT5',
    'R.Toe': 'RTOE',        'L.Toe': 'LTOE',
    'R.TH1': 'RTH1',        'R.TH2': 'RTH2',      'R.TH3': 'RTH3',
    'L.TH1': 'LTH1',        'L.TH2': 'LTH2',      'L.TH3': 'LTH3',
    'R.SH1': 'RTB1',        'R.SH2': 'RTB2',      'R.SH3': 'RTB3',
    'L.SH1': 'LTB1',        'L.SH2': 'LTB2',      'L.SH3': 'LTB3',
    'R.Elbow': 'RLEL',      'L.Elbow': 'LLEL',
    'R.Wrist': 'RFAsuperior', 'L.Wrist': 'LFAsuperior',
    'R.Forearm': 'RFAradius', 'L.Forearm': 'LFAradius',
    'R.Clavicle': 'CLAV',
}

LOWER_LIMB_DOFS = [
    'hip_flexion_r','hip_adduction_r','hip_rotation_r',
    'knee_angle_r','ankle_angle_r','subtalar_angle_r',
    'hip_flexion_l','hip_adduction_l','hip_rotation_l',
    'knee_angle_l','ankle_angle_l','subtalar_angle_l',
]


def rename_trc(src: Path, dst: Path) -> int:
    with open(src) as f:
        lines = f.readlines()
    parts = lines[3].rstrip('\n').split('\t')
    n = 0
    for i, p in enumerate(parts):
        s = p.strip()
        if s in MARKER_MAP:
            parts[i] = p.replace(s, MARKER_MAP[s])
            n += 1
    lines[3] = '\t'.join(parts) + '\n'
    with open(dst, 'w') as f:
        f.writelines(lines)
    return n


def load_sto(path):
    with open(path) as f:
        lines = f.readlines()
    he = next(i+1 for i,l in enumerate(lines) if l.strip().lower() == 'endheader')
    return pd.read_csv(path, sep=r'\s+', skiprows=he)


def run_ik(model_path, trc_path, out_mot):
    print("  IK...", end=" ", flush=True)
    with tempfile.NamedTemporaryFile(suffix='.trc', delete=False, mode='w') as tmp:
        tmp_path = Path(tmp.name)
    n = rename_trc(trc_path, tmp_path)
    print(f"({n} маркёров переименовано)", end=" ", flush=True)

    model = osim.Model(model_path)
    model.initSystem()
    with open(tmp_path) as f:
        times = [float(l.split('\t')[1]) for l in f.readlines()[5:]
                 if len(l.split('\t')) > 1 and l.split('\t')[1].replace('.','').isdigit()]
    t_start, t_end = times[0], times[-1]

    ik = osim.InverseKinematicsTool()
    ik.setModel(model)
    ik.setMarkerDataFileName(str(tmp_path))
    ik.setStartTime(t_start)
    ik.setEndTime(t_end)
    ik.setOutputMotionFileName(str(out_mot))
    ik.set_report_errors(False)
    ik.run()
    tmp_path.unlink(missing_ok=True)
    print(f"✓")


def run_id(model_path, mot_path, grf_mot_path, out_sto):
    print("  ID...", end=" ", flush=True)

    # GRF mot уже в правильном формате — нужен только XML ExternalLoads
    grf_mot = load_sto(grf_mot_path)
    grf_cols = [c for c in grf_mot.columns if c != 'time']
    # Определить тела из имён колонок
    bodies = {}
    for c in grf_cols:
        if '_v' in c or '_p' in c or '_torque' in c or '_t' in c:
            parts = c.split('_')
            side = 'r' if '_r_' in c else 'l'
            if side not in bodies:
                bodies[side] = f'calcn_{side}'

    forces_xml = ""
    for side, body in bodies.items():
        fl = f'ground_force_{side}'
        tl = f'ground_torque_{side}'
        forces_xml += f"""
        <ExternalForce name="{fl}">
            <isDisabled>false</isDisabled>
            <applied_to_body>{body}</applied_to_body>
            <force_expressed_in_body>ground</force_expressed_in_body>
            <point_expressed_in_body>ground</point_expressed_in_body>
            <force_identifier>{fl}_v</force_identifier>
            <point_identifier>{fl}_p</point_identifier>
            <torque_identifier>{tl}</torque_identifier>
        </ExternalForce>"""

    xml_path = OUT_DIR / "grf_walk.xml"
    xml_path.write_text(f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
  <ExternalLoads name="external_loads">
    <objects>{forces_xml}</objects>
    <groups />
    <datafile>{grf_mot_path}</datafile>
    <external_loads_model_kinematics_file />
    <lowpass_cutoff_frequency_for_load_kinematics>-1</lowpass_cutoff_frequency_for_load_kinematics>
  </ExternalLoads>
</OpenSimDocument>""")

    model = osim.Model(model_path)
    model.initSystem()
    mot = osim.Storage(str(mot_path))

    id_tool = osim.InverseDynamicsTool()
    id_tool.setModel(model)
    id_tool.setCoordinatesFileName(str(mot_path))
    id_tool.setExternalLoadsFileName(str(xml_path))
    id_tool.setStartTime(mot.getFirstTime())
    id_tool.setEndTime(mot.getLastTime())
    id_tool.setOutputGenForceFileName(out_sto.name)
    id_tool.setResultsDir(str(OUT_DIR))
    id_tool.setLowpassCutoffFrequency(6.0)
    id_tool.run()
    print("✓")


def extract_model_data(dof_names, q_traj, q_columns):
    from prepare_data import extract_model_data as _extract
    import sys
    sys.path.insert(0, '/Users/ignorance/PycharmProjects/diplom4ik')
    # Используем prepare_data но с walk моделью
    import prepare_data as pd_module
    old_model = pd_module.MODEL_PATH
    pd_module.MODEL_PATH = MODEL_PATH
    result = _extract(dof_names, q_traj, q_columns)
    pd_module.MODEL_PATH = old_model
    return result


def main():
    if not OPENSIM:
        print("❌ OpenSim не установлен")
        return

    print("=== Rajagopal Walk: IK → ID → Оптимизация ===\n")

    mot_path = OUT_DIR / "walk_ik.mot"
    id_path  = OUT_DIR / "walk_id.sto"

    # ── IK ──────────────────────────────────────────────────────
    if not mot_path.exists():
        run_ik(MODEL_PATH, TRC_PATH, mot_path)
    else:
        print("  IK: уже есть")

    # Проверить углы
    df_ik = load_sto(mot_path)
    print(f"  Кадров: {len(df_ik)}")
    for dof in ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']:
        if dof in df_ik.columns:
            print(f"  {dof}: {df_ik[dof].min():.1f}° → {df_ik[dof].max():.1f}°")

    # ── ID ──────────────────────────────────────────────────────
    if not id_path.exists():
        run_id(MODEL_PATH, mot_path, GRF_PATH, id_path)
    else:
        print("  ID: уже есть")

    df_id = load_sto(id_path)
    id_cols = [f'{d}_moment' for d in LOWER_LIMB_DOFS if f'{d}_moment' in df_id.columns]
    id_torques = df_id[id_cols].values
    time_id    = df_id['time'].values
    dof_names  = [c.replace('_moment','') for c in id_cols]

    print(f"\n  ID моменты: {id_torques.min():.0f} → {id_torques.max():.0f} N·м")

    # Интерполировать IK на сетку ID
    q_cols  = [c for c in df_ik.columns if c != 'time']
    q_traj  = df_ik[q_cols].values
    time_ik = df_ik['time'].values
    if not np.allclose(time_ik, time_id, atol=1e-3):
        q_traj = np.column_stack([
            np.interp(time_id, time_ik, q_traj[:,i])
            for i in range(q_traj.shape[1])
        ])

    # ── Момент плечи из модели ───────────────────────────────────
    print("\n  Извлечение моментных плеч...")
    import sys; sys.path.insert(0, '/Users/ignorance/PycharmProjects/diplom4ik')
    from prepare_data import extract_model_data as _extract
    import prepare_data as pdm
    pdm.MODEL_PATH = MODEL_PATH
    max_forces, moment_arms, muscle_names, knee_mask = _extract(
        dof_names, q_traj, q_cols)

    # GRF вертикальная
    df_grf = load_sto(GRF_PATH)
    vy_cols = [c for c in df_grf.columns if 'vy' in c.lower()]
    grf_v = np.interp(time_id, df_grf['time'].values,
                      df_grf[vy_cols].sum(axis=1).values) if vy_cols else np.zeros(len(time_id))

    # ── Оптимизация ─────────────────────────────────────────────
    from static_optimization import (DifferentiableStaticOptimizer,
                                     ModelParams, OptimizationConfig, CostFunction)
    n_m, n_d = len(max_forces), id_torques.shape[1]
    model_p = ModelParams(n_muscles=n_m, n_dof=n_d,
                          max_isometric_force=max_forces,
                          moment_arms=moment_arms)

    # Проверить дефицит
    max_deficit = max(
        max(np.maximum(np.abs(id_torques[:,d]) -
            np.array([np.abs(moment_arms[t,d,:]) @ max_forces for t in range(len(time_id))]), 0).max()
            for d in range(n_d)), 0)
    reserve = max(20.0, float(max_deficit) * 1.1)
    print(f"\n  Дефицит: {max_deficit:.0f} N·м  → резервы: {reserve:.0f} N·м")

    print(f"\n  Оптимизация (∑a³ + резервы {reserve:.0f} N·м)...")
    t0 = time.perf_counter()
    cfg = OptimizationConfig(CostFunction.SUM_CUBES,
                             reserve_actuator_weight=reserve, verbose=False)
    opt = DifferentiableStaticOptimizer(model_p, cfg)
    res = opt.solve_trajectory(moment_arms, max_forces, id_torques)
    t_total = time.perf_counter() - t0

    acts = res['activations']
    body_wt = BODY_MASS * 9.81
    mf = acts * max_forces[np.newaxis,:]
    knee_f = (grf_v + np.sum(mf[:,knee_mask], axis=1)) / body_wt

    print(f"  Время: {t_total:.2f}с  ({res['mean_frame_time']*1000:.1f} мс/кадр)")
    print(f"  Итерации IPOPT: {res['mean_iter_count']:.1f}")
    print(f"  Пик силы колена: {knee_f.max():.2f} BW  "
          f"(норма при ходьбе: 2–4 BW)")

    # ── График ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Rajagopal walk — результаты дифференцируемой оптимизации ∑a³\n"
                 f"(75 кг, 80 мышц, {len(time_id)} кадров, "
                 f"{t_total:.2f}с, {res['mean_frame_time']*1000:.1f} мс/кадр)",
                 fontsize=11, fontweight='bold')

    pct = (time_id - time_id[0]) / (time_id[-1] - time_id[0]) * 100

    # Углы суставов
    ax = axes[0,0]
    for dof, col, lbl in [
        ('hip_flexion_r',  '#e07b54', 'Сгибание бедра'),
        ('knee_angle_r',   '#5b8db8', 'Сгибание колена'),
        ('ankle_angle_r',  '#2e7d32', 'Голеностоп'),
    ]:
        if dof in df_ik.columns:
            vals = np.interp(time_id, time_ik, df_ik[dof].values)
            ax.plot(pct, vals, color=col, linewidth=2, label=lbl)
    ax.set_xlabel('% цикла ходьбы')
    ax.set_ylabel('Угол (°)')
    ax.set_title('Кинематика суставов')
    ax.legend(fontsize=8)
    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax.grid(alpha=0.3)

    # ID моменты
    ax = axes[0,1]
    for i, (dof, col) in enumerate([
        ('hip_flexion_r',  '#e07b54'),
        ('knee_angle_r',   '#5b8db8'),
        ('ankle_angle_r',  '#2e7d32'),
    ]):
        if dof in dof_names:
            idx = dof_names.index(dof)
            ax.plot(pct, id_torques[:,idx], color=col, linewidth=2, label=dof.replace('_r',''))
    ax.set_xlabel('% цикла ходьбы')
    ax.set_ylabel('Момент (N·м)')
    ax.set_title('Суставные моменты (ID)')
    ax.legend(fontsize=8)
    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax.grid(alpha=0.3)

    # Активации мышц
    ax = axes[1,0]
    im = ax.imshow(acts.T, aspect='auto', cmap='hot_r',
                   extent=[0, 100, n_m, 0], vmin=0, vmax=1)
    ax.set_xlabel('% цикла ходьбы')
    ax.set_ylabel('Мышца (индекс)')
    ax.set_title(f'Мышечные активации ({n_m} мышц)')
    plt.colorbar(im, ax=ax, label='Активация')

    # Сила колена
    ax = axes[1,1]
    ax.fill_between(pct, 0, knee_f, alpha=0.25, color='#2e7d32')
    ax.plot(pct, knee_f, color='#2e7d32', linewidth=2.5,
            label=f'∑a³ (пик={knee_f.max():.2f} BW)')
    # Зона нормы
    ax.axhspan(2.0, 4.0, alpha=0.1, color='blue', label='Норма ходьбы 2–4 BW')
    ax.set_xlabel('% цикла ходьбы')
    ax.set_ylabel('Контактная сила (BW)')
    ax.set_title('Контактная сила коленного сустава')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_fig = OUT_DIR / "rajagopal_walk_results.png"
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  График: {out_fig}")


if __name__ == '__main__':
    main()
