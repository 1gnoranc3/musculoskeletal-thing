
try:
    import opensim as osim
except ImportError:
    print("❌ opensim не установлен. conda install -c opensim-org opensim")
    raise

import tempfile
import pandas as pd
from pathlib import Path
from typing import Optional

MODEL_PATH   = "/Users/ignorance/PycharmProjects/diplom4ik/Rajagopal_DM_scaled.osim"
IK_TASKS_XML = "/Users/ignorance/PycharmProjects/diplom4ik/ik_tasks.xml"
DATA_DIR   = "/Users/ignorance/PycharmProjects/diplom4ik/Synchronized Motion Data/Overground Gait Trials"
OUTPUT_DIR = "/Users/ignorance/PycharmProjects/diplom4ik/processed"

MARKER_MAP = {
    'R.Asis': 'RASI', 'L.Asis': 'LASI', 'R.Psis': 'RPSI', 'L.Psis': 'LPSI',
    'Neck': 'C7', 'Sternum': 'CLAV',
    'R.Shoulder': 'RACR', 'L.Shoulder': 'LACR',
    'R.ShoulderAnterior': 'RASH', 'R.ShoulderPosterior': 'RPSH',
    'L.ShoulderAnterior': 'LASH', 'L.ShoulderPosterior': 'LPSH',
    'R.Elbow': 'RLEL', 'R.ElbowMedial': 'RMEL',
    'R.Wrist': 'RFAsuperior', 'R.Radius': 'RFAradius', 'R.Ulna': 'RFAulna',
    'L.Elbow': 'LLEL', 'L.ElbowMedial': 'LMEL',
    'L.Wrist': 'LFAsuperior', 'L.Radius': 'LFAradius', 'L.Ulna': 'LFAulna',
    'R.Thigh.Superior': 'RTH1', 'R.Thigh.Inferior': 'RTH2', 'R.Thigh.Lateral': 'RTH3',
    'R.Knee.Lateral': 'RLFC', 'R.Knee.Medial': 'RMFC',
    'R.Shank.Superior': 'RTB1', 'R.Shank.Inferior': 'RTB2', 'R.Shank.Lateral': 'RTB3',
    'R.Ankle.Lateral': 'RLMAL', 'RankleMedial': 'RMMAL',
    'R.Heel': 'RCAL', 'R.Toe': 'RTOE', 'R.Midfoot.Lateral': 'RMT5',
    'L.Thigh.Superior': 'LTH1', 'L.Thigh.Inferior': 'LTH2', 'L.Thigh.Lateral': 'LTH3',
    'L.Knee.Lateral': 'LLFC', 'L.Knee.Medial': 'LMFC',
    'L.Shank.Superior': 'LTB1', 'L.Shank.Inferior': 'LTB2', 'L.Shank.Lateral': 'LTB3',
    'L.Ankle.Lateral': 'LLMAL', 'L.Ankle.Medial': 'LMMAL',
    'L.Heel': 'LCAL', 'L.Toe': 'LTOE', 'L.Midfoot.Lateral': 'LMT5',
}

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def grf_csv_to_opensim(grf_csv_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Преобразовать Grand Challenge GRF CSV в OpenSim формат:
      - {name}_grf.sto  — данные сил (формат OpenSim Storage)
      - {name}_grf.xml  — ExternalLoads объект для ID Tool

    Grand Challenge формат:
      time(sec), Fx1, Fy1, Fz1, COPx1, COPy1, COPz1, Tz1, Fx2, Fy2, ..., Fx3, ...
      - 3 силовые платформы, до 2-х активны одновременно
      - Lab frame: X = вперёд, Y = вертикаль, Z = латераль
        Fz1 отрицателен при реакции вверх → инвертируем
      - COP в миллиметрах → делим на 1000
    """
    try:
        df = pd.read_csv(grf_csv_path, index_col=False)
    except Exception as e:
        print(f"    [!] Не удалось прочитать GRF: {e}")
        return None

    if df.index.dtype == float or df.index.dtype == object:
        try:
            float(str(df.index[0]))
            df = df.reset_index()
            df.rename(columns={'index': 'time(sec)'}, inplace=True)
        except (ValueError, TypeError):
            pass

    time_col = 'time(sec)' if 'time(sec)' in df.columns else df.columns[0]

    active_plates = []
    for i in [1, 2, 3]:
        fz_col = f'Fz{i}'
        if fz_col in df.columns and df[fz_col].abs().max() > 10:
            active_plates.append(i)

    if not active_plates:
        print(f"    [!] Нет активных платформ в {grf_csv_path.name}")
        return None

    name = grf_csv_path.stem.replace('_grf', '').replace('_GRF', '')

    sides = ['r', 'l']
    sto_data = {time_col: df[time_col]}
    ext_forces = []  # (force_label, point_label, torque_label, body)

    for idx, plate in enumerate(active_plates[:2]):
        side = sides[idx]
        body = f'calcn_{side}'
        fl = f'ground_force_{side}'
        tl = f'ground_torque_{side}'

        sto_data[f'{fl}_vx'] = df[f'Fy{plate}']            # lateral force: GC Y → OS X
        sto_data[f'{fl}_vy'] = df[f'Fz{plate}']            # vertical force: GC Z → OS Y
        sto_data[f'{fl}_vz'] = -df[f'Fx{plate}']           # forward force: -GC X → OS Z

        sto_data[f'{fl}_px'] = df[f'COPy{plate}'] / 1000.0
        sto_data[f'{fl}_py'] = (df[f'COPz{plate}'] / 1000.0) if f'COPz{plate}' in df.columns else 0.0
        sto_data[f'{fl}_pz'] = -df[f'COPx{plate}'] / 1000.0

        sto_data[f'{tl}_x'] = 0.0
        sto_data[f'{tl}_y'] = df[f'Tz{plate}'] if f'Tz{plate}' in df.columns else 0.0
        sto_data[f'{tl}_z'] = 0.0

        ext_forces.append((f'{fl}_v', f'{fl}_p', tl, body))

    sto_df = pd.DataFrame(sto_data)
    sto_cols = list(sto_df.columns)
    sto_cols[0] = 'time'
    n_cols = len(sto_cols)  # включая time

    sto_path = output_dir / f"{name}_grf.sto"
    with open(sto_path, 'w') as f:
        f.write(f"{name}_grf\nversion=1\n")
        f.write(f"nRows={len(sto_df)}\nnColumns={n_cols}\n")
        f.write("inDegrees=no\nendheader\n")
        f.write('\t'.join(sto_cols) + '\n')
        for row in sto_df.itertuples(index=False):
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')

    forces_xml = ""
    for force_id, point_id, torque_id, body in ext_forces:
        forces_xml += f"""
        <ExternalForce name="{force_id}">
            <isDisabled>false</isDisabled>
            <applied_to_body>{body}</applied_to_body>
            <force_expressed_in_body>ground</force_expressed_in_body>
            <point_expressed_in_body>ground</point_expressed_in_body>
            <force_identifier>{force_id}</force_identifier>
            <point_identifier>{point_id}</point_identifier>
            <torque_identifier>{torque_id}</torque_identifier>
        </ExternalForce>"""

    xml_content = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
    <ExternalLoads name="external_loads">
        <objects>{forces_xml}
        </objects>
        <groups />
        <datafile>{sto_path}</datafile>
        <external_loads_model_kinematics_file />
        <lowpass_cutoff_frequency_for_load_kinematics>-1</lowpass_cutoff_frequency_for_load_kinematics>
    </ExternalLoads>
</OpenSimDocument>
"""
    xml_path = output_dir / f"{name}_grf.xml"
    xml_path.write_text(xml_content)

    print(f"    GRF: → {sto_path.name} + {xml_path.name}  (платформы: {active_plates[:2]})")
    return xml_path

def rename_trc_markers(input_path: Path, output_path: Path) -> int:
    """
    Rename markers according to MARKER_MAP AND transform coordinates from
    Grand Challenge frame (X=anterior, Y=lateral, Z=up) to OpenSim frame
    (X=lateral, Y=up, Z=anterior): new(X,Y,Z) = old(Y,Z,X).
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    parts = lines[3].rstrip('\n').split('\t')
    renamed = 0
    for i, part in enumerate(parts):
        stripped = part.strip()
        if stripped in MARKER_MAP:
            parts[i] = part.replace(stripped, MARKER_MAP[stripped])
            renamed += 1
    lines[3] = '\t'.join(parts) + '\n'

    new_lines = lines[:5]
    for line in lines[5:]:
        cols = line.rstrip('\n').split('\t')
        if len(cols) < 3:
            new_lines.append(line)
            continue
        out = cols[:2]
        i = 2
        while i + 2 < len(cols):
            x_str, y_str, z_str = cols[i], cols[i+1], cols[i+2]
            try:
                x_val = float(x_str)
                y_val = float(y_str)
                new_x = f'{-y_val:.6f}'               # OS X = -GC Y
                new_y = z_str                          # OS Y =  GC Z
                new_z = f'{-x_val:.6f}'               # OS Z = -GC X
            except ValueError:
                new_x, new_y, new_z = x_str, y_str, z_str
            out.extend([new_x, new_y, new_z])
            i += 3
        out.extend(cols[i:])
        new_lines.append('\t'.join(out) + '\n')

    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    return renamed

def run_ik(model_path: str, trc_path: Path, output_mot: str):
    print(f"  IK: {trc_path.name} → {Path(output_mot).name}")
    with tempfile.NamedTemporaryFile(suffix='.trc', delete=False, mode='w') as tmp:
        tmp_path = Path(tmp.name)
    try:
        n = rename_trc_markers(trc_path, tmp_path)
        print(f"    Маркеров переименовано: {n}")

        model = osim.Model(model_path)
        model.initSystem()

        t_start, t_end = None, None
        with open(tmp_path, 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                if len(cols) > 1:
                    try:
                        t = float(cols[1])
                        if t_start is None:
                            t_start = t
                        t_end = t
                    except ValueError:
                        continue

        if t_start is None:
            raise ValueError(f"Не удалось прочитать время из {trc_path}")

        ik = osim.InverseKinematicsTool()
        ik.setModel(model)
        ik.setMarkerDataFileName(str(tmp_path))
        ik.setStartTime(t_start)
        ik.setEndTime(t_end)
        ik.setOutputMotionFileName(output_mot)
        ik.set_report_errors(True)
        if Path(IK_TASKS_XML).exists():
            ik.set_IKTaskSet(osim.IKTaskSet(IK_TASKS_XML))
        ik.run()
        print(f"    ✓ {output_mot}")
    finally:
        tmp_path.unlink(missing_ok=True)

def run_id(model_path: str, mot_path: str, grf_xml_path: str, output_sto: str):
    print(f"  ID: {Path(mot_path).name} → {Path(output_sto).name}")

    model = osim.Model(model_path)
    model.initSystem()

    mot     = osim.Storage(mot_path)
    t_start = mot.getFirstTime()
    t_end   = mot.getLastTime()

    id_tool = osim.InverseDynamicsTool()
    id_tool.setModel(model)
    id_tool.setCoordinatesFileName(mot_path)
    id_tool.setExternalLoadsFileName(grf_xml_path)
    id_tool.setStartTime(t_start)
    id_tool.setEndTime(t_end)
    id_tool.setOutputGenForceFileName(Path(output_sto).name)  # имя файла, не путь
    id_tool.setResultsDir(OUTPUT_DIR)
    id_tool.setLowpassCutoffFrequency(6.0)
    id_tool.run()
    print(f"    ✓ {output_sto}")

def find_grf_csv(trc_dir: Path, trial_name: str) -> Optional[Path]:
    base = trial_name.replace('_new', '')
    for name in [f"{base}_grf.csv", f"{base}_GRF.csv",
                 f"{trial_name}_grf.csv", f"{trial_name}_GRF.csv"]:
        p = trc_dir / name
        if p.exists():
            return p
    return None

def main():
    data_dir   = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    trc_files = sorted(data_dir.rglob("*.trc"))
    if not trc_files:
        print(f"[!] Не найдено .trc файлов в {DATA_DIR}")
        return

    print(f"Найдено {len(trc_files)} триалов\n")
    ok, skipped, failed = 0, 0, 0

    for trc_path in trc_files:
        trial_name = trc_path.stem
        print(f"── Триал: {trial_name}")

        mot_path = str(output_dir / f"{trial_name}_ik.mot")
        sto_path = str(output_dir / f"{trial_name}_id.sto")

        if Path(sto_path).exists():
            print(f"  пропускаем (уже есть)")
            skipped += 1
            continue

        if not Path(mot_path).exists():
            try:
                run_ik(MODEL_PATH, trc_path, mot_path)
            except Exception as e:
                print(f"  [!] IK упал: {e}")
                failed += 1
                continue
        else:
            print(f"  IK: уже есть {Path(mot_path).name}, пропускаем")

        grf_csv = find_grf_csv(trc_path.parent, trial_name)
        if grf_csv is None:
            print(f"  [!] GRF CSV не найден")
            failed += 1
            continue

        base = trial_name.replace('_new', '')
        grf_xml = output_dir / f"{base}_grf.xml"
        if not grf_xml.exists():
            grf_xml = grf_csv_to_opensim(grf_csv, output_dir)
            if grf_xml is None:
                failed += 1
                continue
        else:
            print(f"    GRF XML: кэш {grf_xml.name}")

        try:
            run_id(MODEL_PATH, mot_path, str(grf_xml), sto_path)
            ok += 1
        except Exception as e:
            print(f"  [!] ID упал: {e}")
            failed += 1
            continue

        print()

    print("─" * 40)
    print(f"Готово: ✓ {ok}  пропущено: {skipped}  ошибок: {failed}")
    print(f"Результаты в: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()