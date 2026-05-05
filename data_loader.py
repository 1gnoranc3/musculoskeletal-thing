"""
Загрузка и предобработка данных из датасетов Grand Challenge и CAMS-Knee.
Форматы: .sto/.mot (OpenSim) и .csv.

Структура OpenSim STO/MOT:
    # заголовок
    name <имя>
    nRows <N>
    nColumns <M>
    endheader
    time  col1  col2 ...
    0.0   ...
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class GaitCycleData:
    """Данные одного цикла ходьбы."""
    time: np.ndarray              # (n_frames,) [s]
    id_torques: np.ndarray        # (n_frames, n_dof) — из Inverse Dynamics [N·m]
    grf_vertical: np.ndarray      # (n_frames,) [N]
    knee_contact_ref: Optional[np.ndarray] = None  # эталон JRA [BW] если есть
    subject_mass_kg: float = 75.0
    n_frames: int = 0

    def __post_init__(self):
        self.n_frames = len(self.time)


def load_opensim_sto(filepath: str | Path) -> pd.DataFrame:
    """
    Загрузить файл OpenSim .sto или .mot в DataFrame.
    Пропускает все строки заголовка до 'endheader'.
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Найти строку endheader
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i + 1
            break

    df = pd.read_csv(filepath, sep=r"\s+", skiprows=header_end)
    return df


def load_csv_data(filepath: str | Path) -> pd.DataFrame:
    """Загрузить CSV-данные (Grand Challenge формат)."""
    return pd.read_csv(filepath)


def extract_gait_cycle(
    id_df: pd.DataFrame,
    grf_df: pd.DataFrame,
    dof_columns: list[str],
    grf_column: str = "ground_force_vy",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> GaitCycleData:
    """
    Извлечь один цикл ходьбы из данных обратной динамики и GRF.

    Args:
        id_df:       DataFrame с моментами из Inverse Dynamics
        grf_df:      DataFrame с силами реакции опоры
        dof_columns: список названий колонок DOF-моментов
        grf_column:  название колонки вертикальной GRF
        t_start:     начало цикла [s], None = с начала
        t_end:       конец цикла  [s], None = до конца
    """
    # Временная фильтрация
    if t_start is not None:
        id_df  = id_df[id_df["time"] >= t_start]
        grf_df = grf_df[grf_df["time"] >= t_start]
    if t_end is not None:
        id_df  = id_df[id_df["time"] <= t_end]
        grf_df = grf_df[grf_df["time"] <= t_end]

    # Совместить по времени через интерполяцию
    time_common = id_df["time"].values
    grf_interp  = np.interp(time_common, grf_df["time"].values, grf_df[grf_column].values)

    id_torques = id_df[dof_columns].values  # (n_frames, n_dof)

    return GaitCycleData(
        time=time_common,
        id_torques=id_torques,
        grf_vertical=grf_interp,
    )


def load_etibia_reference(
    filepath: str | Path,
    body_mass_kg: float = 64.0,    # масса субъекта DM = 64 кг (Grand Challenge документация)
    min_duration_sec: float = 0.5, # минимальная длина непрерывного отрезка
) -> list[dict]:
    """
    Загрузить эталонные контактные силы колена из eTibia Data (Grand Challenge).

    Файл содержит несколько несвязанных временных отрезков (циклов ходьбы),
    склеенных в один CSV. Функция разбивает их на отдельные циклы.

    Колонки: Time(sec), Fx, Fy, Fz, Tx, Ty, Tz, GON, GRFz
    Fz — вертикальная контактная сила колена [N], отрицательная = компрессия.

    Args:
        filepath:        путь к *_knee_forces.csv
        body_mass_kg:    масса тела субъекта для нормализации в BW
        min_duration_sec: отбрасывать отрезки короче этого значения

    Returns:
        список dict с ключами: time, fz_n, fz_bw, peak_bw, trial_name
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.sort_values("Time(sec)").reset_index(drop=True)

    body_weight = body_mass_kg * 9.81

    # Найти границы непрерывных отрезков по скачкам времени
    dt = df["Time(sec)"].diff()
    median_dt = dt.median()
    # Скачок > 10 × медианного шага = новый отрезок
    break_indices = [0] + list(df.index[dt > median_dt * 10]) + [len(df)]

    cycles = []
    for i in range(len(break_indices) - 1):
        segment = df.iloc[break_indices[i]:break_indices[i+1]].copy()
        if len(segment) < 2:
            continue
        duration = segment["Time(sec)"].max() - segment["Time(sec)"].min()
        if duration < min_duration_sec:
            continue

        fz = segment["Fz"].values
        # Fz отрицательная при компрессии — берём абсолютное значение
        fz_abs = np.abs(fz)

        cycles.append({
            "time":      segment["Time(sec)"].values,
            "fz_n":      fz_abs,
            "fz_bw":     fz_abs / body_weight,
            "peak_bw":   float(fz_abs.max() / body_weight),
            "trial_name": filepath.stem,
            "segment_idx": i,
        })

    return cycles


def load_etibia_directory(
    directory: str | Path,
    pattern: str = "*_knee_forces.csv",
    body_mass_kg: float = 64.0,
) -> dict[str, list[dict]]:
    """
    Загрузить все eTibia файлы из папки.

    Returns:
        dict: {имя_файла: [циклы]}
    """
    directory = Path(directory)
    result = {}
    for f in sorted(directory.glob(pattern)):
        cycles = load_etibia_reference(f, body_mass_kg=body_mass_kg)
        if cycles:
            result[f.stem] = cycles
            print(f"  {f.name}: {len(cycles)} циклов, "
                  f"пик {max(c['peak_bw'] for c in cycles):.2f} BW")
    return result


def generate_synthetic_data(
    n_frames: int = 100,
    n_muscles: int = 44,
    n_dof: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Сгенерировать синтетические данные для тестирования алгоритма.
    Имитирует один цикл ходьбы модели Rajagopal 2016.

    Returns:
        moment_arms:  (n_frames, n_dof, n_muscles)
        max_forces:   (n_muscles,)
        id_torques:   (n_frames, n_dof)
        grf_vertical: (n_frames,)
    """
    rng = np.random.default_rng(seed)

    # Моментные плечи: физиологически реалистичный диапазон [−0.08; 0.08] м
    moment_arms = rng.uniform(-0.08, 0.08, (n_frames, n_dof, n_muscles))

    # Максимальные силы мышц нижних конечностей [200–4000 N] (Rajagopal 2016)
    max_forces = rng.uniform(200, 4000, n_muscles)

    # Суставные моменты из обратной динамики (гладкие кривые ходьбы)
    t = np.linspace(0, 2 * np.pi, n_frames)
    id_torques = np.zeros((n_frames, n_dof))
    for j in range(n_dof):
        amp   = rng.uniform(20, 120)
        phase = rng.uniform(0, np.pi)
        id_torques[:, j] = amp * np.sin(t + phase)

    # GRF: двойной пик (типичная кривая ходьбы, ~600–800 N)
    grf_vertical = (
        600 * np.exp(-((t - 1.5) ** 2) / 0.3)
        + 700 * np.exp(-((t - 4.5) ** 2) / 0.3)
    )

    return moment_arms, max_forces, id_torques, grf_vertical