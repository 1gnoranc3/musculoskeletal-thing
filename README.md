# Дифференцируемый алгоритм расчёта суставных нагрузок нижних конечностей

ВКР: Емельянчик А.А., СПбГЭТУ «ЛЭТИ», 2026.

Дифференцируемая статическая оптимизация мышечных активаций на базе CasADi + IPOPT.
Ускорение ~25× по сравнению с классической реализацией (scipy SLSQP) при 80 мышцах, 12 DOF.

## Требования

Два conda-окружения:

```bash
# Окружение 1 — OpenSim (IK, ID, подготовка данных)
conda create -n opensim python=3.10
conda install -c opensim-org opensim

# Окружение 2 — оптимизация (CasADi, numpy, scipy)
conda create -n opt python=3.11
pip install casadi numpy scipy matplotlib pandas
```

## Структура проекта

```
static_optimization.py     — ядро алгоритма (NLP, CasADi, warm-start)
joint_reaction.py          — расчёт контактных нагрузок суставов
prepare_data.py            — извлечение данных модели из OpenSim
run_pipeline.py            — пайплайн IK + ID для Grand Challenge данных
solve_optimization.py      — оптимизация на Grand Challenge триалах
benchmark.py               — сравнение scipy vs CasADi
run_rajagopal_walk.py      — демонстрация на данных Rajagopal 2016
ik_tasks.xml               — веса маркеров для IK
```

## Запуск

### Шаг 1. Подготовка данных Grand Challenge (окружение opensim)

```bash
conda activate opensim
python run_pipeline.py          # IK + ID для всех триалов
python prepare_data.py          # извлечение момент-плеч из модели
```

Результаты сохраняются в `processed/`.

### Шаг 2. Оптимизация (окружение opt)

```bash
conda activate opt
python solve_optimization.py    # расчёт активаций и нагрузок
python benchmark.py             # сравнение со scipy baseline
```

### Демонстрация на данных Rajagopal 2016

```bash
conda activate opensim
python run_rajagopal_walk.py
```

Результат: `processed_walk/rajagopal_walk_results.png`

## Ключевые результаты

| Метод | мс/кадр | Ускорение |
|---|---|---|
| scipy SLSQP ∑a² (baseline) | 79.2 | 1× |
| CasADi IPOPT ∑a³ (алгоритм) | 3.2 | **25×** |

## Данные

- **Grand Challenge**: `Synchronized Motion Data/` — захват движения субъекта DM,
  eTibia in vivo измерения, 3 силовые платформы Bertec
- **Rajagopal 2016**: `~/Documents/OpenSim/4.5/Models/Rajagopal/ExpData/`
- **Модели**: `Rajagopal_DM_scaled.osim`, `Rajagopal_subject_walk.osim`

## Алгоритм

Задача на каждом кадре t:

```
min  Σ aᵢ³
s.t. R̃(t) · a + τ_res = τ_ID(t)
     0 ≤ aᵢ ≤ 1
     |τ_res_j| ≤ w_res

где R̃(t) = R(t) · diag(F₀) — предвычислено однократно
```

Ключевые оптимизации:
1. **Предвычисление R̃ = R·F₀** — сокращает символьный граф CasADi
2. **Warm-start множителей Лагранжа** между кадрами — 12 итераций vs 25–40 при cold start
