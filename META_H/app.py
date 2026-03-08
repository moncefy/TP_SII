from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Optimization Problem Initialization", layout="wide")


def f1(population: np.ndarray) -> np.ndarray:
    return np.sum(population**2, axis=1)


def f2(population: np.ndarray) -> np.ndarray:
    abs_population = np.abs(population)
    return np.sum(abs_population, axis=1) + np.prod(abs_population, axis=1)


def f3(population: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(population, axis=1)
    return np.sum(cumulative**2, axis=1)


def f4(population: np.ndarray) -> np.ndarray:
    return np.max(np.abs(population), axis=1)


def f5(population: np.ndarray) -> np.ndarray:
    x_i = population[:, :-1]
    x_next = population[:, 1:]
    return np.sum(100 * (x_next - x_i**2) ** 2 + (x_i - 1) ** 2, axis=1)


def f6(population: np.ndarray) -> np.ndarray:
    return np.sum(np.floor(population + 0.5) ** 2, axis=1)


def f7(population: np.ndarray) -> np.ndarray:
    indexes = np.arange(1, population.shape[1] + 1)
    return np.sum(indexes * (population**4), axis=1) + np.random.rand(population.shape[0])


def f8(population: np.ndarray) -> np.ndarray:
    return np.sum(-population * np.sin(np.sqrt(np.abs(population))), axis=1)


def f9(population: np.ndarray) -> np.ndarray:
    return np.sum(population**2 - 10 * np.cos(2 * np.pi * population) + 10, axis=1)


def f11(population: np.ndarray) -> np.ndarray:
    indexes = np.arange(1, population.shape[1] + 1)
    sum_part = np.sum(population**2, axis=1) / 4000
    prod_part = np.prod(np.cos(population / np.sqrt(indexes)), axis=1)
    return 1 + sum_part - prod_part


FUNCTIONS = {
    "F1-UM": {
        "short": "F1",
        "range": (-100.0, 100.0),
        "latex": r"f(x)=\sum_{i=1}^{D}x_i^2",
        "fn": f1,
    },
    "F2-UM": {
        "short": "F2",
        "range": (-10.0, 10.0),
        "latex": r"f(x)=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|",
        "fn": f2,
    },
    "F3-UM": {
        "short": "F3",
        "range": (-100.0, 100.0),
        "latex": r"f(x)=\sum_{i=1}^{D}\left(\sum_{j=1}^{i}x_j\right)^2",
        "fn": f3,
    },
    "F4-UM": {
        "short": "F4",
        "range": (-100.0, 100.0),
        "latex": r"f(x)=\max_{1\le i\le D}|x_i|",
        "fn": f4,
    },
    "F5-UM": {
        "short": "F5",
        "range": (-30.0, 30.0),
        "latex": r"f(x)=\sum_{i=1}^{D-1}\left[100(x_{i+1}-x_i^2)^2+(x_i-1)^2\right]",
        "fn": f5,
    },
    "F6-UM": {
        "short": "F6",
        "range": (-100.0, 100.0),
        "latex": r"f(x)=\sum_{i=1}^{D}\lfloor x_i+0.5\rfloor^2",
        "fn": f6,
    },
    "F7-UM": {
        "short": "F7",
        "range": (-128.0, 128.0),
        "latex": r"f(x)=\sum_{i=1}^{D}ix_i^4+rand(0,1)",
        "fn": f7,
    },
    "F8-MM": {
        "short": "F8",
        "range": (-500.0, 500.0),
        "latex": r"f(x)=\sum_{i=1}^{D}-x_i\sin(\sqrt{|x_i|})",
        "fn": f8,
    },
    "F9-MM": {
        "short": "F9",
        "range": (-5.12, 5.12),
        "latex": r"f(x)=\sum_{i=1}^{D}\left[x_i^2-10\cos(2\pi x_i)+10\right]",
        "fn": f9,
    },
    "F11-MM": {
        "short": "F11",
        "range": (-600.0, 600.0),
        "latex": r"f(x)=1+\frac{1}{4000}\sum_{i=1}^{D}x_i^2-\prod_{i=1}^{D}\cos\left(\frac{x_i}{\sqrt{i}}\right)",
        "fn": f11,
    },
}


def evaluate_population(population: np.ndarray, function_key: str) -> np.ndarray:
    return FUNCTIONS[function_key]["fn"](population)


def parse_candidate(candidate_text: str) -> np.ndarray:
    normalized = candidate_text.replace(";", ",")
    number_tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", normalized)
    values = [float(v) for v in number_tokens]
    if not values:
        raise ValueError("Candidate is empty.")
    return np.array(values, dtype=float)


def parse_population_dataframe(df: pd.DataFrame) -> np.ndarray:
    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="any")
    if numeric_df.empty:
        raise ValueError("No numeric values found.")
    return numeric_df.to_numpy(dtype=float)


def load_population_from_csv(file_obj) -> np.ndarray:
    parsers = [
        {"sep": ";", "header": None},
        {"sep": ",", "header": None},
        {"sep": None, "header": None, "engine": "python"},
    ]
    for parser in parsers:
        file_obj.seek(0)
        try:
            df = pd.read_csv(file_obj, **parser)
            arr = parse_population_dataframe(df)
            if arr.shape[1] >= 2:
                return arr
        except Exception:
            continue
    raise ValueError("Unable to parse CSV file.")


def get_grid(function_key: str, dimension: int, n: int = 110) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum, maximum = FUNCTIONS[function_key]["range"]
    x = np.linspace(minimum, maximum, n)
    y = np.linspace(minimum, maximum, n)
    xx, yy = np.meshgrid(x, y)
    base = np.zeros((n * n, dimension), dtype=float)
    base[:, 0] = xx.ravel()
    base[:, 1] = yy.ravel()
    zz = evaluate_population(base, function_key).reshape(xx.shape)
    return xx, yy, zz


def plot_surface(function_key: str, dimension: int) -> plt.Figure:
    xx, yy, zz = get_grid(function_key, dimension)
    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(f"Function ({function_key})", fontsize=10)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("f")
    fig.tight_layout()
    return fig


def plot_contour_with_population(population: np.ndarray, function_key: str, best_idx: int, dimension: int) -> plt.Figure:
    xx, yy, zz = get_grid(function_key, dimension)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.contour(xx, yy, zz, levels=18, cmap="viridis", linewidths=1)
    ax.scatter(population[:, 0], population[:, 1], s=10, c="black")
    if st.session_state.best_history:
        h = np.array(st.session_state.best_history)
        ax.scatter(h[:, 0], h[:, 1], s=20, c="#f59e0b", alpha=0.6)
    ax.scatter(population[best_idx, 0], population[best_idx, 1], s=60, c="red")
    ax.set_title(f"Search History ({function_key})", fontsize=10)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    fig.tight_layout()
    return fig


def run_pso_pseudocode(
    function_key: str,
    dimension: int,
    population_size: int,
    max_iterations: int,
    w: float,
    c1: float,
    c2: float,
    seed: int | None,
    initial_population: np.ndarray | None = None,
    enable_early_stopping: bool = True,
) -> dict:
    if seed is not None:
        np.random.seed(seed)

    minimum, maximum = FUNCTIONS[function_key]["range"]

    if initial_population is not None:
        population = initial_population.astype(float).copy()
        population_size = population.shape[0]
        dimension = population.shape[1]
    else:
        population = np.random.uniform(minimum, maximum, size=(population_size, dimension))

    # Canonical PSO uses a velocity vector and then updates position with x <- x + v.
    velocity_scale = 0.1 * (maximum - minimum)
    velocity = np.random.uniform(-velocity_scale, velocity_scale, size=(population_size, dimension))

    initial_population = population.copy()
    personal_best = population.copy()

    fitness = evaluate_population(population, function_key)
    personal_best_fitness = fitness.copy()

    best_idx = int(np.argmin(fitness))
    global_best = population[best_idx].copy()
    global_best_fitness = float(fitness[best_idx])

    initial_best = float(np.min(fitness))
    initial_worst = float(np.max(fitness))

    history_population = [population.copy()]
    history_best = [global_best_fitness]
    history_avg = [float(np.mean(fitness))]
    first_solution_x1 = [float(population[0, 0])]
    best_positions = [global_best[:2].copy()]

    no_change_count = 0
    stagnation_iteration = max_iterations
    last_best_fitness = global_best_fitness

    for t in range(1, max_iterations + 1):
        for i in range(population_size):
            rand_1 = np.random.rand(dimension)
            rand_2 = np.random.rand(dimension)
            velocity[i] = (
                w * velocity[i]
                + c1 * rand_1 * (global_best - population[i])
                + c2 * rand_2 * (personal_best[i] - population[i])
            )
            population[i] = population[i] + velocity[i]
            population[i] = np.clip(population[i], minimum, maximum)

        fitness = evaluate_population(population, function_key)

        for i in range(population_size):
            if fitness[i] < global_best_fitness:
                global_best = population[i].copy()
                global_best_fitness = float(fitness[i])

            if fitness[i] < personal_best_fitness[i]:
                personal_best[i] = population[i].copy()
                personal_best_fitness[i] = fitness[i]

        history_population.append(population.copy())
        history_best.append(global_best_fitness)
        history_avg.append(float(np.mean(fitness)))
        first_solution_x1.append(float(population[0, 0]))
        best_positions.append(global_best[:2].copy())

        if np.isclose(global_best_fitness, last_best_fitness):
            no_change_count += 1
        else:
            no_change_count = 0
            last_best_fitness = global_best_fitness

        if enable_early_stopping and no_change_count >= 3:
            stagnation_iteration = t
            break

    final_population = history_population[-1]
    final_fitness = evaluate_population(final_population, function_key)
    final_best_idx = int(np.argmin(final_fitness))
    all_positions_2d = np.vstack([pop[:, :2] for pop in history_population])

    return {
        "initial_population": initial_population,
        "initial_best": initial_best,
        "initial_worst": initial_worst,
        "final_population": final_population,
        "final_best_idx": final_best_idx,
        "final_best": float(np.min(final_fitness)),
        "global_best": global_best,
        "history_population": history_population,
        "history_best": history_best,
        "history_avg": history_avg,
        "first_solution_x1": first_solution_x1,
        "best_positions": np.array(best_positions),
        "all_positions_2d": all_positions_2d,
        "stagnation_iteration": stagnation_iteration,
    }


def plot_contour_population_state(
    function_key: str,
    dimension: int,
    population: np.ndarray,
    best_point: np.ndarray,
    title: str,
    extra_points: np.ndarray | None = None,
    trail_points: np.ndarray | None = None,
) -> plt.Figure:
    xx, yy, zz = get_grid(function_key, dimension)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.contour(xx, yy, zz, levels=18, cmap="viridis", linewidths=1)

    if extra_points is not None and len(extra_points) > 0:
        ax.scatter(extra_points[:, 0], extra_points[:, 1], s=8, c="black", alpha=0.7)

    if trail_points is not None and len(trail_points) > 0:
        ax.scatter(trail_points[:, 0], trail_points[:, 1], s=24, c="#f59e0b", alpha=0.85)

    ax.scatter(population[:, 0], population[:, 1], s=10, c="black")
    ax.scatter(best_point[0], best_point[1], s=50, c="red")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"$x_1$", fontsize=8)
    ax.set_ylabel(r"$x_2$", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def plot_simple_curve(y_values: list[float], color: str, title: str, y_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(range(len(y_values)), y_values, color=color, linewidth=1.6)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _pad_history(history: list[float], length: int) -> np.ndarray:
    arr = np.asarray(history, dtype=float)
    if arr.size == 0:
        return np.zeros(length, dtype=float)
    if arr.size >= length:
        return arr[:length]
    return np.pad(arr, (0, length - arr.size), mode="edge")


def run_multiple_pso_experiments(
    function_key: str,
    dimension: int,
    population_size: int,
    max_iterations: int,
    w: float,
    c1: float,
    c2: float,
    seed: int,
    runs: int,
) -> dict:
    run_results = []
    target_length = max_iterations + 1

    for run_id in range(runs):
        result = run_pso_pseudocode(
            function_key=function_key,
            dimension=dimension,
            population_size=population_size,
            max_iterations=max_iterations,
            w=w,
            c1=c1,
            c2=c2,
            seed=seed + run_id,
            initial_population=None,
            enable_early_stopping=False,
        )
        run_results.append(result)

    best_per_run = np.array([r["final_best"] for r in run_results], dtype=float)
    best_idx = int(np.argmin(best_per_run))

    mean_best_curve = np.mean(
        np.vstack([_pad_history(r["history_best"], target_length) for r in run_results]),
        axis=0,
    )
    mean_avg_curve = np.mean(
        np.vstack([_pad_history(r["history_avg"], target_length) for r in run_results]),
        axis=0,
    )
    mean_first_x1_curve = np.mean(
        np.vstack([_pad_history(r["first_solution_x1"], target_length) for r in run_results]),
        axis=0,
    )

    all_positions = np.vstack([r["all_positions_2d"] for r in run_results])
    best_iteration_points = np.vstack([r["best_positions"] for r in run_results])
    best_run_points = np.vstack([r["global_best"][:2] for r in run_results])

    return {
        "runs": runs,
        "best": float(best_per_run[best_idx]),
        "mean": float(np.mean(best_per_run)),
        "std": float(np.sqrt(np.var(best_per_run))),
        "best_point": run_results[best_idx]["global_best"][:2],
        "all_positions": all_positions,
        "best_iteration_points": best_iteration_points,
        "best_run_points": best_run_points,
        "mean_best_curve": mean_best_curve.tolist(),
        "mean_avg_curve": mean_avg_curve.tolist(),
        "mean_first_x1_curve": mean_first_x1_curve.tolist(),
    }


def plot_multi_run_contour(
    function_key: str,
    dimension: int,
    best_iteration_points: np.ndarray,
    best_run_points: np.ndarray,
    best_point: np.ndarray,
) -> plt.Figure:
    xx, yy, zz = get_grid(function_key, dimension)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.contour(xx, yy, zz, levels=18, cmap="viridis", linewidths=1)

    if best_iteration_points.size > 0:
        ax.scatter(best_iteration_points[:, 0], best_iteration_points[:, 1], s=9, c="black", alpha=0.65)
    if best_run_points.size > 0:
        ax.scatter(best_run_points[:, 0], best_run_points[:, 1], s=56, c="#f59e0b", edgecolors="black")

    ax.scatter(best_point[0], best_point[1], s=85, c="red", edgecolors="white", linewidths=0.9)
    ax.set_title(f"Search History ({function_key}), Final iteration", fontsize=9)
    ax.set_xlabel(r"$x_1$", fontsize=8)
    ax.set_ylabel(r"$x_2$", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


if "candidate_text" not in st.session_state:
    st.session_state.candidate_text = ""
if "fitness_value" not in st.session_state:
    st.session_state.fitness_value = ""
if "population" not in st.session_state:
    st.session_state.population = None
if "best_history" not in st.session_state:
    st.session_state.best_history = []
if "latest_eval" not in st.session_state:
    st.session_state.latest_eval = None

st.markdown(
    """
<style>
    :root { color-scheme: light; }
    .stApp { background-color: #f3f4f6; }
    [data-testid="stHeader"] { background-color: #f3f4f6; }
    h1, h2, h3, h4, p, label, span { color: #2f3440; }
    [data-testid="stWidgetLabel"] p { color: #2f3440 !important; font-weight: 600; }
    .block-container { padding-top: 1.2rem; max-width: 1240px; }
    .soft-title {
        background-color: #efe2bc;
        border: 1px solid #f0b429;
        padding: 10px 14px;
        font-size: 40px;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .metric-box {
        background-color: #efe2bc;
        border: 1px solid #f0b429;
        padding: 12px 14px;
        font-size: 30px;
        font-weight: 700;
        margin-top: 0.3rem;
        margin-bottom: 0.5rem;
        width: fit-content;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div { background-color: #e5e7eb !important; border-color: #d1d5db !important; }
    .stTextInput input,
    .stNumberInput input { background-color: #e5e7eb !important; color: #2f3440 !important; }
    .stTextInput input:disabled,
    .stNumberInput input:disabled { -webkit-text-fill-color: #5b6470 !important; opacity: 1 !important; }
    div[data-baseweb="select"] span { color: #2f3440 !important; }
    div[data-baseweb="menu"] {
        background-color: #e5e7eb !important;
        border: 1px solid #d1d5db !important;
    }
    div[data-baseweb="menu"] li,
    div[role="option"] {
        background-color: #e5e7eb !important;
        color: #2f3440 !important;
    }
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="listbox"],
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #e5e7eb !important;
        color: #2f3440 !important;
    }
    div[data-baseweb="menu"] li:hover,
    div[role="option"]:hover {
        background-color: #d8dde5 !important;
        color: #1f2937 !important;
    }
    div[data-baseweb="popover"] li:hover,
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #d8dde5 !important;
        color: #1f2937 !important;
    }
    div[role="option"][aria-selected="true"] {
        background-color: #cfd6e0 !important;
        color: #1f2937 !important;
    }
    .stButton > button {
        background-color: #f3f4f6 !important;
        color: #2f3440 !important;
        border: 1px solid #c7ccd4 !important;
        border-radius: 10px !important;
    }
    .stButton > button:hover { background-color: #e8ebf0 !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: #e5e7eb !important;
        border: 1px solid #d1d5db !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #4b5563 !important; }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #ef4444;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: #ef4444;
    }
    hr { border-top: 1px solid #d1d5db; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("## Part 1 \\ Optimization Problem Initialization")
st.markdown("### Standard Continuous Optimization Benchmark Problems in Metaheuristics")

function_options = list(FUNCTIONS.keys())
if "active_function" not in st.session_state:
    st.session_state.active_function = function_options[0]

active_function = st.session_state.active_function
top1, top2, top3, top4, top5 = st.columns([1.05, 1.35, 1.35, 1.35, 2.0])

with top1:
    st.markdown("#### Solution:")
with top2:
    dimension = int(st.number_input("Dimension (D)", min_value=2, value=30, step=1))
with top3:
    st.text_input("Function", value=active_function, disabled=True)
    min_val, max_val = FUNCTIONS[active_function]["range"]
    st.text_input("Range (Min)", value=f"{min_val:g}", disabled=True)
with top4:
    st.write("")
    st.write("")
    st.text_input("Range (Max)", value=f"+{max_val:g}", disabled=True)
with top5:
    st.write("")
    st.write("")
    generate_sol = st.button("Generate solution", use_container_width=True)
    evaluate_sol = st.button("Evaluate solution", use_container_width=True)

if generate_sol:
    candidate = np.random.uniform(min_val, max_val, size=dimension)
    st.session_state.candidate_text = ",".join([f"{v:.6f}" for v in candidate])
    st.session_state.fitness_value = ""

candidate_text = st.text_input("Candidate solution example", value=st.session_state.candidate_text)
st.session_state.candidate_text = candidate_text

function_row_1, function_row_2 = st.columns([1.6, 4.2])
with function_row_1:
    st.markdown("#### Function:")
    st.selectbox(
        "Function selector",
        options=function_options,
        index=function_options.index(active_function),
        key="active_function",
        label_visibility="collapsed",
    )

active_function = st.session_state.active_function
with function_row_2:
    st.latex(FUNCTIONS[active_function]["latex"])

active_min, active_max = FUNCTIONS[active_function]["range"]

if evaluate_sol:
    try:
        parsed = parse_candidate(candidate_text)
        if parsed.size != dimension:
            st.warning(
                f"Dimension (D) is {dimension} but candidate has {parsed.size} values. "
                "Evaluation is computed using candidate size."
            )
        score = float(evaluate_population(parsed.reshape(1, -1), active_function)[0])
        st.session_state.fitness_value = f"{score:.3f}"
    except Exception as exc:
        st.session_state.fitness_value = f"Error: {exc}"

st.text_input("Fitness", value=st.session_state.fitness_value, disabled=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="soft-title">Population Initialization</div>', unsafe_allow_html=True)

pop_row_1, pop_row_2, pop_row_3 = st.columns([1.05, 4.6, 1.6])
with pop_row_1:
    st.markdown("#### Population:")
with pop_row_2:
    pop_size = int(st.slider("Size", min_value=2, max_value=200, value=30, step=1))
with pop_row_3:
    st.write("")
    st.write("")
    generate_pop = st.button("Generate population", use_container_width=True)

tp2_dir = Path(__file__).resolve().parent / "TP2"
built_in = sorted(tp2_dir.glob("Population_*.csv"))
file_col_1, file_col_2 = st.columns([3.2, 1.8])
with file_col_1:
    uploaded_file = st.file_uploader("Upload population CSV", type=["csv"], label_visibility="collapsed")
with file_col_2:
    selected_builtin = st.selectbox(
        "Select predefined CSV",
        ["None"] + [f.name for f in built_in],
        label_visibility="collapsed",
    )

if generate_pop:
    st.session_state.population = np.random.uniform(active_min, active_max, size=(pop_size, dimension))
    st.session_state.latest_eval = None

if uploaded_file is not None:
    try:
        st.session_state.population = load_population_from_csv(uploaded_file)
        st.session_state.latest_eval = None
    except Exception as exc:
        st.error(str(exc))
elif selected_builtin != "None":
    try:
        df_builtin = pd.read_csv(tp2_dir / selected_builtin, sep=";", header=None)
        st.session_state.population = parse_population_dataframe(df_builtin)
        st.session_state.latest_eval = None
    except Exception as exc:
        st.error(str(exc))

eval_col_1, eval_col_2, eval_col_3 = st.columns([1.4, 1.1, 2.5])
with eval_col_1:
    evaluate_pop = st.button("Evaluate population", use_container_width=True)

if evaluate_pop:
    population = st.session_state.population
    if population is None:
        st.warning("Generate or load a population first.")
    else:
        scores = evaluate_population(population, active_function)
        best_idx = int(np.argmin(scores))
        worst_idx = int(np.argmax(scores))
        best_value = float(scores[best_idx])
        worst_value = float(scores[worst_idx])
        st.session_state.best_history.append((population[best_idx, 0], population[best_idx, 1]))
        st.session_state.latest_eval = {
            "best": best_value,
            "worst": worst_value,
            "best_idx": best_idx,
            "population": population.copy(),
        }

latest_eval = st.session_state.latest_eval

if latest_eval is not None:
    st.markdown(
        f'<div class="metric-box">Best — {latest_eval["best"]:.2f}, Worst — {latest_eval["worst"]:.2f}</div>',
        unsafe_allow_html=True,
    )

    graph_col_1, graph_col_2 = st.columns(2)
    with graph_col_1:
        st.pyplot(plot_surface(active_function, latest_eval["population"].shape[1]))
    with graph_col_2:
        st.pyplot(
            plot_contour_with_population(
                latest_eval["population"],
                active_function,
                latest_eval["best_idx"],
                latest_eval["population"].shape[1],
            )
        )

if st.session_state.population is not None:
    export_csv = pd.DataFrame(st.session_state.population).to_csv(index=False, header=False, sep=";")
    st.download_button("Download generated population CSV", data=export_csv, file_name="population_generated.csv")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"### Application of PSO for {active_function} function:")

tp3_dir = Path(__file__).resolve().parent / "TP3"
tp3_population_files = sorted(tp3_dir.glob("Population_*.csv"))
tp3_population_names = [f.name for f in tp3_population_files]

preferred_token = FUNCTIONS[active_function]["short"]
preferred_file = next((name for name in tp3_population_names if preferred_token in name), None)

if "pso_init_mode" not in st.session_state:
    st.session_state.pso_init_mode = "CSV"
if "pso_csv_choice" not in st.session_state:
    st.session_state.pso_csv_choice = preferred_file if preferred_file is not None else "None"
if st.session_state.pso_csv_choice not in (["None"] + tp3_population_names):
    st.session_state.pso_csv_choice = preferred_file if preferred_file is not None else "None"

pso_source_col_1, pso_source_col_2, pso_source_col_3 = st.columns([1.2, 1.4, 1.4])
with pso_source_col_1:
    pso_init_mode = st.radio("PSO init", ["Random", "CSV"], horizontal=True, key="pso_init_mode")
with pso_source_col_2:
    pso_csv_choice = st.selectbox(
        "TP3 CSV",
        ["None"] + tp3_population_names,
        key="pso_csv_choice",
        disabled=(pso_init_mode != "CSV"),
    )
with pso_source_col_3:
    pso_csv_upload = st.file_uploader(
        "Upload TP3 CSV",
        type=["csv"],
        disabled=(pso_init_mode != "CSV"),
    )

pso_preview_text = "Source: Random initialization"
if pso_init_mode == "CSV":
    try:
        if pso_csv_upload is not None:
            preview_population = load_population_from_csv(pso_csv_upload)
            preview_scores = evaluate_population(preview_population, active_function)
            pso_preview_text = (
                f"Source: Uploaded CSV ({preview_population.shape[0]}x{preview_population.shape[1]}) | "
                f"Initial Best={np.min(preview_scores):.2f}, Worst={np.max(preview_scores):.2f}"
            )
        elif pso_csv_choice != "None":
            preview_df = pd.read_csv(tp3_dir / pso_csv_choice, sep=";", header=None)
            preview_population = parse_population_dataframe(preview_df)
            preview_scores = evaluate_population(preview_population, active_function)
            pso_preview_text = (
                f"Source: {pso_csv_choice} ({preview_population.shape[0]}x{preview_population.shape[1]}) | "
                f"Initial Best={np.min(preview_scores):.2f}, Worst={np.max(preview_scores):.2f}"
            )
        else:
            pso_preview_text = "Source: CSV mode selected (choose file to run)"
    except Exception:
        pso_preview_text = "Source: CSV preview unavailable (invalid file/format)"

st.caption(pso_preview_text)

pso_box_top_1, pso_box_top_2, pso_box_top_3 = st.columns([1.6, 1.2, 1.0])
with pso_box_top_1:
    pso_population_size = int(st.number_input("Single run - Population size", min_value=5, value=30, step=1))
    run_pso = st.button("Evaluate PSO", use_container_width=True)
with pso_box_top_2:
    st.text_input("Metaheuristic", value="PSO", disabled=True)
with pso_box_top_3:
    pso_max_iter = int(st.number_input("Max Iteration (T)", min_value=5, value=200, step=1))

pso_params_col = st.columns(4)
with pso_params_col[0]:
    pso_w = st.number_input("w", value=0.3, step=0.1, format="%.3f")
with pso_params_col[1]:
    pso_c1 = st.number_input("c1", value=1.4, step=0.1, format="%.3f")
with pso_params_col[2]:
    pso_c2 = st.number_input("c2", value=1.4, step=0.1, format="%.3f")
with pso_params_col[3]:
    pso_seed = int(st.number_input("Seed", min_value=0, value=14, step=1))

if "pso_result" not in st.session_state:
    st.session_state.pso_result = None

if run_pso:
    pso_initial_population = None
    can_run_pso = True
    pso_source_label = "Random initialization"
    if pso_init_mode == "CSV":
        try:
            if pso_csv_upload is not None:
                pso_initial_population = load_population_from_csv(pso_csv_upload)
                pso_source_label = "Uploaded CSV"
            elif pso_csv_choice != "None":
                pso_df = pd.read_csv(tp3_dir / pso_csv_choice, sep=";", header=None)
                pso_initial_population = parse_population_dataframe(pso_df)
                pso_source_label = pso_csv_choice
            else:
                st.warning("Choose a TP3 CSV file or upload one for PSO CSV mode.")
                can_run_pso = False
        except Exception as exc:
            st.error(f"PSO CSV load error: {exc}")
            can_run_pso = False

    if can_run_pso:
        st.session_state.pso_result = run_pso_pseudocode(
            function_key=active_function,
            dimension=int(dimension),
            population_size=pso_population_size,
            max_iterations=pso_max_iter,
            w=float(pso_w),
            c1=float(pso_c1),
            c2=float(pso_c2),
            seed=pso_seed,
            initial_population=pso_initial_population,
        )
        st.session_state.pso_result["source_label"] = pso_source_label

pso_result = st.session_state.pso_result

if pso_result is not None:
    pso_top = st.columns([1.15, 1.15, 1.15, 1.0])

    with pso_top[0]:
        st.pyplot(plot_surface(active_function, int(dimension)))

    with pso_top[1]:
        first_best = pso_result["initial_population"][int(np.argmin(evaluate_population(pso_result["initial_population"], active_function)))]
        st.pyplot(
            plot_contour_population_state(
                active_function,
                int(dimension),
                pso_result["initial_population"],
                first_best,
                f"Search History ({active_function}), 1st iteration",
            )
        )

    with pso_top[2]:
        st.pyplot(
            plot_contour_population_state(
                active_function,
                int(dimension),
                pso_result["final_population"],
                pso_result["global_best"],
                f"Search History ({active_function}), Final iteration",
                extra_points=pso_result["all_positions_2d"],
                trail_points=pso_result["best_positions"],
            )
        )

    with pso_top[3]:
        st.caption(f"Source: {pso_result.get('source_label', 'Unknown')}")
        st.markdown("**Initial population:**")
        st.write(f"Best — {pso_result['initial_best']:.2f}, Worst — {pso_result['initial_worst']:.2f}")
        st.markdown("**Final population:**")
        st.write(f"Best — {pso_result['final_best']:.1f}")
        st.markdown(f"**Stagnation — Iteration N°{pso_result['stagnation_iteration']}**")

    pso_bottom = st.columns(3)
    with pso_bottom[0]:
        st.pyplot(plot_simple_curve(pso_result["history_best"], "red", "Convergence Curve", "Fitness"))
    with pso_bottom[1]:
        st.pyplot(
            plot_simple_curve(
                pso_result["first_solution_x1"],
                "limegreen",
                "Trajectory of 1st solution",
                r"x1",
            )
        )
    with pso_bottom[2]:
        st.pyplot(plot_simple_curve(pso_result["history_avg"], "royalblue", "Average Fitness", "Fitness"))

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Running Multiple PSO Experiments")
st.caption("TP4: Running PSO with multiple populations")

multi_ctrl_1, multi_ctrl_2 = st.columns([3.2, 1.3])
with multi_ctrl_1:
    multiple_runs = int(st.slider("Run", min_value=2, max_value=50, value=30, step=1))
with multi_ctrl_2:
    st.write("")
    st.write("")
    evaluate_multi = st.button("Evaluate", use_container_width=True)

tp4_max_iterations = 500

if "pso_multi_result" not in st.session_state:
    st.session_state.pso_multi_result = None

if evaluate_multi:
    st.session_state.pso_multi_result = run_multiple_pso_experiments(
        function_key=active_function,
        dimension=int(dimension),
        population_size=pso_population_size,
        max_iterations=tp4_max_iterations,
        w=float(pso_w),
        c1=float(pso_c1),
        c2=float(pso_c2),
        seed=int(pso_seed),
        runs=multiple_runs,
    )

multi_result = st.session_state.pso_multi_result

if multi_result is not None:
    multi_top = st.columns([1.25, 1.25, 1.0])
    with multi_top[0]:
        st.pyplot(plot_surface(active_function, int(dimension)))
    with multi_top[1]:
        st.pyplot(
            plot_multi_run_contour(
                active_function,
                int(dimension),
                multi_result["best_iteration_points"],
                multi_result["best_run_points"],
                multi_result["best_point"],
            )
        )

    with multi_top[2]:
        st.markdown(
            f"""
<div style="border:1px solid #202124; padding:16px 20px; margin-top:8px; background:#f3f4f6;">
  <p style="font-size:20px; font-weight:700; margin:0 0 16px 0;">Best -- {multi_result['best']:.2f},</p>
  <p style="font-size:20px; font-weight:700; margin:0 0 16px 0;">Mean (average error) -- {multi_result['mean']:.2f},</p>
  <p style="font-size:20px; font-weight:700; margin:0;">STD -- {multi_result['std']:.2f},</p>
</div>
""",
            unsafe_allow_html=True,
        )

    legend_row = st.columns([1.15, 1.4, 1.15])
    with legend_row[0]:
        st.markdown("Best solution at each iteration")
        st.markdown("---")
        st.markdown("Best solution of each run")
        st.markdown("---")
        st.markdown("Best solution across all runs")
    with legend_row[1]:
        st.write("")
    with legend_row[2]:
        st.markdown("### Best Solution Position Updates")
        st.markdown("---")

    multi_bottom = st.columns(3)
    with multi_bottom[0]:
        st.pyplot(
            plot_simple_curve(
                multi_result["mean_best_curve"],
                "red",
                "Convergence Curve",
                "Fitness",
            )
        )
        st.caption("Mean Best Fitness of All Runs vs. Iteration")
    with multi_bottom[1]:
        st.pyplot(
            plot_simple_curve(
                multi_result["mean_first_x1_curve"],
                "limegreen",
                "Trajectory of 1st solution",
                r"x1",
            )
        )
        st.caption(r"Mean $x_1^{(1)}$ of all Runs vs. Iteration")
    with multi_bottom[2]:
        st.pyplot(
            plot_simple_curve(
                multi_result["mean_avg_curve"],
                "royalblue",
                "Average Fitness of population",
                "Fitness",
            )
        )
        st.caption("Mean Population Average Fitness of All Runs vs. Iteration")