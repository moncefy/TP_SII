from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="TP4 - Multiple PSO Experiments", layout="wide")


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
}


def evaluate_population(population: np.ndarray, function_key: str) -> np.ndarray:
    return FUNCTIONS[function_key]["fn"](population)


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


def run_pso_pseudocode(
    function_key: str,
    dimension: int,
    population_size: int,
    max_iterations: int,
    w: float,
    c1: float,
    c2: float,
    seed: int | None,
) -> dict:
    if seed is not None:
        np.random.seed(seed)

    minimum, maximum = FUNCTIONS[function_key]["range"]
    population = np.random.uniform(minimum, maximum, size=(population_size, dimension))

    velocity_scale = 0.1 * (maximum - minimum)
    velocity = np.random.uniform(-velocity_scale, velocity_scale, size=(population_size, dimension))

    personal_best = population.copy()
    fitness = evaluate_population(population, function_key)
    personal_best_fitness = fitness.copy()

    best_idx = int(np.argmin(fitness))
    global_best = population[best_idx].copy()
    global_best_fitness = float(fitness[best_idx])

    history_best = [global_best_fitness]
    history_avg = [float(np.mean(fitness))]
    first_solution_x1 = [float(population[0, 0])]
    best_positions = [global_best[:2].copy()]

    for _ in range(1, max_iterations + 1):
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

        history_best.append(global_best_fitness)
        history_avg.append(float(np.mean(fitness)))
        first_solution_x1.append(float(population[0, 0]))
        best_positions.append(global_best[:2].copy())

    return {
        "final_best": float(global_best_fitness),
        "global_best": global_best,
        "history_best": history_best,
        "history_avg": history_avg,
        "first_solution_x1": first_solution_x1,
        "best_positions": np.array(best_positions),
    }


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

    best_iteration_points = np.vstack([r["best_positions"] for r in run_results])
    best_run_points = np.vstack([r["global_best"][:2] for r in run_results])

    return {
        "best": float(best_per_run[best_idx]),
        "mean": float(np.mean(best_per_run)),
        "std": float(np.sqrt(np.var(best_per_run))),
        "best_point": run_results[best_idx]["global_best"][:2],
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


def render_curve_with_caption(y_values: list[float], color: str, title: str, y_label: str, caption: str) -> None:
    st.pyplot(plot_simple_curve(y_values, color, title, y_label))
    st.caption(caption)


base_dir = Path(__file__).resolve().parent
css_candidates = [base_dir / "app_styles.css", base_dir.parent / "app_styles.css"]
for css_path in css_candidates:
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
        break

st.markdown("### Running Multiple PSO Experiments")
st.caption("TP4: Running PSO with multiple populations")

function_col, dim_col, pop_col, iter_col = st.columns([1.4, 0.8, 1.1, 0.9])
with function_col:
    selected_function = st.selectbox("Function", list(FUNCTIONS.keys()), index=7)
with dim_col:
    dimension = int(st.number_input("Dimension (D)", min_value=2, value=30, step=1))
with pop_col:
    population_size = int(st.number_input("Population size", min_value=5, value=30, step=1))
with iter_col:
    max_iterations = 500
    st.number_input("Max Iteration (T)", min_value=500, max_value=500, value=500, step=1, disabled=True)

st.latex(FUNCTIONS[selected_function]["latex"])

param_col = st.columns(4)
with param_col[0]:
    pso_w = st.number_input("w", value=0.3, step=0.1, format="%.3f")
with param_col[1]:
    pso_c1 = st.number_input("c1", value=1.4, step=0.1, format="%.3f")
with param_col[2]:
    pso_c2 = st.number_input("c2", value=1.4, step=0.1, format="%.3f")
with param_col[3]:
    pso_seed = int(st.number_input("Seed", min_value=0, value=14, step=1))

multi_ctrl_1, multi_ctrl_2 = st.columns([3.2, 1.3])
with multi_ctrl_1:
    multiple_runs = int(st.slider("Run", min_value=2, max_value=50, value=30, step=1))
with multi_ctrl_2:
    st.write("")
    st.write("")
    evaluate_multi = st.button("Evaluate", use_container_width=True)

if "pso_multi_result_pso_py" not in st.session_state:
    st.session_state.pso_multi_result_pso_py = None

if evaluate_multi:
    with st.spinner("Running multiple PSO experiments..."):
        st.session_state.pso_multi_result_pso_py = run_multiple_pso_experiments(
            function_key=selected_function,
            dimension=dimension,
            population_size=population_size,
            max_iterations=max_iterations,
            w=float(pso_w),
            c1=float(pso_c1),
            c2=float(pso_c2),
            seed=int(pso_seed),
            runs=multiple_runs,
        )

multi_result = st.session_state.pso_multi_result_pso_py

if multi_result is not None:
    multi_top = st.columns([1.25, 1.25, 1.0])
    with multi_top[0]:
        st.pyplot(plot_surface(selected_function, dimension))
    with multi_top[1]:
        st.pyplot(
            plot_multi_run_contour(
                selected_function,
                dimension,
                multi_result["best_iteration_points"],
                multi_result["best_run_points"],
                multi_result["best_point"],
            )
        )
    with multi_top[2]:
        st.markdown(
            f"""
<div style="border:1px solid #4b5563; padding:16px 20px; margin-top:8px; background:#e5e7eb; color:#111827 !important; border-radius:6px;">
    <p style="font-size:20px; font-weight:700; margin:0 0 16px 0; color:#111827 !important;">Best -- {multi_result['best']:.2f},</p>
    <p style="font-size:20px; font-weight:700; margin:0 0 16px 0; color:#111827 !important;">Mean (average error) -- {multi_result['mean']:.2f},</p>
    <p style="font-size:20px; font-weight:700; margin:0; color:#111827 !important;">STD -- {multi_result['std']:.2f},</p>
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
        render_curve_with_caption(
            multi_result["mean_best_curve"],
            "red",
            "Convergence Curve",
            "Fitness",
            "Mean Best Fitness of All Runs vs. Iteration",
        )
    with multi_bottom[1]:
        render_curve_with_caption(
            multi_result["mean_first_x1_curve"],
            "limegreen",
            "Trajectory of 1st solution",
            r"x1",
            r"Mean $x_1^{(1)}$ of all Runs vs. Iteration",
        )
    with multi_bottom[2]:
        render_curve_with_caption(
            multi_result["mean_avg_curve"],
            "royalblue",
            "Average Fitness of population",
            "Fitness",
            "Mean Population Average Fitness of All Runs vs. Iteration",
        )
