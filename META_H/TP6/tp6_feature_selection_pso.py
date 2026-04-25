import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import load_digits, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


@st.cache_data
def load_datasets():
    digits = load_digits()
    x_digits = digits.data
    y_digits = digits.target

    x_syn, y_syn = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=5,
        n_redundant=10,
        n_classes=2,
        random_state=42,
    )

    return {
        "Synthetic": (x_syn, y_syn),
        "Digits": (x_digits, y_digits),
    }


def evaluate_solution(solution, x, y, alpha, test_size=0.3, random_state=42):
    d = x.shape[1]
    selected = np.where(solution > 0.5)[0]
    sf = int(selected.size)

    if sf == 0:
        accuracy = 0.0
        f1 = 1.0
        f2 = 0.0
        fitness = alpha * f1 + (1.0 - alpha) * f2
        return fitness, accuracy, sf

    x_selected = x[:, selected]
    x_train, x_test, y_train, y_test = train_test_split(
        x_selected,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = 1.0 - accuracy
    f2 = sf / d
    fitness = alpha * f1 + (1.0 - alpha) * f2

    return float(fitness), float(accuracy), sf


def run_one_pso(x, y, alpha, n_particles, n_iterations, w, c1, c2, seed):
    rng = np.random.default_rng(seed)
    d = x.shape[1]

    positions = rng.random((n_particles, d))
    velocities = rng.uniform(-0.2, 0.2, size=(n_particles, d))

    pbest_positions = positions.copy()
    pbest_fitness = np.zeros(n_particles)
    pbest_accuracy = np.zeros(n_particles)
    pbest_selected = np.zeros(n_particles, dtype=int)

    for i in range(n_particles):
        fit, acc, sel = evaluate_solution(positions[i], x, y, alpha)
        pbest_fitness[i] = fit
        pbest_accuracy[i] = acc
        pbest_selected[i] = sel

    gbest_idx = int(np.argmin(pbest_fitness))
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = float(pbest_fitness[gbest_idx])
    gbest_accuracy = float(pbest_accuracy[gbest_idx])
    gbest_selected = int(pbest_selected[gbest_idx])

    convergence = [gbest_fitness]
    avg_fitness = [float(np.mean(pbest_fitness))]
    first_traj = [float(positions[0, 0])]

    for _ in range(n_iterations):
        r1 = rng.random((n_particles, d))
        r2 = rng.random((n_particles, d))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )
        positions = positions + velocities
        positions = np.clip(positions, 0.0, 1.0)

        current_fitness = np.zeros(n_particles)
        current_accuracy = np.zeros(n_particles)
        current_selected = np.zeros(n_particles, dtype=int)

        for i in range(n_particles):
            fit, acc, sel = evaluate_solution(positions[i], x, y, alpha)
            current_fitness[i] = fit
            current_accuracy[i] = acc
            current_selected[i] = sel

            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_accuracy[i] = acc
                pbest_selected[i] = sel
                pbest_positions[i] = positions[i].copy()

                if fit < gbest_fitness:
                    gbest_fitness = float(fit)
                    gbest_accuracy = float(acc)
                    gbest_selected = int(sel)
                    gbest_position = positions[i].copy()

        convergence.append(gbest_fitness)
        avg_fitness.append(float(np.mean(current_fitness)))
        first_traj.append(float(positions[0, 0]))

    return {
        "best_fitness": gbest_fitness,
        "best_accuracy": gbest_accuracy,
        "best_selected": gbest_selected,
        "best_position": gbest_position,
        "convergence": np.array(convergence, dtype=float),
        "avg_fitness": np.array(avg_fitness, dtype=float),
        "first_traj": np.array(first_traj, dtype=float),
        "final_positions": positions.copy(),
    }


def run_pso_multi_run(x, y, alpha, n_particles, n_iterations, n_runs, w, c1, c2, base_seed):
    runs = []
    for r in range(n_runs):
        result = run_one_pso(
            x=x,
            y=y,
            alpha=alpha,
            n_particles=n_particles,
            n_iterations=n_iterations,
            w=w,
            c1=c1,
            c2=c2,
            seed=base_seed + r,
        )
        runs.append(result)

    run_best_values = np.array([r["best_fitness"] for r in runs], dtype=float)
    run_selected_values = np.array([r["best_selected"] for r in runs], dtype=float)
    best_run_idx = int(np.argmin(run_best_values))
    best_run = runs[best_run_idx]

    return {
        "best": float(np.min(run_best_values)),
        "mean": float(np.mean(run_best_values)),
        "std": float(np.std(run_best_values)),
        "selected_display": int(np.rint(np.mean(run_selected_values))),
        "best_run": best_run,
    }


def make_line_plot(values, title, y_label, color):
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.plot(np.arange(len(values)), values, color=color, linewidth=1.6)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Iteration", fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def make_search_plot(final_positions, best_position):
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    x1 = final_positions[:, 0]
    x2 = final_positions[:, 1] if final_positions.shape[1] > 1 else np.zeros_like(x1)
    ax.scatter(x1, x2, c="black", s=9)
    ax.scatter(best_position[0], best_position[1] if best_position.shape[0] > 1 else 0.0, c="red", s=35)
    ax.set_title("Search History (Final Iteration)", fontsize=8)
    ax.set_xlabel("x1", fontsize=7)
    ax.set_ylabel("x2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="TP6 - Feature Selection with PSO", layout="wide")

st.markdown(
    """
<style>
div.tp6-card {
    border: 1.6px solid #303030;
    padding: 10px 14px 8px 14px;
    margin-top: 4px;
}
div.tp6-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
}
div.tp6-subtitle {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="tp6-card">', unsafe_allow_html=True)
st.markdown('<div class="tp6-title"></div>', unsafe_allow_html=True)

left, mid, right = st.columns([1.2, 1.15, 0.7])

with left:
    st.markdown('<div class="tp6-subtitle">Feature Selection parameters</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox("Data", ["Synthetic", "Digits"], index=0)
    alpha = st.number_input("α", min_value=0.0, max_value=1.0, value=0.99, step=0.01, format="%.2f")

    st.markdown('<div class="tp6-subtitle" style="margin-top:14px;">PSO parameters</div>', unsafe_allow_html=True)
    pso_col1, pso_col2, pso_col3 = st.columns(3)
    with pso_col1:
        w = st.number_input("w", min_value=0.0, max_value=1.5, value=0.5, step=0.1, format="%.1f")
    with pso_col2:
        c1 = st.number_input("c1", min_value=0.0, max_value=4.0, value=2.0, step=0.1, format="%.1f")
    with pso_col3:
        c2 = st.number_input("c2", min_value=0.0, max_value=4.0, value=2.0, step=0.1, format="%.1f")

with mid:
    st.markdown('<div class="tp6-subtitle">Metaheuristic parameters</div>', unsafe_allow_html=True)
    n_particles = st.slider("Population (N)", min_value=5, max_value=100, value=10, step=1)
    n_iterations = st.slider("Max iteration (T)", min_value=5, max_value=200, value=20, step=1)
    n_runs = st.slider("Run", min_value=1, max_value=50, value=15, step=1)

with right:
    st.write("")
    st.write("")
    evaluate_clicked = st.button("Evaluation", use_container_width=True)

if "tp6_result" not in st.session_state:
    st.session_state.tp6_result = None

if evaluate_clicked:
    datasets = load_datasets()
    x, y = datasets[dataset_name]

    result = run_pso_multi_run(
        x=x,
        y=y,
        alpha=alpha,
        n_particles=n_particles,
        n_iterations=n_iterations,
        n_runs=n_runs,
        w=w,
        c1=c1,
        c2=c2,
        base_seed=42,
    )
    st.session_state.tp6_result = result

if st.session_state.tp6_result is None:
    datasets = load_datasets()
    x, y = datasets[dataset_name]
    st.session_state.tp6_result = run_pso_multi_run(
        x=x,
        y=y,
        alpha=alpha,
        n_particles=n_particles,
        n_iterations=n_iterations,
        n_runs=n_runs,
        w=w,
        c1=c1,
        c2=c2,
        base_seed=42,
    )

if st.session_state.tp6_result is not None:
    result = st.session_state.tp6_result
    best_run = result["best_run"]

    with right:
        st.write("")
        st.markdown(f"**Best** - {result['best']:.4f},")
        st.markdown(f"**Mean (average error)** - {result['mean']:.4f},")
        st.markdown(f"**Accuracy** - {best_run['best_accuracy']:.2f},")
        st.markdown(f"**Selected** - {result['selected_display']},")
        st.markdown(f"**STD** - {result['std']:.4f},")

    chart1, chart2, chart3, chart4 = st.columns(4)

    with chart1:
        fig1 = make_line_plot(best_run["convergence"], "Convergence Curve", "Fitness", "#e66a6a")
        st.pyplot(fig1)

    with chart2:
        fig2 = make_line_plot(best_run["first_traj"], "Trajectory of 1st solution", "x1", "#7ed957")
        st.pyplot(fig2)

    with chart3:
        fig3 = make_line_plot(best_run["avg_fitness"], "Average Fitness of population", "Fitness", "#5b7cff")
        st.pyplot(fig3)

    with chart4:
        fig4 = make_search_plot(best_run["final_positions"], best_run["best_position"])
        st.pyplot(fig4)

st.markdown('</div>', unsafe_allow_html=True)
