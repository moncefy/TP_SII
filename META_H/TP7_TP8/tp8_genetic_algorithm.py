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
        fitness = alpha * 1.0 + (1.0 - alpha) * 0.0
        return float(fitness), float(accuracy), sf

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
    fitness = alpha * (1.0 - accuracy) + (1.0 - alpha) * (sf / d)
    return float(fitness), float(accuracy), sf


def generate_random_population(n_particles, dimension, rng):
    return rng.integers(0, 2, size=(n_particles, dimension), endpoint=False).astype(int)


def evaluate_population(population, x, y, alpha):
    fitness = np.zeros(population.shape[0], dtype=float)
    accuracy = np.zeros(population.shape[0], dtype=float)
    selected = np.zeros(population.shape[0], dtype=int)
    for i in range(population.shape[0]):
        fitness[i], accuracy[i], selected[i] = evaluate_solution(population[i], x, y, alpha)
    return fitness, accuracy, selected


def cumulative_selection(population, fitness, rng):
    scores = 1.0 / (fitness + 1e-12)
    probs = scores / np.sum(scores)
    cdf = np.cumsum(probs)

    def pick_one():
        r = rng.random()
        idx = int(np.searchsorted(cdf, r, side="right"))
        return min(idx, len(population) - 1)

    i1 = pick_one()
    i2 = pick_one()
    while i2 == i1 and len(population) > 1:
        i2 = pick_one()
    return population[i1].copy(), population[i2].copy()


def random_selection(population, rng):
    idx = rng.choice(len(population), size=2, replace=False if len(population) > 1 else True)
    return population[int(idx[0])].copy(), population[int(idx[1])].copy()


def one_point_crossover(parent1, parent2, rng):
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()
    if rng.random() >= RC_DEFAULT:
        return parent1.copy(), parent2.copy()
    k = rng.integers(1, len(parent1))
    child1 = np.concatenate([parent1[:k], parent2[k:]])
    child2 = np.concatenate([parent2[:k], parent1[k:]])
    return child1, child2


def two_point_crossover(parent1, parent2, rng):
    if len(parent1) < 3:
        return one_point_crossover(parent1, parent2, rng)
    if rng.random() >= RC_DEFAULT:
        return parent1.copy(), parent2.copy()
    k1 = rng.integers(1, len(parent1) - 1)
    k2 = rng.integers(k1 + 1, len(parent1))
    child1 = np.concatenate([parent1[:k1], parent2[k1:k2], parent1[k2:]])
    child2 = np.concatenate([parent2[:k1], parent1[k1:k2], parent2[k2:]])
    return child1, child2


def mutate(child, rng):
    mutated = child.copy()
    for j in range(mutated.shape[0]):
        if rng.random() < RM_DEFAULT:
            mutated[j] = 1 - mutated[j]
    return mutated


def replacement_population(population, children, fitness, child_fitness, mode):
    combined = np.vstack([population, children])
    combined_fitness = np.concatenate([fitness, child_fitness])
    order = np.argsort(combined_fitness)

    if mode == "Children":
        return children

    # Best solutions from union
    return combined[order[: population.shape[0]]]


def run_ga_one(x, y, alpha, n_particles, n_iterations, selection_mode, crossover_mode, replacement_mode, rc, rm, seed):
    rng = np.random.default_rng(seed)
    dimension = x.shape[1]
    population = generate_random_population(n_particles, dimension, rng)

    fitness, accuracy, selected = evaluate_population(population, x, y, alpha)
    best_idx = int(np.argmin(fitness))
    best_solution = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    best_accuracy = float(accuracy[best_idx])
    best_selected = int(selected[best_idx])

    convergence = [best_fitness]
    average_fitness = [float(np.mean(fitness))]
    first_solution_traj = [float(population[0, 0])]

    for _ in range(n_iterations):
        children = []
        for _pair in range(n_particles // 2):
            if selection_mode == "Cumulative":
                parent1, parent2 = cumulative_selection(population, fitness, rng)
            else:
                parent1, parent2 = random_selection(population, rng)

            if crossover_mode == "1-Point":
                child1, child2 = one_point_crossover(parent1, parent2, rng)
            elif crossover_mode == "2-Point":
                child1, child2 = two_point_crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = mutate(child1, rng)
            child2 = mutate(child2, rng)
            children.append(child1)
            children.append(child2)

        if len(children) < n_particles:
            while len(children) < n_particles:
                children.append(rng.integers(0, 2, size=dimension).astype(int))

        children = np.array(children[:n_particles], dtype=int)
        child_fitness, child_accuracy, child_selected = evaluate_population(children, x, y, alpha)

        if replacement_mode == "Children":
            population = children.copy()
            fitness = child_fitness.copy()
            accuracy = child_accuracy.copy()
            selected = child_selected.copy()
        else:
            combined = np.vstack([population, children])
            combined_fitness = np.concatenate([fitness, child_fitness])
            combined_accuracy = np.concatenate([accuracy, child_accuracy])
            combined_selected = np.concatenate([selected, child_selected])
            order = np.argsort(combined_fitness)
            chosen = order[:n_particles]
            population = combined[chosen].copy()
            fitness = combined_fitness[chosen].copy()
            accuracy = combined_accuracy[chosen].copy()
            selected = combined_selected[chosen].copy()

        cur_best_idx = int(np.argmin(fitness))
        if fitness[cur_best_idx] < best_fitness:
            best_fitness = float(fitness[cur_best_idx])
            best_accuracy = float(accuracy[cur_best_idx])
            best_selected = int(selected[cur_best_idx])
            best_solution = population[cur_best_idx].copy()

        convergence.append(best_fitness)
        average_fitness.append(float(np.mean(fitness)))
        first_solution_traj.append(float(population[0, 0]))

    return {
        "best_fitness": best_fitness,
        "best_accuracy": best_accuracy,
        "best_selected": best_selected,
        "best_solution": best_solution,
        "convergence": np.array(convergence, dtype=float),
        "average_fitness": np.array(average_fitness, dtype=float),
        "first_solution_traj": np.array(first_solution_traj, dtype=float),
    }


def run_ga_multi(x, y, alpha, n_particles, n_iterations, selection_mode, crossover_mode, replacement_mode, rc, rm, n_runs, seed):
    runs = []
    for r in range(n_runs):
        runs.append(
            run_ga_one(
                x=x,
                y=y,
                alpha=alpha,
                n_particles=n_particles,
                n_iterations=n_iterations,
                selection_mode=selection_mode,
                crossover_mode=crossover_mode,
                replacement_mode=replacement_mode,
                rc=rc,
                rm=rm,
                seed=seed + r,
            )
        )

    best_values = np.array([r["best_fitness"] for r in runs], dtype=float)
    selected_values = np.array([r["best_selected"] for r in runs], dtype=float)
    best_run_idx = int(np.argmin(best_values))
    best_run = runs[best_run_idx]

    return {
        "best": float(np.min(best_values)),
        "mean": float(np.mean(best_values)),
        "std": float(np.std(best_values)),
        "selected_display": int(np.rint(np.mean(selected_values))),
        "best_run": best_run,
    }


def make_line_plot(values, title, y_label, color):
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    ax.plot(np.arange(len(values)), values, color=color, linewidth=1.6)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Iteration", fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


RC_DEFAULT = 0.7
RM_DEFAULT = 0.10

st.set_page_config(page_title="TP7/TP8 - Genetic Algorithm", layout="wide")

st.markdown(
    """
<style>
div.tp-header-grid {
    border: 1.2px solid #3d3d3d;
    display: grid;
    grid-template-columns: 1fr 1.25fr 1fr;
    margin-top: 6px;
}
div.tp-header-cell {
    border-right: 1.2px solid #3d3d3d;
    padding: 8px 10px;
    min-height: 72px;
    font-size: 15px;
    line-height: 1.25;
}
div.tp-header-cell:last-child {
    border-right: 0;
}
div.tp-card {
    border: 1.6px solid #303030;
    padding: 10px 14px 8px 14px;
    margin-top: 6px;
}
div.tp-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
}
div.section-title {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
        """
<div class="tp-header-grid">
    <div class="tp-header-cell">
        Année Universitaire : 2025/2026<br>
        Master 2 : SII<br>
        Module : MÉTA
    </div>
    <div class="tp-header-cell" style="text-align:center;">
        Université des Sciences et de la Technologie Houari Boumediene<br>
        Faculté d'Informatique<br>
        Département d'Intelligence Artificielle et Sciences des Données
    </div>
    <div class="tp-header-cell" style="text-align:right;">
        TP N°8<br>
        Genetic Algorithm<br>
        Part 2
    </div>
</div>
""",
        unsafe_allow_html=True,
)

st.markdown('<div class="tp-card">', unsafe_allow_html=True)
st.markdown('<div class="tp-title">Feature Selection with GA</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 13px; font-weight: 700; margin-top: -4px;">Application of GA</div>', unsafe_allow_html=True)

left, mid, right = st.columns([1.15, 1.2, 0.7])

with left:
    st.markdown('<div class="section-title">Feature Selection parameters</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox("Data", ["Synthetic", "Digits"], index=0)
    alpha = st.number_input("α", min_value=0.0, max_value=1.0, value=0.99, step=0.01, format="%.2f")

    st.markdown('<div class="section-title" style="margin-top: 14px;">GA parameters</div>', unsafe_allow_html=True)
    selection_mode = st.selectbox("Selection", ["Cumulative", "Random"], index=0)
    crossover_mode = st.selectbox("Crossover", ["1-Point", "2-Point", "None"], index=0)
    replacement_choice = st.selectbox("Replacement", ["Children", "Best"], index=0)
    replacement_mode = "Children" if replacement_choice == "Children" else "Best"
    ga_col1, ga_col2 = st.columns(2)
    with ga_col1:
        rc = st.number_input("R_c", min_value=0.0, max_value=1.0, value=RC_DEFAULT, step=0.01, format="%.2f")
    with ga_col2:
        rm = st.number_input("R_m", min_value=0.0, max_value=1.0, value=RM_DEFAULT, step=0.01, format="%.2f")

with mid:
    st.markdown('<div class="section-title">Metaheuristic parameters</div>', unsafe_allow_html=True)
    n_particles = st.slider("Population (N)", min_value=5, max_value=100, value=10, step=1)
    n_iterations = st.slider("Max iteration (T)", min_value=5, max_value=200, value=20, step=1)
    n_runs = st.slider("Run", min_value=1, max_value=50, value=15, step=1)

with right:
    st.write("")
    st.write("")
    evaluate_clicked = st.button("Evaluation", use_container_width=True)

if "tp8_result" not in st.session_state:
    st.session_state.tp8_result = None

if evaluate_clicked or st.session_state.tp8_result is None:
    datasets = load_datasets()
    x, y = datasets[dataset_name]
    st.session_state.tp8_result = run_ga_multi(
        x=x,
        y=y,
        alpha=alpha,
        n_particles=n_particles,
        n_iterations=n_iterations,
        selection_mode=selection_mode,
        crossover_mode=crossover_mode,
        replacement_mode=replacement_mode,
        rc=rc,
        rm=rm,
        n_runs=n_runs,
        seed=42,
    )

if st.session_state.tp8_result is not None:
    result = st.session_state.tp8_result
    best_run = result["best_run"]

    with right:
        st.markdown(f"**Best** - {result['best']:.4f},")
        st.markdown(f"**Mean (average error)** - {result['mean']:.4f},")
        st.markdown(f"**Accuracy** - {best_run['best_accuracy']:.2f},")
        st.markdown(f"**Selected** - {result['selected_display']},")
        st.markdown(f"**STD** - {result['std']:.4f},")

    chart1, chart2, chart3, _ = st.columns(4)
    with chart1:
        st.pyplot(make_line_plot(best_run["convergence"], "Convergence Curve", "Fitness", "#f66a6a"))
    with chart2:
        st.pyplot(make_line_plot(best_run["first_solution_traj"], "Trajectory of 1st solution", "x1", "#7ed957"))
    with chart3:
        st.pyplot(make_line_plot(best_run["average_fitness"], "Average Fitness of population", "Fitness", "#5b7cff"))

st.markdown('</div>', unsafe_allow_html=True)
