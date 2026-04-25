import numpy as np
import streamlit as st
from sklearn.datasets import load_digits, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data
def load_data():
    digits = load_digits()
    x_digits, y_digits = digits.data, digits.target

    x_syn, y_syn = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=5,
        n_redundant=10,
        n_classes=2,
        random_state=42,
    )

    return {
        "Digits": (x_digits, y_digits),
        "Synthetic": (x_syn, y_syn),
    }


def get_selected_indices(solution: np.ndarray, sf: int) -> np.ndarray:
    ranked = sorted(enumerate(solution), key=lambda item: (-item[1], item[0]))
    selected = [idx for idx, _ in ranked[:sf]]
    return np.array(sorted(selected), dtype=int)


def objective_function(
    solution: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sf: int,
    alpha: float,
    test_size: float = 0.3,
    random_state: int = 42,
    n_neighbors: int = 5,
) -> dict:
    d = x.shape[1]
    if sf <= 0 or sf > d:
        raise ValueError(f"SF must be in [1, {d}]")

    if solution.shape[0] != d:
        raise ValueError(f"Solution length must be {d}, got {solution.shape[0]}")

    selected_indices = get_selected_indices(solution, sf)
    x_selected = x[:, selected_indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x_selected,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = 1.0 - accuracy
    f2 = sf / d
    fitness = alpha * f1 + (1 - alpha) * f2

    return {
        "fitness": float(fitness),
        "accuracy": float(accuracy),
        "selected_features": selected_indices,
    }


def generate_random_solution(dimension: int) -> np.ndarray:
    return np.random.default_rng().random(dimension)


def format_solution(solution: np.ndarray) -> str:
    return " | ".join(f"{value:.2f}" for value in solution)


def format_indices(indices: np.ndarray) -> str:
    return " | ".join(str(i) for i in indices)


def evaluate_and_store(dataset_name: str, sf: int, alpha: float):
    datasets = load_data()
    x, y = datasets[dataset_name]
    solution = st.session_state.solutions[dataset_name]

    result = objective_function(solution, x, y, sf=sf, alpha=alpha)

    st.session_state.last_result = {
        "dataset": dataset_name,
        "sf": sf,
        "alpha": alpha,
        "solution": solution.copy(),
        **result,
    }


def init_state():
    datasets = load_data()
    if "solutions" not in st.session_state:
        st.session_state.solutions = {}

    for name, (x, _) in datasets.items():
        d = x.shape[1]
        existing = st.session_state.solutions.get(name)
        if existing is None or existing.shape[0] != d:
            st.session_state.solutions[name] = generate_random_solution(d)

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = "Synthetic"
    if "sf" not in st.session_state:
        st.session_state.sf = 5
    if "alpha" not in st.session_state:
        st.session_state.alpha = 0.9

    if st.session_state.last_result is None:
        evaluate_and_store("Synthetic", sf=5, alpha=0.9)


st.set_page_config(page_title="TP5 - Feature Selection with PSO", layout="wide")
st.markdown(
    """
<style>
div.tp5-panel {
    border: 2px solid #1f1f1f;
    border-radius: 0;
    padding: 14px 14px 10px 14px;
    margin-top: 6px;
}
div.tp5-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 22px;
}
</style>
""",
    unsafe_allow_html=True,
)

init_state()

st.markdown('<div class="tp5-panel">', unsafe_allow_html=True)
st.markdown('<div class="tp5-title">Part 3 \\ Feature Selection with PSO</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1.2, 0.9, 0.9, 1.2, 1.2])
with col1:
    dataset_name = st.radio("Data", ["Synthetic", "Digits"], horizontal=True, key="dataset_name")
with col2:
    sf = st.number_input("Selected Features (SF)", min_value=1, max_value=200, step=1, key="sf")
with col3:
    alpha = st.number_input("α", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="alpha")
with col4:
    st.write("")
    evaluate_button = st.button("Model Evaluation", type="secondary", use_container_width=True)
with col5:
    st.write("")
    reevaluate_button = st.button("Model Re-evaluation", use_container_width=True)

x_current, _ = load_data()[dataset_name]
d_current = x_current.shape[1]
if sf > d_current:
    st.warning(f"For {dataset_name}, D={d_current}. SF set to {d_current}.")
    st.session_state.sf = d_current
    sf = d_current

if reevaluate_button:
    st.session_state.solutions[dataset_name] = generate_random_solution(d_current)
    evaluate_and_store(dataset_name, sf=sf, alpha=alpha)

if evaluate_button:
    evaluate_and_store(dataset_name, sf=sf, alpha=alpha)

st.markdown("##### Solution")
solution_for_view = st.session_state.solutions[dataset_name]
solution_text = format_solution(solution_for_view)

if st.session_state.last_result is not None and st.session_state.last_result["dataset"] == dataset_name:
    result = st.session_state.last_result
    indices_text = format_indices(result["selected_features"])

    full_text = (
        f"Solution:\n{solution_text}\n\n"
        f"Indices of selected features:\n{indices_text}"
    )
    st.text_area("", value=full_text, height=180, label_visibility="collapsed")

    st.markdown(
        f"**Fitness** — {result['fitness']:.4f}, "
        f"**Accuracy** — {result['accuracy']:.4f}, "
        f"**Selected Features** — {len(result['selected_features'])}"
    )
else:
    st.text_area("", value=f"Solution:\n{solution_text}\n\nIndices of selected features:\n", height=180, label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)
