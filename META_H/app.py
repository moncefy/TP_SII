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
    cumulative = np.cumsum(population, axis=1)
    return np.sum(cumulative**2, axis=1)


def f5(population: np.ndarray) -> np.ndarray:
    x_i = population[:, :-1]
    x_next = population[:, 1:]
    return np.sum(100 * (x_next - x_i**2) ** 2 + (x_i - 1) ** 2, axis=1)


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
        "range": (-100.0, 100.0),
        "latex": r"f(x)=\sum_{i=1}^{D}\left(\sum_{j=1}^{i}x_j\right)^2",
        "fn": f2,
    },
    "F5-UM": {
        "short": "F5",
        "range": (-30.0, 30.0),
        "latex": r"f(x)=\sum_{i=1}^{D-1}\left[100(x_{i+1}-x_i^2)^2+(x_i-1)^2\right]",
        "fn": f5,
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