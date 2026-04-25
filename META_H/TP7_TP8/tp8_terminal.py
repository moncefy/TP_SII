import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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
        fitness = alpha * 1.0
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


def one_point_crossover(parent1, parent2, rc, rng):
    if len(parent1) < 2 or rng.random() >= rc:
        return parent1.copy(), parent2.copy()
    k = rng.integers(1, len(parent1))
    child1 = np.concatenate([parent1[:k], parent2[k:]])
    child2 = np.concatenate([parent2[:k], parent1[k:]])
    return child1, child2


def two_point_crossover(parent1, parent2, rc, rng):
    if len(parent1) < 3:
        return one_point_crossover(parent1, parent2, rc, rng)
    if rng.random() >= rc:
        return parent1.copy(), parent2.copy()
    k1 = rng.integers(1, len(parent1) - 1)
    k2 = rng.integers(k1 + 1, len(parent1))
    child1 = np.concatenate([parent1[:k1], parent2[k1:k2], parent1[k2:]])
    child2 = np.concatenate([parent2[:k1], parent1[k1:k2], parent2[k2:]])
    return child1, child2


def mutate(child, rm, rng):
    mutated = child.copy()
    for j in range(mutated.shape[0]):
        if rng.random() < rm:
            mutated[j] = 1 - mutated[j]
    return mutated


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
                child1, child2 = one_point_crossover(parent1, parent2, rc, rng)
            elif crossover_mode == "2-Point":
                child1, child2 = two_point_crossover(parent1, parent2, rc, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = mutate(child1, rm, rng)
            child2 = mutate(child2, rm, rng)
            children.append(child1)
            children.append(child2)

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
        "all_runs": runs,
    }


def prompt_text(message, default):
    value = input(f"{message} [{default}]: ").strip()
    return value if value else default


def prompt_int(message, default, minimum=None, maximum=None):
    while True:
        raw = prompt_text(message, default)
        try:
            value = int(raw)
            if minimum is not None and value < minimum:
                print(f"Value must be >= {minimum}")
                continue
            if maximum is not None and value > maximum:
                print(f"Value must be <= {maximum}")
                continue
            return value
        except ValueError:
            print("Enter an integer.")


def prompt_float(message, default, minimum=None, maximum=None):
    while True:
        raw = prompt_text(message, default)
        try:
            value = float(raw)
            if minimum is not None and value < minimum:
                print(f"Value must be >= {minimum}")
                continue
            if maximum is not None and value > maximum:
                print(f"Value must be <= {maximum}")
                continue
            return value
        except ValueError:
            print("Enter a number.")


def prompt_choice(message, choices, default_index=0):
    default = choices[default_index]
    joined = "/".join(choices)
    while True:
        raw = prompt_text(f"{message} ({joined})", default)
        for choice in choices:
            if raw.lower() == choice.lower():
                return choice
        print(f"Choose one of: {', '.join(choices)}")


def main():
    print("TP8 Genetic Algorithm - terminal mode")
    datasets = load_datasets()

    dataset_name = prompt_choice("Data", ["Synthetic", "Digits"], 0)
    alpha = prompt_float("alpha", 0.99, 0.0, 1.0)

    print("GA parameters")
    selection_mode = prompt_choice("Selection", ["Cumulative", "Random"], 0)
    crossover_mode = prompt_choice("Crossover", ["1-Point", "2-Point", "None"], 0)
    replacement_choice = prompt_choice("Replacement", ["Children", "Best"], 1)
    replacement_mode = "Children" if replacement_choice == "Children" else "Best"

    rc = prompt_float("Rc", 0.7, 0.0, 1.0)
    rm = prompt_float("Rm", 0.10, 0.0, 1.0)

    print("Metaheuristic parameters")
    n_particles = prompt_int("Population (N)", 10, 2, 1000)
    n_iterations = prompt_int("Max iteration (T)", 20, 1, 10000)
    n_runs = prompt_int("Run", 15, 1, 1000)

    x, y = datasets[dataset_name]
    result = run_ga_multi(
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

    best_run = result["best_run"]
    solution = best_run["best_solution"]
    selected_indices = np.where(solution > 0.5)[0]

    print("\nResults:")
    print(f"Best = {result['best']:.4f}")
    print(f"Mean (average error) = {result['mean']:.4f}")
    print(f"Accuracy = {best_run['best_accuracy']:.2f}")
    print(f"Selected = {result['selected_display']}")
    print(f"STD = {result['std']:.4f}")
    print(f"Selected feature indices = {selected_indices.tolist()}")
    print("Solution =")
    print(" | ".join(f"{v:.2f}" for v in solution))


if __name__ == "__main__":
    main()