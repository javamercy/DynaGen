BBOB_BASELINES = {
    "random_search": """import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        best_x = lb + self.rng.random(self.dim) * (ub - lb)
        best_value = float(func(best_x))
        report_best(best_value, best_x)
        for _ in range(1, self.budget):
            x = lb + self.rng.random(self.dim) * (ub - lb)
            value = float(func(x))
            if value < best_value:
                best_value = value
                best_x = x.copy()
                report_best(best_value, best_x)
        return best_value, best_x
""",
    "differential_evolution": """import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = int(budget)
        self.dim = int(dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)
        span = ub - lb
        pop_size = min(max(4, 4 * self.dim), self.budget)
        if pop_size < 4:
            best_x = lb + self.rng.random(self.dim) * span
            best_value = float(func(best_x))
            report_best(best_value, best_x)
            for _ in range(1, self.budget):
                x = lb + self.rng.random(self.dim) * span
                value = float(func(x))
                if value < best_value:
                    best_value = value
                    best_x = x.copy()
                    report_best(best_value, best_x)
            return best_value, best_x

        population = lb + self.rng.random((pop_size, self.dim)) * span
        fitness = np.empty(pop_size, dtype=float)
        best_index = 0
        best_value = np.inf
        evaluations = 0
        for i in range(pop_size):
            fitness[i] = float(func(population[i]))
            evaluations += 1
            if fitness[i] < best_value:
                best_value = float(fitness[i])
                best_index = i
                report_best(best_value, population[i].copy())

        f = 0.7
        cr = 0.9
        while evaluations < self.budget:
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break
                choices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = self.rng.choice(choices, 3, replace=False)
                mutant = population[r1] + f * (population[r2] - population[r3])
                mutant = np.clip(mutant, lb, ub)
                cross = self.rng.random(self.dim) < cr
                cross[self.rng.integers(0, self.dim)] = True
                trial = np.where(cross, mutant, population[i])
                value = float(func(trial))
                evaluations += 1
                if value <= fitness[i]:
                    population[i] = trial
                    fitness[i] = value
                    if value < best_value:
                        best_value = value
                        best_index = i
                        report_best(best_value, population[i].copy())
            f = min(0.95, f + 0.02)
        return best_value, population[best_index].copy()
""",
}


def get_bbob_baseline_code(name: str) -> str:
    try:
        return BBOB_BASELINES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown BBOB baseline: {name}") from exc
