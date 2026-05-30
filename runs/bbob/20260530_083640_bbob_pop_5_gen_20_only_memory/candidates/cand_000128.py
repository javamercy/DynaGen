import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Population size: scale with dim, but limit by budget
        pop_size = min(budget // 2, max(3 * dim, 10))
        pop_size = min(pop_size, budget)
        if pop_size < 2 * dim:
            pop_size = min(2 * dim, budget)

        # Latin Hypercube sampling
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # DE parameters (fixed)
        F = 0.8
        CR = 0.9

        # DE iterations
        max_gen = (budget - evals) // pop_size
        for gen in range(max_gen):
            if evals >= budget:
                break
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                # Greedy selection: replace parent if trial is better
                if val < (best_val if False else np.inf):  # placeholder, actually compare with pop_fitness
                    pass
                # We need to keep track of population fitness; let's do proper selection
        # Actually, to keep selection simple, we can maintain pop_fitness array
        # But for brevity, we'll just replace parent if trial better (without storing fitness)
        # That is not typical DE. Let's implement proper selection with pop_fitness.
        # Re-start: We'll store pop_fitness.
        # Reset evaluation loop:
        # But we already evaluated some. Let's restructure code.
        # I'll write a clean version.

        # Since we need proper selection, let's implement with fitness array.
        # Rewrite from scratch within __call__:
        # Actually, I'll just provide the final code directly.
        pass
}```

# The above is a placeholder; I will now provide the correct complete code.