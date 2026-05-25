import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10*dim, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        F = self.F
        CR = self.CR
        rng = self.rng

        # Latin Hypercube Sampling initialization
        pop = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = rng.permutation(pop_size)
            pop[:, j] = lb[j] + (perm + rng.rand(pop_size)) / pop_size * (ub[j] - lb[j])
        pop = np.clip(pop, lb, ub)

        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        while evals < self.budget:
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                jrand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == jrand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i, j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    improved_this_gen = True

            # Restart condition
            if not improved_this_gen and evals < self.budget:
                if pop_size > 1:
                    mean_pos = np.mean(pop, axis=0)
                    distances = np.linalg.norm(pop - mean_pos, axis=1)
                    diversity = np.mean(distances)
                else:
                    diversity = 0.0
                bound_span = np.linalg.norm(ub - lb)
                if diversity < 0.01 * bound_span and pop_size > 1:
                    num_restart = pop_size // 2
                    sorted_indices = np.argsort(fitness)
                    best_idx = sorted_indices[0]
                    candidates = [i for i in range(pop_size) if i != best_idx]
                    rng.shuffle(candidates)
                    to_restart = candidates[:num_restart]
                    for idx in to_restart:
                        if evals >= self.budget:
                            break
                        pop[idx] = lb + rng.rand(dim) * (ub - lb)
                        val = func(pop[idx])
                        evals += 1
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = pop[idx].copy()
                            report_best(best_val, best_x)

        return best_val, best_x