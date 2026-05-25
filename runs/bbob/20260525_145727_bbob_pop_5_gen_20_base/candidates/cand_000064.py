import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        pop_size = max(5, min(15*dim, budget // 5))
        if pop_size > budget:
            pop_size = budget

        budget_phase1 = int(0.6 * budget)
        if budget_phase1 < pop_size:
            budget_phase1 = budget

        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # Initial evaluations
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        CR = 0.9
        generation = 0

        while evals < budget:
            # Phase switching
            if evals < budget_phase1:
                # DE/rand/1/bin
                for i in range(pop_size):
                    if evals >= budget or evals >= budget_phase1:
                        break
                    fraction = evals / budget_phase1
                    F = 0.5 + 0.4 * np.sin(np.pi * fraction)

                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b, c = candidates[:3]

                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, lb, ub)

                    j_rand = rng.randint(dim)
                    trial = np.empty(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                        else:
                            trial[j] = pop[i][j]
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
            else:
                # DE/current-to-best/1/bin with sinusoidal F
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    fraction = (evals - budget_phase1) / (budget - budget_phase1)
                    F = 0.5 + 0.4 * np.sin(np.pi * fraction)

                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b = candidates[:2]

                    mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                    mutant = np.clip(mutant, lb, ub)

                    j_rand = rng.randint(dim)
                    trial = np.empty(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                        else:
                            trial[j] = pop[i][j]
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

            # Diversity maintenance: check variance and reinitialize worst if low
            if evals % (2 * pop_size) == 0:
                var = np.mean(np.var(pop, axis=0))
                if var < 1e-4:
                    # Replace worst half with random points
                    worst_indices = np.argsort(fitness)[-pop_size//2:]
                    for idx in worst_indices:
                        if evals >= budget:
                            break
                        pop[idx] = lb + rng.rand(dim) * (ub - lb)
                        val = func(pop[idx])
                        evals += 1
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = pop[idx].copy()
                            report_best(best_val, best_x)

            generation += 1

        return best_val, best_x