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

        # Larger population for exploration
        pop_size = max(4, min(20 * dim, budget // 3))
        if pop_size > budget:
            pop_size = budget

        # Budget allocation: 70% for DE phases, 30% for final random exploration
        budget_de = int(0.7 * budget)
        if budget_de < pop_size:
            budget_de = budget

        # Initialize population (first evaluation is guaranteed by initial pop)
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        def evaluate_and_update(x):
            nonlocal evals, best_val, best_x
            if evals >= budget:
                return None
            val = func(x)
            evals += 1
            return val

        # Initial evaluations
        for i in range(pop_size):
            if evals >= budget:
                break
            val = evaluate_and_update(pop[i])
            if val is None:
                break
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        if best_x is None:
            # Should not happen, but fallback
            best_x = pop[0].copy()
            best_val = fitness[0]
            report_best(best_val, best_x)

        # DE parameters: exploratory
        F = 0.9
        CR = 0.3
        stagnation_counter = 0
        max_stagnation = 5  # restart if no improvement for 5 generations
        restart_threshold = budget_de // 2  # at least half of DE budget before restart
        last_improvement_evals = 0

        # Main DE loop with possible restarts
        while evals < budget_de and evals < budget:
            # Check for restart condition
            if evals - last_improvement_evals >= stagnation_counter * pop_size and stagnation_counter >= max_stagnation:
                # Restart: reinitialize population except best
                if evals < budget_de - pop_size:
                    new_pop = lb + rng.rand(pop_size - 1, dim) * (ub - lb)
                    for i in range(pop_size - 1):
                        if evals >= budget_de or evals >= budget:
                            break
                        val = evaluate_and_update(new_pop[i])
                        if val is None:
                            break
                        pop[i] = new_pop[i]
                        fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_pop[i].copy()
                            report_best(best_val, best_x)
                    # Keep best solution at last position
                    pop[-1] = best_x
                    fitness[-1] = best_val
                    stagnation_counter = 0
                    last_improvement_evals = evals
                    continue

            # Normal DE generation
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]  # for rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                trial = np.clip(trial, lb, ub)

                val = evaluate_and_update(trial)
                if val is None:
                    break
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

        # Final random exploration around best
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            sigma = 0.2 * (ub - lb)
            for _ in range(remaining):
                perturb = rng.normal(0, sigma, dim)
                candidate = best_x + perturb
                candidate = np.clip(candidate, lb, ub)
                val = evaluate_and_update(candidate)
                if val is None:
                    break
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x