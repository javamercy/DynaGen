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

        pop_size = max(4 * dim, 5)
        if pop_size > budget:
            pop_size = budget

        # LHS initialization
        pop = np.empty((pop_size, dim))
        for i in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            pop[:, i] = (perm + u) / pop_size
        pop = lb + pop * (ub - lb)

        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            f = func(x)
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        F_low, F_high = 0.5, 1.0
        CR_low, CR_high = 0.3, 0.9
        stagnation_limit = pop_size * 2
        no_improve = 0

        local_search_period = 5
        step_size = 0.1 * (ub - lb).mean()
        generation = 0

        while evals < budget:
            # Stagnation restart
            if no_improve >= stagnation_limit and evals < budget:
                remaining = budget - evals
                if remaining < pop_size:
                    pop_size = max(1, remaining)
                else:
                    pop_size = min(pop_size, remaining)
                pop = np.empty((pop_size, dim))
                for i in range(dim):
                    perm = rng.permutation(pop_size)
                    u = rng.rand(pop_size)
                    pop[:, i] = (perm + u) / pop_size
                pop = lb + pop * (ub - lb)
                fitness = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    x = pop[i]
                    f = func(x)
                    evals += 1
                    fitness[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                no_improve = 0
                step_size = 0.1 * (ub - lb).mean()
                continue

            # DE generation
            improved_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                F = F_low + (F_high - F_low) * rng.rand()
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                CR = CR_low + (CR_high - CR_low) * rng.rand()
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                # Evaluation
                f = func(trial)
                evals += 1
                if f < fitness[i]:
                    if f < best_f:
                        best_f = f
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                    fitness[i] = f
                    pop[i] = trial.copy()
                    improved_gen = True
                    no_improve = 0
                else:
                    no_improve += 1

            # Local search every few generations
            generation += 1
            if generation % local_search_period == 0 and evals < budget:
                for _ in range(2):
                    if evals >= budget:
                        break
                    direction = rng.randn(dim)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    else:
                        direction = np.zeros(dim)
                    candidate = best_x + step_size * direction
                    candidate = np.clip(candidate, lb, ub)
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                        step_size *= 1.2
                    else:
                        step_size *= 0.85
                    step_size = max(step_size, 1e-10)
                    step_size = min(step_size, (ub - lb).mean() * 0.5)

        return best_f, best_x