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

        # population size: moderate for exploration
        pop_size = max(4, min(10 * dim, budget // 4))
        if pop_size > budget:
            pop_size = budget

        # Latin Hypercube Sampling for initial population
        def lhs_sample(n, d, lb, ub):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                samples[:, j] = (perm + rng.uniform(0, 1, n)) / n
            return lb + samples * (ub - lb)

        pop = lhs_sample(pop_size, dim, lb, ub)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # initial evaluation
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

        # DE parameters: explorative
        F = 0.9
        CR = 0.2

        # budget split: 80% for DE, rest for restarts
        budget_de = int(0.8 * budget)
        if budget_de < pop_size:
            budget_de = budget

        # main DE loop
        while evals < budget_de and evals < budget:
            # check diversity: if std of population is too small, restart
            if pop_size > 1:
                std_pop = np.mean(np.std(pop, axis=0))
                if std_pop < 0.05 * np.mean(ub - lb):
                    # restart with random points, keep best
                    new_pop = np.zeros_like(pop)
                    new_pop[0] = best_x.copy()
                    for i in range(1, pop_size):
                        new_pop[i] = lb + rng.rand(dim) * (ub - lb)
                        val = func(new_pop[i])
                        evals += 1
                        fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_pop[i].copy()
                            report_best(best_val, best_x)
                    pop = new_pop
                    continue

            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                # choose three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                # DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
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

        # remaining budget: restart with random perturbations around best
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            sigma = 0.2 * (ub - lb)
            for _ in range(remaining):
                perturb = rng.normal(0, sigma, dim)
                candidate = best_x + perturb
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x