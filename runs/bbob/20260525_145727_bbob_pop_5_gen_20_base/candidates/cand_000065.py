import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
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

        # population size: smaller for more generations
        pop_size = max(4, min(5*dim, budget // 8))
        if pop_size > budget:
            pop_size = budget

        # allocate 60% for DE, 40% for local search
        budget_de = int(0.6 * budget)
        if budget_de < pop_size:
            budget_de = budget

        # initialization
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

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

        # DE parameters
        F = 0.5
        CR = 0.9

        # main DE loop
        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                # choose two distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]
                # DE/best/1
                mutant = best_x + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
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

        # local refinement: adaptive random search with step size
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            # initial step size as 0.1 * range
            step_size = 0.1 * (ub - lb)
            for _ in range(remaining):
                if evals >= budget:
                    break
                # generate random direction
                direction = rng.normal(0, 1, dim)
                direction = direction / (np.linalg.norm(direction) + 1e-15)
                candidate = best_x + step_size * direction
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    step_size *= 1.2
                    report_best(best_val, best_x)
                else:
                    # try opposite direction
                    if evals < budget:
                        candidate2 = best_x - step_size * direction
                        candidate2 = np.clip(candidate2, lb, ub)
                        val2 = func(candidate2)
                        evals += 1
                        if val2 < best_val:
                            best_val = val2
                            best_x = candidate2.copy()
                            step_size *= 1.2
                            report_best(best_val, best_x)
                        else:
                            step_size *= 0.9
                    else:
                        step_size *= 0.9
                # ensure step size doesn't become too small or too large
                step_size = np.clip(step_size, 1e-12 * (ub - lb), 0.5 * (ub - lb))

        return best_val, best_x