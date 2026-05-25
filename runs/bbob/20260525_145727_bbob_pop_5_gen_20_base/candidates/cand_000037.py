import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(20, budget // 2, 2 * dim))
        if self.pop_size > budget:
            self.pop_size = budget
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.0
        self.CR_u = 1.0
        self.tau1 = 0.2
        self.tau2 = 0.2

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        rng = self.rng
        budget = self.budget

        # Initialize population
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        F = rng.uniform(self.F_l, self.F_u, pop_size)
        CR = rng.uniform(self.CR_l, self.CR_u, pop_size)

        best_x = None
        best_val = np.inf
        evals = 0

        # Evaluate initial population
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

        if evals >= budget:
            return best_val, best_x

        # Compute budget splits
        remaining = budget - evals
        if remaining > 0:
            DE_budget = int(0.7 * remaining)
            local_budget = remaining - DE_budget
        else:
            DE_budget = 0
            local_budget = 0

        # Self-adaptive DE phase
        while evals < budget and evals < evals + DE_budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Generate new F and CR
                F_new = F[i]
                CR_new = CR[i]
                if rng.rand() < self.tau1:
                    F_new = rng.uniform(self.F_l, self.F_u)
                if rng.rand() < self.tau2:
                    CR_new = rng.uniform(self.CR_l, self.CR_u)

                # Select two distinct random indices
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]

                # Mutation: DE/current-to-best/1
                mutant = pop[i] + F_new * (best_x - pop[i]) + F_new * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                # Exponential crossover
                j_start = rng.randint(dim)
                L = 1
                while rng.rand() < CR_new and L < dim:
                    L += 1
                trial = pop[i].copy()
                for k in range(L):
                    j = (j_start + k) % dim
                    trial[j] = mutant[j]

                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    F[i] = F_new
                    CR[i] = CR_new
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Local search phase from best
        if local_budget > 0:
            step_init = 0.1 * (ub - lb)
            for i in range(local_budget):
                if evals >= budget:
                    break
                progress = i / max(1, local_budget)
                step = step_init * np.exp(-3 * progress)
                trial = best_x + rng.randn(dim) * step
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                    # Optionally update population: replace worst
                    worst_idx = np.argmax(fitness)
                    if val < fitness[worst_idx]:
                        pop[worst_idx] = trial
                        fitness[worst_idx] = val

        return best_val, best_x