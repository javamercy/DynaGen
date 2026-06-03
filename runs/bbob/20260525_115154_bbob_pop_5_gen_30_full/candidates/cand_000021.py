class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        import numpy as np
        rng = np.random.RandomState(self.seed)
        dim = self.dim
        budget = self.budget
        lb = func.bounds.lb
        ub = func.bounds.ub

        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)

        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # Success-history memory for F and CR
        mem_f = np.full(5, 0.5)
        mem_cr = np.full(5, 0.8)
        mem_idx = 0

        # Track initial diversity for restart
        init_std = np.std(pop, axis=0).mean()

        while evals < budget:
            # Adaptive F and CR per individual
            F_array = np.zeros(pop_size)
            CR_array = np.zeros(pop_size)
            for i in range(pop_size):
                idx = rng.randint(len(mem_f))
                F_array[i] = rng.uniform(mem_f[idx] - 0.1, mem_f[idx] + 0.1)
                F_array[i] = np.clip(F_array[i], 0.1, 0.9)
                CR_array[i] = rng.uniform(mem_cr[idx] - 0.1, mem_cr[idx] + 0.1)
                CR_array[i] = np.clip(CR_array[i], 0.0, 1.0)

            new_pop = np.empty_like(pop)
            new_fitness = np.empty(pop_size)
            success_F = []
            success_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                mutant = pop[a] + F_array[i] * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR_array[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_fit
                    success_F.append(F_array[i])
                    success_CR.append(CR_array[i])
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            pop = new_pop
            fitness = new_fitness

            # Update memory with success parameters
            if len(success_F) > 0:
                weights = np.array([1.0] * len(success_F))
                mean_F = np.average(success_F, weights=weights)
                mean_CR = np.average(success_CR, weights=weights)
                mem_f[mem_idx] = mean_F
                mem_cr[mem_idx] = mean_CR
                mem_idx = (mem_idx + 1) % len(mem_f)

            # Check diversity for restart
            if evals < budget:
                current_std = np.std(pop, axis=0).mean()
                if current_std < 0.1 * init_std and evals < budget - pop_size:
                    # Keep best, reinitialize 70% of rest
                    keep = np.argmin(fitness)
                    n_restart = int(0.7 * pop_size)
                    idx_restart = rng.choice([i for i in range(pop_size) if i != keep], size=n_restart, replace=False)
                    for idx in idx_restart:
                        pop[idx] = rng.uniform(lb, ub, size=dim)
                        if evals >= budget:
                            break
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < best_val:
                            best_val = fitness[idx]
                            best_x = pop[idx].copy()
                            report_best(best_val, best_x)

        return best_val, best_x