import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Population size
        NP = min(10 * dim, max(4, budget // 2 - 1))
        if NP < 4:
            NP = 4
        if budget < NP:
            NP = budget

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.zeros(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_val = fitness[best_idx]
        report_best(best_val, best_x)

        func_evals = NP

        # SHADE parameters
        H = 5
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        mem_idx = 0
        success_CR = []
        success_F = []

        # Restart threshold based on typical scale
        scale = np.mean(ub - lb)
        diversity_threshold = 1e-6 * scale

        # Main loop
        while func_evals < budget:
            # Check diversity for restart
            if np.std(pop) < diversity_threshold and func_evals < budget - NP:
                # Reinitialize population except best
                new_pop = lb + (ub - lb) * rng.rand(NP-1, dim)
                new_fit = np.zeros(NP-1)
                for i in range(NP-1):
                    new_fit[i] = func(new_pop[i])
                func_evals += NP-1
                pop = np.vstack([best_x.reshape(1, -1), new_pop])
                fitness = np.array([best_val] + new_fit.tolist())
                # Reset memory?
                M_CR = np.full(H, 0.5)
                M_F = np.full(H, 0.5)
                mem_idx = 0
                success_CR = []
                success_F = []
                if func_evals >= budget:
                    break

            max_gen = (budget - func_evals) // NP
            if max_gen == 0:
                break

            for gen in range(max_gen):
                # Generate parameters for each individual
                F_vals = np.zeros(NP)
                CR_vals = np.zeros(NP)
                for i in range(NP):
                    r = rng.randint(H)
                    CR_vals[i] = rng.normal(M_CR[r], 0.1)
                    CR_vals[i] = np.clip(CR_vals[i], 0.0, 1.0)
                    F_vals[i] = rng.standard_cauchy() * 0.1 + M_F[r]
                    F_vals[i] = np.clip(F_vals[i], 0.0, 1.0)

                for i in range(NP):
                    # Select a,b,c distinct from pop (excluding i? We'll exclude i but it's fine)
                    candidates = list(range(NP))
                    candidates.remove(i)
                    idx = rng.choice(candidates, 3, replace=False)
                    a, b, c = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                    mutant = a + F_vals[i] * (b - c)
                    # Crossover
                    trial = pop[i].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR_vals[i] or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, lb, ub)
                    trial_fitness = func(trial)
                    func_evals += 1
                    if trial_fitness < fitness[i]:
                        pop[i] = trial
                        fitness[i] = trial_fitness
                        success_CR.append(CR_vals[i])
                        success_F.append(F_vals[i])
                        if trial_fitness < best_val:
                            best_val = trial_fitness
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                    if func_evals >= budget:
                        break
                if func_evals >= budget:
                    break
                # Update memory with successful parameters
                if len(success_CR) > 0:
                    # Compute weighted Lehmer mean for F
                    w = np.array([1.0] * len(success_F))
                    wF = np.sum(w * success_F**2) / np.sum(w * success_F)
                    # Arithmetic mean for CR
                    wCR = np.mean(success_CR)
                    M_F[mem_idx] = wF
                    M_CR[mem_idx] = wCR
                    mem_idx = (mem_idx + 1) % H
                    success_CR = []
                    success_F = []
            # End generation loop
            if func_evals >= budget:
                break
        return best_val, best_x