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
        if NP > budget:
            NP = budget
        NP = max(1, NP)

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.array([func(pop[i]) for i in range(NP)])
        evals = NP

        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_x = pop[best_idx].copy()
        report_best(best_val, best_x)

        if NP == 1:
            return best_val, best_x

        # SHADE memory
        H = 5
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.9)
        p = 0

        max_gen = (budget - evals) // NP
        gen = 0
        while evals < budget and gen < max_gen + 1:
            # Diversity-based restart every 8 generations
            if gen > 0 and gen % 8 == 0:
                std_dim = np.std(pop, axis=0)
                diversity = np.mean(std_dim) / np.mean(ub - lb)
                if diversity < 1e-6:
                    remaining = budget - evals
                    needed = NP - 1  # keep best
                    if remaining >= needed:
                        pop[0] = best_x.copy()
                        new_pop = lb + (ub - lb) * rng.rand(NP - 1, dim)
                        for i in range(1, NP):
                            pop[i] = new_pop[i - 1]
                        for i in range(1, NP):
                            f = func(pop[i])
                            evals += 1
                            if f < best_val:
                                best_val = f
                                best_x = pop[i].copy()
                                report_best(best_val, best_x)
                        M_F[:] = 0.5
                        M_CR[:] = 0.9
                        p = 0

            # DE generation
            S_F = []
            S_CR = []
            delta_f = []
            for i in range(NP):
                if evals >= budget:
                    break
                # Sample F and CR from memory
                r = rng.randint(H)
                F = M_F[r] + 0.1 * rng.standard_cauchy()
                if F <= 0:
                    F = 0.1
                elif F > 1:
                    F = 1.0
                CR = M_CR[r] + 0.1 * rng.randn()
                CR = np.clip(CR, 0, 1)

                # Select three distinct random indices not equal to i
                candidates = list(range(NP))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                idx = rng.choice(candidates, 3, replace=False)
                a, b, c = pop[idx[0]], pop[idx[1]], pop[idx[2]]
                mutant = a + F * (b - c)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    delta_f.append(fitness[i] - f_trial)
                    S_F.append(F)
                    S_CR.append(CR)
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_val:
                        best_val = f_trial
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Update memory with successful parameters
            if len(S_F) > 0:
                w = np.array(delta_f) / (np.sum(delta_f) + 1e-30)
                M_F[p] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[p] = np.sum(w * np.array(S_CR)**2) / (np.sum(w * np.array(S_CR)) + 1e-30)
                p = (p + 1) % H
            gen += 1

        return best_val, best_x