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

        NP = min(10 * dim, max(4, budget // 2 - 1))
        if NP > budget:
            NP = budget
        NP = max(1, NP)

        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.array([func(pop[i]) for i in range(NP)])
        evals = NP

        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_x = pop[best_idx].copy()
        report_best(best_val, best_x)

        if NP == 1:
            return best_val, best_x

        H = 7
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.9)
        p = 0

        max_gen = (budget - evals) // NP
        gen = 0
        no_improve_count = 0
        prev_best = best_val

        while evals < budget and gen < max_gen + 1:
            # Stagnation restart
            if gen > 0 and no_improve_count >= 10:
                remaining = budget - evals
                needed = NP - 1
                if remaining >= needed:
                    pop[0] = best_x.copy()
                    new_pop = lb + (ub - lb) * rng.rand(NP - 1, dim)
                    for i in range(1, NP):
                        pop[i] = new_pop[i-1]
                    for i in range(1, NP):
                        if evals >= budget:
                            break
                        f = func(pop[i])
                        evals += 1
                        if f < best_val:
                            best_val = f
                            best_x = pop[i].copy()
                            report_best(best_val, best_x)
                    M_F[:] = 0.5
                    M_CR[:] = 0.9
                    p = 0
                    no_improve_count = 0
                    prev_best = best_val

            # sort fitness for pbest selection
            idx_sorted = np.argsort(fitness)
            top_N = max(1, int(NP * 0.2))
            S_F = []
            S_CR = []
            delta_f = []
            for i in range(NP):
                if evals >= budget:
                    break
                r = rng.randint(H)
                F = M_F[r] + 0.1 * rng.standard_cauchy()
                if F <= 0:
                    F = 0.1
                elif F > 1:
                    F = 1.0
                CR = M_CR[r] + 0.1 * rng.randn()
                CR = np.clip(CR, 0, 1)

                # current-to-pbest/1
                pbest_idx = rng.randint(top_N)
                pbest = pop[idx_sorted[pbest_idx]]
                candidates = list(range(NP))
                candidates.remove(i)
                if idx_sorted[pbest_idx] in candidates:
                    candidates.remove(idx_sorted[pbest_idx])
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, 2, replace=False)
                a, b = pop[r1], pop[r2]
                mutant = pop[i] + F * (pbest - pop[i]) + F * (a - b)

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

            if len(S_F) > 0:
                w = np.array(delta_f) / (np.sum(delta_f) + 1e-30)
                M_F[p] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[p] = np.sum(w * np.array(S_CR)**2) / (np.sum(w * np.array(S_CR)) + 1e-30)
                p = (p + 1) % H

            if best_val < prev_best - 1e-15:
                no_improve_count = 0
            else:
                no_improve_count += 1
            prev_best = best_val
            gen += 1

        return best_val, best_x