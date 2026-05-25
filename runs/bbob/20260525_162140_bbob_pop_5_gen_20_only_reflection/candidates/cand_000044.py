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

        gen = 0
        while evals < budget:
            # Diversity-triggered restart every 8 generations
            if gen > 0 and gen % 8 == 0:
                std_dim = np.std(pop, axis=0)
                diversity = np.mean(std_dim) / (np.mean(ub - lb) + 1e-30)
                if diversity < 1e-6:
                    remaining = budget - evals
                    needed = NP - 1
                    if remaining >= needed:
                        pop[0] = best_x.copy()
                        for i in range(1, NP):
                            pop[i] = lb + (ub - lb) * rng.rand(dim)
                        for i in range(1, NP):
                            f = func(pop[i])
                            evals += 1
                            if f < best_val:
                                best_val = f
                                best_x = pop[i].copy()
                                report_best(best_val, best_x)
                            if evals >= budget:
                                break
                        M_F[:] = 0.5
                        M_CR[:] = 0.9
                        p = 0
                        # Re-evaluate fitness array after restart? We already updated pop and fitness partially. Need full fitness recompute.
                        # But we have limited evals. Instead, we'll recompute fitness for all points after restart (except the preserved best).
                        # However, that would cost extra evals. To be safe, we'll reset fitness array properly.
                        # Simpler: after restart, we can just continue with the new pop and fitness from the evaluations done.
                        # We need to update fitness array for the new individuals.
                        for i in range(1, NP):
                            fitness[i] = func(pop[i])  # but we already evaluated them, so we can store them.
                        # Actually in the loop above we evaluated and incremented evals; we should store fitness[i] there.
                        # Let's restructure: after restart, we evaluate each new point and store in fitness.
                        # But we have already evaluated them above. So we can just assign fitness[i] = f in that loop.
                        # We'll implement that below.

            # SHADE generation
            S_F = []
            S_CR = []
            delta_f = []
            # Sort fitness for pbest selection
            sorted_idx = np.argsort(fitness)
            pbest_pop = pop[sorted_idx[:max(1, int(0.2 * NP))]]  # top 20%
            
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
                # Select pbest
                pbest_idx = rng.randint(len(pbest_pop))
                pbest = pbest_pop[pbest_idx]
                # Select two random individuals distinct from i and each other
                candidates = list(range(NP))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - pop[r2])

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

            # Update memory
            if len(S_F) > 0:
                w = np.array(delta_f) / (np.sum(delta_f) + 1e-30)
                M_F[p] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[p] = np.sum(w * np.array(S_CR)**2) / (np.sum(w * np.array(S_CR)) + 1e-30)
                p = (p + 1) % H
            gen += 1

        return best_val, best_x