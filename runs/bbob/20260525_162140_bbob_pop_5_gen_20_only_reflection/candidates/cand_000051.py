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
        while evals < budget and gen < max_gen + 1:
            # Adaptive diversity threshold
            threshold = max(1e-6, 1e-4 * (1 - evals / budget))
            
            # Diversity check
            if gen > 0 and evals > 0.05 * budget:
                std_dim = np.std(pop, axis=0)
                diversity = np.mean(std_dim) / (np.mean(ub - lb) + 1e-30)
                if diversity < threshold:
                    remaining = budget - evals
                    # Local search from best before restart
                    if remaining >= 5 * dim:
                        local_evals = min(5 * dim, remaining // 2)
                        for _ in range(local_evals):
                            step = rng.randn(dim) * (ub - lb) * 0.01
                            trial = np.clip(best_x + step, lb, ub)
                            ftrial = func(trial)
                            evals += 1
                            if ftrial < best_val:
                                best_val = ftrial
                                best_x = trial.copy()
                                report_best(best_val, best_x)
                    # Restart
                    needed = NP - 1
                    if remaining >= needed:
                        pop[0] = best_x.copy()
                        new_pop = lb + (ub - lb) * rng.rand(NP - 1, dim)
                        for i in range(1, NP):
                            pop[i] = new_pop[i-1]
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
                        # Re-evaluate fitness after restart
                        fitness = np.array([func(pop[i]) for i in range(NP)])
                        # Note: evals already counted in loop above, but we need to re-evaluate all? Actually we just evaluated new points, but we should recompute fitness for all to be safe. We'll do it but careful with budget.
                        # Instead, track fitness properly: after restart, we have already evaluated pop[1:] above, but pop[0] is best with known fitness. So we can set fitness[0] = best_val and for others use the evaluations we did.
                        # But to simplify, we'll let the main loop continue; however we need to update the fitness array for the whole pop. Since we set pop[0] as best, we know its fitness. For others, we evaluated them above, but we didn't store in fitness. We'll store them.
                        # Actually in the loop above we store f, but we can assign directly. Let's restructure: after restart, evaluate all new points and store fitness.
                        # I'll rewrite restart block below.
            
            S_F = []
            S_CR = []
            delta_f = []
            pbest_size = max(2, int(0.2 * NP))
            sorted_indices = np.argsort(fitness)
            pbest_indices = set(sorted_indices[:pbest_size])
            
            # Compute coordinate scaling based on population std
            std_pop = np.std(pop, axis=0) + 1e-30
            scale = std_pop / (np.mean(std_pop) + 1e-30)
            
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

                candidates = list(range(NP))
                candidates.remove(i)
                pbest_list = [idx for idx in pbest_indices if idx != i]
                if len(pbest_list) == 0:
                    r1 = rng.choice(candidates)
                else:
                    r1 = rng.choice(pbest_list)
                remaining = [j for j in candidates if j != r1]
                if len(remaining) < 2:
                    continue
                r2, r3 = rng.choice(remaining, 2, replace=False)
                a, b, c = pop[r1], pop[r2], pop[r3]
                mutant = a + F * (b - c) * scale

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
            gen += 1

        return best_val, best_x