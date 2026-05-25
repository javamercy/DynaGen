import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Population size: small, at least 4
        NP = max(4, min(20, int(4 + 3 * np.log(dim))))
        if budget < NP + 1:
            # Very small budget: random search
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            for _ in range(budget - 1):
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                if val < best_val:
                    best_val = val
                    best_x = x
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(ind) for ind in pop])
        calls = NP
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_x = pop[best_idx].copy()
        report_best(best_val, best_x)

        # Parameters
        F_base = 0.8
        CR_amplitude = 0.4
        CR_offset = 0.5
        max_generations = max(1, (budget - NP) // NP)
        initial_span = np.max(ub - lb)
        diversity_threshold = 0.01 * initial_span
        generation = 0
        restart_interval = max(1, int(0.1 * max_generations))

        while calls < budget:
            CR = CR_offset + CR_amplitude * np.sin(2 * np.pi * generation / max_generations)
            F = F_base * (1 - calls / budget)

            for i in range(NP):
                if calls >= budget:
                    break
                idxs = list(range(NP))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                x_best = best_x
                x_curr = pop[i]
                v = x_curr + F * (x_best - x_curr) + F * (pop[r1] - pop[r2])
                v = np.clip(v, lb, ub)
                j_rand = np.random.randint(dim)
                u = np.where(np.random.rand(dim) < CR, v, x_curr)
                u[j_rand] = v[j_rand]
                u = np.clip(u, lb, ub)
                val = func(u)
                calls += 1
                if val < fitness[i]:
                    pop[i] = u
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = u.copy()
                        report_best(best_val, best_x)

            generation += 1

            # Diversity-triggered restart every restart_interval generations
            if generation % restart_interval == 0 and calls < budget:
                sum_dist = 0.0
                count = 0
                for i in range(NP):
                    for j in range(i+1, NP):
                        sum_dist += np.linalg.norm(pop[i] - pop[j])
                        count += 1
                avg_dist = sum_dist / count if count > 0 else 0
                if avg_dist < diversity_threshold:
                    num_restart = max(1, int(0.3 * NP))
                    sorted_idx = np.argsort(fitness)
                    worst_idx = sorted_idx[-num_restart:]
                    for idx in worst_idx:
                        if calls >= budget:
                            break
                        new_ind = np.random.uniform(lb, ub, dim)
                        pop[idx] = new_ind
                        val = func(new_ind)
                        calls += 1
                        fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_ind.copy()
                            report_best(best_val, best_x)

        return best_val, best_x