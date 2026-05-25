import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        pop_size = min(30, max(5, budget // 2))
        if budget < pop_size:
            pop_size = max(1, budget)

        if budget <= 0:
            best_x = np.zeros(dim)
            best_f = float('inf')
            report_best(best_f, best_x)
            return best_f, best_x

        # Initialize positions and velocities
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        vel = rng.uniform(-0.5 * (ub - lb), 0.5 * (ub - lb), size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1
            if budget <= 0:
                best_idx = np.argmin(pop_f[:i+1])
                best_x = pop[best_idx].copy()
                best_f = pop_f[best_idx]
                report_best(best_f, best_x)
                return best_f, best_x

        # Personal bests
        pbest = pop.copy()
        pbest_f = pop_f.copy()
        
        # Global best
        gbest_idx = np.argmin(pbest_f)
        gbest_x = pbest[gbest_idx].copy()
        gbest_f = pbest_f[gbest_idx]
        report_best(gbest_f, gbest_x)

        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        stagnation_limit = max(1, budget // (4 * pop_size))
        stagnation_counter = 0
        gen = 0

        while budget > 0:
            w = w_start - (w_start - w_end) * gen / (budget / pop_size + 1)  # decaying
            for i in range(pop_size):
                if budget <= 0:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest_x - pop[i])
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                pop_f[i] = func(pop[i])
                budget -= 1
                if pop_f[i] < pbest_f[i]:
                    pbest[i] = pop[i].copy()
                    pbest_f[i] = pop_f[i]
                    if pop_f[i] < gbest_f:
                        gbest_x = pop[i].copy()
                        gbest_f = pop_f[i]
                        report_best(gbest_f, gbest_x)
                        improved = True
            # Check stagnation
            if budget > 0 and stagnation_counter >= stagnation_limit and budget >= pop_size:
                # Restart: keep gbest and a perturbed copy, reinitialize others uniformly
                pert_std = 0.05 * (ub - lb)
                gbest_pert = np.clip(gbest_x + rng.randn(dim) * pert_std, lb, ub)
                f_pert = func(gbest_pert)
                budget -= 1
                # Reinitialize remaining particles
                new_size = pop_size - 2
                new_pop = rng.uniform(lb, ub, size=(new_size, dim))
                new_vel = rng.uniform(-0.5 * (ub - lb), 0.5 * (ub - lb), size=(new_size, dim))
                new_pbest = new_pop.copy()
                new_pbest_f = np.full(new_size, np.inf)
                for k in range(new_size):
                    if budget <= 0:
                        break
                    new_pbest_f[k] = func(new_pop[k])
                    budget -= 1
                # Assemble new swarm
                pop = np.vstack((gbest_x.reshape(1, -1), gbest_pert.reshape(1, -1), new_pop))
                vel = np.vstack((np.zeros((1, dim)), np.zeros((1, dim)), new_vel))
                pbest = np.vstack((gbest_x.reshape(1, -1), gbest_pert.reshape(1, -1), new_pbest))
                pbest_f = np.concatenate(([gbest_f], [f_pert], new_pbest_f))
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        
        return gbest_f, gbest_x