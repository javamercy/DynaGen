import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.pop_size = max(self.pop_size, 1)
        self.max_generations = (budget - self.pop_size) // self.pop_size if self.pop_size > 0 else 0
        self.restart_threshold = max(5, dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        if pop_size <= 0:
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < self.budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        vel = np.zeros((pop_size, dim))
        pbest = pop.copy()
        pbest_val = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            pbest_val[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        w_max = 0.9
        w_min = 0.4
        c1 = 2.0
        c2 = 2.0
        no_improve = 0
        generation = 0
        while evals < self.budget and generation < self.max_generations:
            w = w_max - (w_max - w_min) * generation / max(1, self.max_generations - 1)
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (best_x - pop[i])
                # optional velocity clamping? we skip for simplicity
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                val = func(pop[i])
                evals += 1
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = pop[i].copy()
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                        improved_this_gen = True
            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.restart_threshold:
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    new_pop[0] = best_x.copy()
                new_vel = np.zeros((pop_size, dim))
                new_pbest = new_pop.copy()
                new_pbest_val = np.full(pop_size, np.inf)
                new_pbest_val[0] = best_val
                for i in range(1, pop_size):
                    if evals >= self.budget:
                        break
                    x = new_pop[i]
                    val = func(x)
                    evals += 1
                    new_pbest_val[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                vel = new_vel
                pbest = new_pbest
                pbest_val = new_pbest_val
                no_improve = 0
            generation += 1
        return best_val, best_x