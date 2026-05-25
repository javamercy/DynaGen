import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # population size adaptive
        pop_size = max(4 * dim, 20)
        pop_size = min(pop_size, budget // 2)
        if pop_size < 1:
            pop_size = 1
        self.pop_size = pop_size
        self.max_generations = (budget - pop_size) // pop_size if pop_size > 0 else 0
        self.restart_threshold = max(5, 2 * dim)
        # PSO parameters
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = self.pop_size
        if n <= 0:
            # fallback: random search
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

        # Initialize population
        pop = np.random.uniform(lb, ub, (n, dim))
        vel = np.zeros((n, dim))
        pbest = pop.copy()
        pbest_fit = np.full(n, np.inf)
        gbest_val = np.inf
        gbest_x = None
        evals = 0

        # Evaluate initial population
        for i in range(n):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            pbest_fit[i] = val
            pbest[i] = x.copy()
            if val < gbest_val:
                gbest_val = val
                gbest_x = x.copy()
                report_best(gbest_val, gbest_x)

        # Main loop
        no_improve_streak = 0
        generation = 0
        while evals < self.budget and generation < self.max_generations:
            improved_in_gen = False
            for i in range(n):
                if evals >= self.budget:
                    break
                # Update velocity
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                vel[i] = self.w * vel[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (gbest_x - pop[i])
                # Clamp velocity
                max_vel = 0.5 * (ub - lb)
                vel[i] = np.clip(vel[i], -max_vel, max_vel)
                # Update position
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                # Evaluate
                val = func(pop[i])
                evals += 1
                # Update personal best
                if val < pbest_fit[i]:
                    pbest_fit[i] = val
                    pbest[i] = pop[i].copy()
                    if val < gbest_val:
                        gbest_val = val
                        gbest_x = pop[i].copy()
                        report_best(gbest_val, gbest_x)
                        improved_in_gen = True
            # Update no improvement streak
            if improved_in_gen:
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            # Restart if no improvement for threshold generations
            if no_improve_streak >= self.restart_threshold:
                # Reinitialize all particles except best
                new_pop = np.random.uniform(lb, ub, (n, dim))
                new_vel = np.zeros((n, dim))
                if gbest_x is not None:
                    new_pop[0] = gbest_x.copy()
                else:
                    new_pop[0] = np.random.uniform(lb, ub, dim)
                # Evaluate new particles (except the first which already has known value)
                new_pbest = new_pop.copy()
                new_pbest_fit = np.full(n, np.inf)
                new_pbest_fit[0] = gbest_val
                for i in range(1, n):
                    if evals >= self.budget:
                        break
                    x = new_pop[i]
                    val = func(x)
                    evals += 1
                    new_pbest_fit[i] = val
                    new_pbest[i] = x.copy()
                    if val < gbest_val:
                        gbest_val = val
                        gbest_x = x.copy()
                        report_best(gbest_val, gbest_x)
                pop = new_pop
                vel = new_vel
                pbest = new_pbest
                pbest_fit = new_pbest_fit
                no_improve_streak = 0

            generation += 1

        return gbest_val, gbest_x