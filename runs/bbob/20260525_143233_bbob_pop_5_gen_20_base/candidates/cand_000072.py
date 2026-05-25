import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(10, min(2 * dim, budget // 20))
        if self.pop_size < 2:
            self.pop_size = 2
        self.restart_threshold = max(5, dim)
        self.inertia_start = 0.9
        self.inertia_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        # Handle tiny budget or dimension
        if budget < 2:
            x = rng.uniform(lb, ub, dim)
            val = func(x)
            report_best(val, x)
            return val, x

        # Initialize population
        if pop_size < 2:
            pop_size = 2
        pop = rng.uniform(lb, ub, (pop_size, dim))
        vel = rng.uniform(-0.5, 0.5, (pop_size, dim)) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        pbest = pop.copy()
        pbest_fitness = np.full(pop_size, np.inf)
        gbest = None
        gbest_fitness = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            pbest_fitness[i] = val
            pbest[i] = x
            if val < gbest_fitness:
                gbest_fitness = val
                gbest = x.copy()
                report_best(gbest_fitness, gbest)

        generation = 0
        max_generations = (budget - evals) // pop_size if pop_size > 0 else 0
        no_improve = 0
        inertia = self.inertia_start

        while evals < budget and generation < max_generations:
            # Update inertia
            inertia = self.inertia_start - (self.inertia_start - self.inertia_end) * (generation / max_generations) if max_generations > 0 else self.inertia_end
            improved_this_gen = False

            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                vel[i] = inertia * vel[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (gbest - pop[i])
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                val = func(pop[i])
                evals += 1
                if val < pbest_fitness[i]:
                    pbest_fitness[i] = val
                    pbest[i] = pop[i].copy()
                if val < gbest_fitness:
                    gbest_fitness = val
                    gbest = pop[i].copy()
                    report_best(gbest_fitness, gbest)
                    improved_this_gen = True

            # Local search around global best
            if evals < budget and gbest is not None:
                local_evals = min(2, budget - evals)
                for _ in range(local_evals):
                    sigma = 0.01 * (ub - lb)
                    x = gbest + sigma * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < gbest_fitness:
                        gbest_fitness = val
                        gbest = x.copy()
                        report_best(gbest_fitness, gbest)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                # Restart: reinitialize population except global best
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                new_pop[0] = gbest.copy()
                new_vel = rng.uniform(-0.5, 0.5, (pop_size, dim)) * (ub - lb)
                new_vel[0] = 0
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i].copy()
                    val = func(x)
                    evals += 1
                    pbest_fitness[i] = val
                    pbest[i] = x
                    if val < gbest_fitness:
                        gbest_fitness = val
                        gbest = x.copy()
                        report_best(gbest_fitness, gbest)
                pop = new_pop
                vel = new_vel
                pbest[0] = gbest.copy()
                pbest_fitness[0] = gbest_fitness
                no_improve = 0
                # Reset inertia to start
                inertia = self.inertia_start

            generation += 1

        return gbest_fitness, gbest