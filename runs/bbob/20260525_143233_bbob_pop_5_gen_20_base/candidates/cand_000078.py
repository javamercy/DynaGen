import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(10, min(5 * dim, budget // 4))
        self.restart_threshold = max(10, 2 * dim)
        self.inertia_start = 0.9
        self.inertia_end = 0.4
        self.max_velocity_factor = 0.2

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        if pop_size < 2:
            best_x = rng.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = rng.uniform(lb, ub, (pop_size, dim))
        max_vel = self.max_velocity_factor * (ub - lb)
        velocity = rng.uniform(-max_vel, max_vel, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        personal_best = pop.copy()
        personal_best_fitness = np.full(pop_size, np.inf)
        global_best_val = np.inf
        global_best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            personal_best_fitness[i] = val
            personal_best[i] = x.copy()
            if val < global_best_val:
                global_best_val = val
                global_best_x = x.copy()
                report_best(global_best_val, global_best_x)

        no_improve = 0
        generation = 0

        while evals < budget:
            inertia = self.inertia_start - (self.inertia_start - self.inertia_end) * (evals / budget)
            improved_this_gen = False

            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                cognitive = 1.5 * r1 * (personal_best[i] - pop[i])
                social = 1.5 * r2 * (global_best_x - pop[i])
                velocity[i] = inertia * velocity[i] + cognitive + social
                velocity[i] = np.clip(velocity[i], -max_vel, max_vel)
                pop[i] = pop[i] + velocity[i]
                pop[i] = np.clip(pop[i], lb, ub)
                val = func(pop[i])
                evals += 1
                if val < fitness[i]:
                    fitness[i] = val
                    improved_this_gen = True
                    if val < personal_best_fitness[i]:
                        personal_best_fitness[i] = val
                        personal_best[i] = pop[i].copy()
                    if val < global_best_val:
                        global_best_val = val
                        global_best_x = pop[i].copy()
                        report_best(global_best_val, global_best_x)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                new_pop[0] = global_best_x.copy()
                new_velocity = rng.uniform(-max_vel, max_vel, (pop_size, dim))
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = global_best_val
                new_personal_best = new_pop.copy()
                new_personal_best_fitness = np.full(pop_size, np.inf)
                new_personal_best_fitness[0] = global_best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i].copy()
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    new_personal_best_fitness[i] = val
                    new_personal_best[i] = x.copy()
                    if val < global_best_val:
                        global_best_val = val
                        global_best_x = x.copy()
                        report_best(global_best_val, global_best_x)
                pop = new_pop
                velocity = new_velocity
                fitness = new_fitness
                personal_best = new_personal_best
                personal_best_fitness = new_personal_best_fitness
                no_improve = 0

            generation += 1

        return global_best_val, global_best_x