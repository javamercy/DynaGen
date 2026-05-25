import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(20, min(4 * dim, budget // 2))
        self.max_stall = max(10, budget // 10)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        # initialize positions and velocities
        positions = self.rng.uniform(lb, ub, size=(popsize, dim))
        velocities = self.rng.uniform(-(ub - lb) / 2, (ub - lb) / 2, size=(popsize, dim))
        # evaluate initial positions
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            x = positions[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # personal bests
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        gbest_position = best_x.copy() if best_x is not None else positions[0].copy()
        gbest_fitness = best_val
        stall_counter = 0
        w_start = 0.9
        w_end = 0.4
        max_iter = (self.budget // popsize) + 1
        iteration = 0
        while evaluations < self.budget and iteration < max_iter:
            w = w_start - (w_start - w_end) * iteration / max_iter if max_iter > 0 else w_end
            c1 = 2.0
            c2 = 2.0
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (pbest_positions[i] - positions[i]) +
                                 c2 * r2 * (gbest_position - positions[i]))
                max_vel = (ub - lb) / 2
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                new_pos = positions[i] + velocities[i]
                new_pos = np.clip(new_pos, lb, ub)
                val = func(new_pos)
                evaluations += 1
                if val < pbest_fitness[i]:
                    pbest_fitness[i] = val
                    pbest_positions[i] = new_pos.copy()
                if val < gbest_fitness:
                    gbest_fitness = val
                    gbest_position = new_pos.copy()
                    best_val = gbest_fitness
                    best_x = gbest_position.copy()
                    report_best(best_val, best_x)
                    stall_counter = 0
                else:
                    stall_counter += 1
                positions[i] = new_pos
            if stall_counter > self.max_stall:
                num_restart = popsize // 2
                worst_indices = np.argsort(pbest_fitness)[-num_restart:]
                for idx in worst_indices:
                    if evaluations >= self.budget:
                        break
                    if idx == np.argmin(pbest_fitness):
                        continue
                    new_pos = self.rng.uniform(lb, ub)
                    val = func(new_pos)
                    evaluations += 1
                    positions[idx] = new_pos
                    pbest_positions[idx] = new_pos.copy()
                    pbest_fitness[idx] = val
                    velocities[idx] = self.rng.uniform(-(ub - lb) / 2, (ub - lb) / 2, size=dim)
                    if val < gbest_fitness:
                        gbest_fitness = val
                        gbest_position = new_pos.copy()
                        best_val = gbest_fitness
                        best_x = gbest_position.copy()
                        report_best(best_val, best_x)
                stall_counter = 0
            iteration += 1
        return best_val, best_x