import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = min(5 * dim, max(10, budget // 4))
        # Initialize positions and velocities
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        velocities = np.random.uniform(-np.abs(ub - lb) * 0.1, np.abs(ub - lb) * 0.1, (pop_size, dim))
        pbest_positions = positions.copy()
        pbest_values = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        # Evaluate initial swarm
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(positions[i], lb, ub)
            val = func(x)
            fcalls += 1
            pbest_values[i] = val
            pbest_positions[i] = x.copy()
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        # PSO parameters
        w = 0.7
        c1 = 1.5
        c2 = 1.5
        # Stagnation tracking
        max_stagnation = max(10, budget // (pop_size * 2))
        no_improve_counter = 0
        # Main loop
        while fcalls < budget:
            improved_this_gen = False
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                r1 = random.random()
                r2 = random.random()
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - positions[i]) + c2 * r2 * (best_x - positions[i])
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                # Random perturbation for diversification
                if random.random() < 0.05:
                    step = np.random.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb))
                    positions[i] = np.clip(positions[i] + step, lb, ub)
                # Evaluate
                val = func(positions[i])
                fcalls += 1
                # Update personal best
                if val < pbest_values[i]:
                    pbest_values[i] = val
                    pbest_positions[i] = positions[i].copy()
                    if val < best_f:
                        best_f = val
                        best_x = positions[i].copy()
                        improved_this_gen = True
                        report_best(best_f, best_x)
            # Update stagnation counter
            if improved_this_gen:
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            # Restart if stagnation
            if no_improve_counter >= max_stagnation:
                # Restart worst half of particles
                indices = np.argsort(pbest_values)
                num_restart = max(1, pop_size // 2)
                restart_indices = indices[-num_restart:] if num_restart < pop_size else indices
                for idx in restart_indices:
                    if fcalls >= budget:
                        break
                    positions[idx] = np.random.uniform(lb, ub)
                    velocities[idx] = np.random.uniform(-np.abs(ub - lb) * 0.1, np.abs(ub - lb) * 0.1)
                    val = func(positions[idx])
                    fcalls += 1
                    pbest_values[idx] = val
                    pbest_positions[idx] = positions[idx].copy()
                    if val < best_f:
                        best_f = val
                        best_x = positions[idx].copy()
                        report_best(best_f, best_x)
                no_improve_counter = 0
        return best_f, best_x