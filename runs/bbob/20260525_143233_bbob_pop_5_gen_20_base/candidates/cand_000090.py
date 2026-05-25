import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(10, min(4 * dim, budget // 4))
        self.restart_threshold = max(10, 2 * dim)
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.05
        self.c2 = 2.05
        self.phi = self.c1 + self.c2
        self.chi = 2.0 / (self.phi - 2.0 + np.sqrt(self.phi ** 2 - 4.0 * self.phi))  # constriction factor

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        # Fallback for tiny population
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

        # Initialize swarm
        positions = rng.uniform(lb, ub, (pop_size, dim)).astype(np.float64)
        velocities = rng.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), (pop_size, dim)).astype(np.float64)
        fitness = np.full(pop_size, np.inf)
        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(pop_size, np.inf)
        global_best_x = None
        global_best_val = np.inf

        evals = 0
        # Initial evaluation
        for i in range(pop_size):
            if evals >= budget:
                break
            x = positions[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            personal_best_fitness[i] = val
            personal_best_positions[i] = x.copy()
            if val < global_best_val:
                global_best_val = val
                global_best_x = x.copy()
                report_best(global_best_val, global_best_x)

        no_improve = 0
        generation = 0
        max_generations = (budget - evals) // pop_size if pop_size > 0 else 0

        while evals < budget and generation < max_generations:
            # Update inertia weight
            w = self.w_start - (self.w_start - self.w_end) * generation / max_generations
            improved_this_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.uniform(0, 1, dim)
                r2 = rng.uniform(0, 1, dim)
                velocities[i] = self.chi * (w * velocities[i] + self.c1 * r1 * (personal_best_positions[i] - positions[i]) + self.c2 * r2 * (global_best_x - positions[i]))
                positions[i] = positions[i] + velocities[i]
                # Clip to bounds
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evals += 1
                fitness[i] = val
                if val < personal_best_fitness[i]:
                    personal_best_fitness[i] = val
                    personal_best_positions[i] = positions[i].copy()
                    if val < global_best_val:
                        global_best_val = val
                        global_best_x = positions[i].copy()
                        report_best(global_best_val, global_best_x)
                        improved_this_gen = True

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                # Reinitialize swarm but keep global best
                new_positions = rng.uniform(lb, ub, (pop_size, dim)).astype(np.float64)
                new_positions[0] = global_best_x.copy()
                new_velocities = rng.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), (pop_size, dim)).astype(np.float64)
                new_fitness = np.full(pop_size, np.inf)
                new_personal_best_positions = new_positions.copy()
                new_personal_best_fitness = np.full(pop_size, np.inf)
                new_personal_best_fitness[0] = global_best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_positions[i]
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    new_personal_best_fitness[i] = val
                    if val < global_best_val:
                        global_best_val = val
                        global_best_x = x.copy()
                        report_best(global_best_val, global_best_x)
                positions = new_positions
                velocities = new_velocities
                fitness = new_fitness
                personal_best_positions = new_personal_best_positions
                personal_best_fitness = new_personal_best_fitness
                no_improve = 0
            generation += 1

        return global_best_val, global_best_x