import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        pop_size = max(4, min(20, self.budget // 20))
        
        # Initialize particles
        positions = lb + (ub - lb) * self.rng.rand(pop_size, self.dim)
        velocities = (ub - lb) * (self.rng.rand(pop_size, self.dim) - 0.5) * 0.2
        personal_best_pos = positions.copy()
        personal_best_val = np.full(pop_size, np.inf)
        
        # Evaluate initial positions
        for i in range(pop_size):
            if evals >= self.budget:
                break
            val = func(positions[i])
            evals += 1
            personal_best_val[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = positions[i].copy()
                report_best(self.best_val, self.best_x)
        
        # PSO parameters
        w = 0.7
        c1 = 2.0
        c2 = 2.0
        
        # Main loop
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                r1 = self.rng.rand(self.dim)
                r2 = self.rng.rand(self.dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_pos[i] - positions[i]) + c2 * r2 * (self.best_x - positions[i])
                # Clamp velocity
                max_vel = 0.5 * (ub - lb)
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                val = func(positions[i])
                evals += 1
                if val < personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = positions[i].copy()
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = positions[i].copy()
                        report_best(self.best_val, self.best_x)
        
        return self.best_val, self.best_x