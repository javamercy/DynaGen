import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.step_size = np.ones(dim) * 0.1  # initial step size per dimension
        self.success_window = max(10, dim * 5)
        self.success_counter = 0
        self.eval_counter = 0
        self.stall_counter = 0
        self.restart_threshold = max(1, budget // 4)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        # Initialize at random point
        best_x = self.rng.uniform(lb, ub, dim)
        best_val = func(best_x)
        self.eval_counter = 1
        report_best(best_val, best_x)
        
        # Keep track of recent successes for step size adaptation
        recent_successes = []
        
        while self.eval_counter < self.budget:
            # Generate offspring by perturbing all coordinates with Gaussian noise
            noise = self.rng.normal(0, self.step_size, dim)
            candidate = np.clip(best_x + noise, lb, ub)
            val = func(candidate)
            self.eval_counter += 1
            
            if val < best_val:
                # Improvement: accept
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                recent_successes.append(1)
                self.stall_counter = 0
            else:
                recent_successes.append(0)
                self.stall_counter += 1
            
            # Keep window of recent successes
            if len(recent_successes) > self.success_window:
                recent_successes.pop(0)
            
            # Adapt step size using 1/5 rule
            if len(recent_successes) == self.success_window:
                success_rate = np.mean(recent_successes)
                if success_rate > 0.2:
                    self.step_size *= 1.2
                elif success_rate < 0.2:
                    self.step_size *= 0.8
                # Reset window
                recent_successes = []
            
            # Restart if stalled
            if self.stall_counter >= self.restart_threshold:
                # New random point
                best_x = self.rng.uniform(lb, ub, dim)
                best_val = func(best_x)
                self.eval_counter += 1
                report_best(best_val, best_x)
                self.stall_counter = 0
                recent_successes = []
                self.step_size = np.ones(dim) * 0.1
            
            # If budget exceeded, break
            if self.eval_counter >= self.budget:
                break
        
        return best_val, best_x