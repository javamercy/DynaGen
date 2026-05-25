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
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        # initial point
        best_x = np.random.uniform(lb, ub, dim)
        best_f = func(best_x)
        fcalls = 1
        report_best(best_f, best_x)
        # (1+1)-ES
        parent = best_x.copy()
        parent_f = best_f
        sigma = 0.2 * np.mean(ub - lb)
        sigma_init = sigma
        window = max(10, dim)
        successes = 0
        gen = 0
        no_improve_gen = 0
        max_no_improve = max(50, 5 * dim)
        while fcalls < budget:
            child = parent + sigma * np.random.randn(dim)
            child = np.clip(child, lb, ub)
            child_f = func(child)
            fcalls += 1
            if child_f < parent_f:
                parent = child
                parent_f = child_f
                if child_f < best_f:
                    best_f = child_f
                    best_x = child.copy()
                    report_best(best_f, best_x)
                successes += 1
                no_improve_gen = 0
            else:
                no_improve_gen += 1
            gen += 1
            if gen % window == 0:
                success_rate = successes / window
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma /= 1.2
                successes = 0
                sigma = max(sigma, 1e-10 * (ub - lb).mean())
            if no_improve_gen >= max_no_improve or sigma < 1e-8 * (ub - lb).mean():
                if fcalls >= budget:
                    break
                new_x = np.random.uniform(lb, ub, dim)
                new_f = func(new_x)
                fcalls += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    report_best(best_f, best_x)
                parent = new_x
                parent_f = new_f
                sigma = sigma_init
                successes = 0
                no_improve_gen = 0
        return best_f, best_x