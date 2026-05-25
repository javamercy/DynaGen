import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_x = lb + (ub - lb) * self.rng.rand(self.dim)
        best_f = func(best_x)
        report_best(best_f, best_x)
        evals = 1
        step = 0.05 * np.mean(ub - lb)
        while evals < self.budget:
            improved = False
            for i in range(self.dim):
                if evals >= self.budget:
                    break
                d = np.zeros(self.dim)
                d[i] = step
                xp = np.clip(best_x + d, lb, ub)
                xn = np.clip(best_x - d, lb, ub)
                # Evaluate positive direction if distinct
                f_plus = None
                if np.any(xp != best_x):
                    f_plus = func(xp)
                    evals += 1
                    if f_plus < best_f:
                        best_f = f_plus
                        best_x = xp.copy()
                        improved = True
                        report_best(best_f, best_x)
                        if evals >= self.budget:
                            break
                # Evaluate negative direction if distinct and not already evaluated
                f_minus = None
                if np.any(xn != best_x) and evals < self.budget:
                    # Avoid re-evaluating if xp == xn (shouldn't happen)
                    if np.any(xn != xp) or f_plus is None:
                        f_minus = func(xn)
                        evals += 1
                        if f_minus < best_f:
                            best_f = f_minus
                            best_x = xn.copy()
                            improved = True
                            report_best(best_f, best_x)
                            if evals >= self.budget:
                                break
                # Attempt quadratic fit if both sides evaluated and distinct from best_x
                if f_plus is not None and f_minus is not None and evals < self.budget:
                    x0 = best_x.copy()
                    # Re-evaluate best_f if it changed? Actually we might have updated best_x.
                    # So we need to use the original best_x before this coordinate's updates.
                    # This is tricky. For simplicity, we skip quadratic if best changed.
                    # Instead, we'll compute quadratic using the original best_x and f0.
                    # But we already may have updated best_x. Let's store original.
                    # This makes code messy. We'll skip quadratic for this version to keep it simple and correct.
                    pass
            if not improved:
                step *= 0.9
                if step < 1e-15:
                    break
            else:
                # Optionally increase step? For exploitation, we keep same or reduce. We'll reduce slightly.
                step *= 0.95
        return best_f, best_x