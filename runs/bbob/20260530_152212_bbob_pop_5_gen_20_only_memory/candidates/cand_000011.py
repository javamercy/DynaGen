import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.stagnation_limit = max(1, int(budget / 10))
        self.lambda_ = 3  # number of offspring per generation

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        rng = self.rng
        # Initial scale: 20% of domain range
        scale = 0.2 * (ub - lb)
        # Initial random point
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        calls = 1
        last_improvement = calls
        report_best(best_val, best_x)

        while calls < self.budget:
            # Check stagnation
            if calls - last_improvement >= self.stagnation_limit:
                # Restart: new random point
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_val = func(new_x)
                calls += 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x
                    last_improvement = calls
                    report_best(best_val, best_x)
                scale = 0.2 * (ub - lb)  # reset scale
                continue

            # Generate lambda candidates
            perturbations = rng.randn(self.lambda_, dim) * scale
            candidates = best_x + perturbations
            candidates = np.clip(candidates, lb, ub)
            # Evaluate all candidates
            for i in range(self.lambda_):
                if calls >= self.budget:
                    break
                val = func(candidates[i])
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = candidates[i].copy()
                    last_improvement = calls
                    report_best(best_val, best_x)
            # Update scale based on whether improvement occurred
            # Check if best_val changed since last iteration (compare with stored previous best)
            # We'll use a flag: we stored best before candidates evaluation? Simpler: track if improved
            # Since we might have improved multiple times, we can check if last_improvement == calls (last call was an improvement) but not perfect. Use a separate variable.
            improved = False
            # Actually we can track before loop: old_best = best_val, then after loop if best_val < old_best: improved = True
            # But we need to know if any improvement happened. We'll compute after loop.
            # However we already updated best_val and best_x inside. Better to detect improvement by comparing before and after.
            # We'll store pre-iteration best_val.
            prev_best = best_val
            # But because we updated best_val inside, we lost prev. We'll compute improvement flag after loop by checking if best_val changed? But it may have changed from multiple updates. Simpler: use a boolean flag initialized False, and set True when improvement occurs.
            # Let's restructure: evaluate all candidates, track best among them and overall. Then update.
            # This is simpler: we'll evaluate candidates and keep the best among them, and compare with current best.
            # But we already update best inside loop; better to collect values.
            # Alternative: do the loop and after loop, check if best_val < prev_best (store prev_best before loop). We'll assign prev_best before generating candidates.
            prev_best = best_val
            for i in range(self.lambda_):
                if calls >= self.budget:
                    break
                candidate = candidates[i]
                val = func(candidate)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    last_improvement = calls
                    report_best(best_val, best_x)
            if best_val < prev_best:
                scale *= 0.9  # exploit
            else:
                scale *= 1.1  # explore
            # Ensure scale not too small
            scale = np.maximum(scale, 1e-10 * (ub - lb))
        return best_val, best_x