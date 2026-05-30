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
        dim = self.dim
        budget = self.budget
        rng = self.rng
        diff = ub - lb

        # Initial distribution parameters
        mean = (lb + ub) / 2.0
        std = 0.3 * diff  # initial std
        # To avoid singularity, use diagonal covariance

        # Elite fraction
        elite_frac = 0.2
        sample_size = min(20 * dim, budget // 2)
        sample_size = max(sample_size, 10)

        # Initial sampling to evaluate at least one point
        best_x = mean.copy()
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        # Main loop
        while calls < budget:
            # Adjust sample size to remaining budget
            remaining = budget - calls
            if sample_size > remaining:
                sample_size = remaining
            if sample_size < 1:
                break

            # Sample from Gaussian
            samples = rng.randn(sample_size, dim) * std + mean
            # Clip to bounds
            samples = np.clip(samples, lb, ub)

            # Evaluate all samples
            vals = np.full(sample_size, np.inf)
            for i in range(sample_size):
                if calls >= budget:
                    break
                x = samples[i]
                val = func(x)
                calls += 1
                vals[i] = val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

            if calls >= budget:
                break

            # Sort indices by value (elite selection)
            idx_sorted = np.argsort(vals)
            n_elite = max(1, int(elite_frac * sample_size))
            elite_idx = idx_sorted[:n_elite]
            elite_samples = samples[elite_idx]

            # Update mean and std
            new_mean = np.mean(elite_samples, axis=0)
            new_std = np.std(elite_samples, axis=0) + 1e-10  # avoid zero
            # Smooth update to avoid instability
            alpha = 0.8
            mean = (1 - alpha) * mean + alpha * new_mean
            std = (1 - alpha) * std + alpha * new_std

            # Ensure std not too small
            min_std = 1e-6 * diff
            std = np.maximum(std, min_std)

        return best_val, best_x