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

        # Initial mean
        mean = rng.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)

        # Step size (10% of range)
        sigma = 0.1 * np.mean(ub - lb)
        # (mu, lambda) parameters
        mu = 4
        lam = 8
        # weights for recombination (linear decrease)
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights = weights / np.sum(weights)

        # Success counter for 1/5 rule
        success_counter = 0
        generation = 0
        # Maximum generations based on budget
        max_generations = budget // lam

        for gen in range(max_generations):
            # Sample offspring
            offspring = rng.normal(loc=mean, scale=sigma, size=(lam, dim))
            # Clip to bounds
            offspring = np.clip(offspring, lb, ub)
            # Evaluate
            vals = np.full(lam, np.inf)
            for i in range(lam):
                if evals >= budget:
                    break
                vals[i] = func(offspring[i])
                evals += 1
                if vals[i] < best_val:
                    best_val = vals[i]
                    best_x = offspring[i].copy()
                    report_best(best_val, best_x)
            if evals >= budget:
                break

            # Sort by fitness
            idx = np.argsort(vals)
            # Select best mu
            selected = offspring[idx[:mu]]
            # Recombine (weighted mean)
            mean_new = np.dot(weights, selected)
            # Clip new mean
            mean_new = np.clip(mean_new, lb, ub)
            mean = mean_new

            # Success rate for 1/5 rule: check how many of the best mu improved over previous mean?
            # Simplified: compare each selected's value to previous mean value? We'll use best in generation.
            # Use fraction of offspring that are better than best mean so far? Actually simpler: track if best offspring improved
            # Use success if best offspring value < previous best value? But that might be too strict. Use: if best offspring value < mean value? We'll use: if vals[idx[0]] < best_val or using a relative threshold.
            # Standard 1/5 rule: if success rate > 1/5 increase sigma, else decrease. Here success = number of offspring better than mean? We'll compute success counter based on offspring better than previous mean.
            # Compute mean value once before evaluation? Not needed. Use: count how many offspring are better than the population mean? But we don't have mean fitness. Instead, use: if the best offspring improved global best, count as success.
            # Simpler: if vals[idx[0]] < best_val (but best_val already updated), we can track previous best before gen.
            # Store previous best before gen:
            prev_best = best_val
            # After evaluation, we updated best_val; so if best_val < prev_best, success.
            success = (best_val < prev_best)
            if success:
                success_counter += 1
            else:
                success_counter = 0

            # Adjust sigma every 'lam' evaluations? Actually 1/5 rule adapts each generation.
            # Use cumulative success rate over a window? Simplified: if success_counter > 0.2*lam? Not ideal. Use: if success in this gen, increase; else decrease.
            # But to mimic 1/5 rule, we need a window. We'll use a simple rule: if success (improvement) then sigma *= 1.2 else sigma /= 1.2
            if success:
                sigma *= 1.2
            else:
                sigma /= 1.2

            # Ensure sigma doesn't become too small
            sigma = max(sigma, 1e-10 * np.mean(ub - lb))

            # Restart if sigma too small relative to range? Not necessary.

        # If budget remains, do random sampling
        while evals < budget:
            x_rand = rng.uniform(lb, ub)
            val_rand = func(x_rand)
            evals += 1
            if val_rand < best_val:
                best_val = val_rand
                best_x = x_rand.copy()
                report_best(best_val, best_x)

        return best_val, best_x