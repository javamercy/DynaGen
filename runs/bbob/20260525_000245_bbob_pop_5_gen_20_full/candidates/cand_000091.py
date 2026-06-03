import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        n = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget

        # initial feasible point
        best_x = np.random.uniform(lb, ub, n)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        if budget <= 1:
            return best_val, best_x

        # CMA-ES parameters
        lambda_ = min(budget - calls, 4 + int(3 * np.log(n)))
        lambda_ = max(2, lambda_)
        mu = lambda_ // 2
        if mu < 1:
            mu = 1
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / np.sum(w ** 2)

        c_s = (mu_eff + 2) / (n + mu_eff + 5)
        d_s = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        if mu == 1:
            c_mu = 0.0

        max_restarts = 3
        restart_count = 0
        no_improve_thresh = max(1, int(lambda_ * 5))
        mean_range = np.mean(ub - lb)

        # Enhanced DE parameters
        de_np = max(4, 2 * n)  # population size 2*dim
        de_F = 0.9
        de_CR_base = 0.5
        de_CR_amp = 0.4
        de_CR_cycle = 10.0
        de_generations = 12  # more generations

        while calls < budget and restart_count < max_restarts:
            if restart_count == 0:
                mean = best_x.copy()
                sigma = 0.2 * mean_range
            else:
                mean = np.random.uniform(lb, ub, n)
                sigma = 0.5 * mean_range  # high-sigma restart
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            no_improve_count = 0
            best_val_restart = best_val
            gen = 0

            while calls < budget:
                if calls + lambda_ > budget:
                    lambda_actual = budget - calls
                else:
                    lambda_actual = lambda_
                if lambda_actual < 1:
                    break

                # Sample
                try:
                    samples = np.random.multivariate_normal(mean, sigma ** 2 * C, size=lambda_actual)
                except np.linalg.LinAlgError:
                    samples = mean + sigma * np.random.randn(lambda_actual, n) * np.sqrt(np.diag(C))
                samples = np.clip(samples, lb, ub)

                # Evaluate
                vals = np.array([func(s) for s in samples])
                calls += lambda_actual

                idx = np.argsort(vals)
                vals_sorted = vals[idx]
                samples_sorted = samples[idx]

                if vals_sorted[0] < best_val:
                    best_val = vals_sorted[0]
                    best_x = samples_sorted[0]
                    report_best(best_val, best_x)
                    no_improve_count = 0
                    best_val_restart = best_val
                else:
                    no_improve_count += 1

                old_mean = mean.copy()
                mean = np.dot(w, samples_sorted[:mu])

                # Inverse sqrt C
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    invsqrtC = np.dot(eigvecs, np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T))
                except np.linalg.LinAlgError:
                    invsqrtC = np.eye(n)

                # Update evolution paths
                ps = (1 - c_s) * ps + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(invsqrtC, (mean - old_mean) / sigma)
                norm_ps = np.linalg.norm(ps)
                expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                sigma = sigma * np.exp((c_s / d_s) * (norm_ps / expected_norm - 1))
                sigma = max(sigma, 1e-12 * mean_range)

                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

                diffs = (samples_sorted[:mu] - old_mean) / sigma
                C_mu = np.zeros((n, n))
                for i in range(mu):
                    C_mu += w[i] * np.outer(diffs[i], diffs[i])
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * C_mu
                C = (C + C.T) / 2
                # Ensure positive definite
                eigvals, _ = np.linalg.eigh(C)
                if np.any(eigvals <= 0):
                    C = np.eye(n)

                gen += 1

                # Check stagnation and run enhanced DE local search before restart
                if no_improve_count >= no_improve_thresh and calls < budget:
                    # DE local search around best
                    de_improved = False
                    de_calls = 0
                    de_pop = np.random.uniform(lb, ub, size=(de_np, n))
                    de_fitness = np.full(de_np, np.inf)
                    for de_gen in range(de_generations):
                        if calls >= budget:
                            break
                        F = de_F
                        CR = de_CR_base + de_CR_amp * np.sin(2 * np.pi * (de_gen % de_CR_cycle) / de_CR_cycle)
                        CR = max(0, min(1, CR))
                        for i in range(de_np):
                            if calls >= budget:
                                break
                            r1, r2 = np.random.choice([j for j in range(de_np) if j != i], size=2, replace=False)
                            mutant = de_pop[i] + F * (best_x - de_pop[i]) + F * (de_pop[r1] - de_pop[r2])
                            trial = np.clip(mutant, lb, ub)
                            val = func(trial)
                            calls += 1
                            if val < de_fitness[i]:
                                de_pop[i] = trial
                                de_fitness[i] = val
                                if val < best_val:
                                    best_val = val
                                    best_x = trial.copy()
                                    report_best(best_val, best_x)
                                    de_improved = True
                    if de_improved:
                        # Reset stagnation counter, continue CMA-ES with new best mean
                        no_improve_count = 0
                        best_val_restart = best_val
                        mean = best_x.copy()
                        sigma = 0.2 * mean_range
                        C = np.eye(n)
                        pc = np.zeros(n)
                        ps = np.zeros(n)
                    else:
                        break  # break out of CMA-ES loop to perform full restart

            restart_count += 1

        return best_val, best_x