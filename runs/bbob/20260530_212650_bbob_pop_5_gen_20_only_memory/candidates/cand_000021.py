import numpy as np
import math

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial random point
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)

        # CMA-ES parameters (more exploitation: smaller population)
        sigma = 0.5 * np.mean(ub - lb)
        C = np.eye(dim)
        lam = 4 + int(2 * math.log(dim))  # reduced from 3*log(dim)
        lam = min(lam, self.budget - evals)
        if lam < 2:
            lam = max(2, self.budget - evals)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        last_improvement_evals = evals
        stagnation_limit = max(10, int(0.2 * self.budget))
        restart_count = 0
        max_restarts = 2

        # CMA-ES loop
        while evals < self.budget:
            if (evals - last_improvement_evals > stagnation_limit and
                self.budget - evals > 5 and restart_count < max_restarts):
                # Restart
                mean = np.random.uniform(lb, ub)
                sigma = 0.5 * np.mean(ub - lb)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                if evals < self.budget:
                    val = func(mean)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = mean.copy()
                        report_best(best_val, best_x)
                    last_improvement_evals = evals
                restart_count += 1
                lam = 4 + int(2 * math.log(dim))
                lam = min(lam, self.budget - evals)
                if lam < 2:
                    lam = max(2, self.budget - evals)
                mu = lam // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
                cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2 / ((dim + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff))
                damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs

            # Sample population
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            candidates = []
            for i in range(lam):
                z = np.random.randn(dim)
                x = mean + sigma * A @ z
                x = np.clip(x, lb, ub)
                candidates.append(x)
            vals = []
            for x in candidates:
                if evals >= self.budget:
                    break
                val = func(x)
                vals.append(val)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                    last_improvement_evals = evals
            if len(vals) == 0:
                break
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            old_mean = mean.copy()
            mean = np.sum([w * candidates[i] for i, w in enumerate(weights[:len(weights)])], axis=0)
            mean = np.clip(mean, lb, ub)
            z_mean = (mean - old_mean) / sigma
            try:
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
            except:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs)**(2*evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            sigma *= math.exp((cs/damps) * (np.linalg.norm(ps)/math.sqrt(dim) - 1))
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)

        # Pattern search (Hooke-Jeeves style) from best point
        if evals < self.budget:
            step = 0.1 * (ub - lb)
            step = np.maximum(step, 1e-5 * np.ones(dim))
            while evals < self.budget:
                improved = False
                # Exploratory moves along coordinates in random order (seed-controlled via numpy)
                order = np.random.permutation(dim)
                for i in order:
                    # Positive direction
                    trial = best_x.copy()
                    trial[i] += step[i]
                    trial = np.clip(trial, lb, ub)
                    if evals >= self.budget:
                        break
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved = True
                        break  # accept first improvement
                    # Negative direction
                    trial = best_x.copy()
                    trial[i] -= step[i]
                    trial = np.clip(trial, lb, ub)
                    if evals >= self.budget:
                        break
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved = True
                        break
                if improved:
                    # Step size increase on success (acceleration)
                    step *= 2.0
                    step = np.minimum(step, 0.5 * (ub - lb))
                else:
                    # No improvement: contract step size
                    step *= 0.5
                # Termination if step size too small
                if np.max(step) < 1e-12 or np.min(step) < 1e-14:
                    break
        return best_val, best_x