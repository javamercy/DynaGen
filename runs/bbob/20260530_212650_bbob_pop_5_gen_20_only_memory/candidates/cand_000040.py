import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point
        best_x = lb + rng.rand(dim) * (ub - lb)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # CMA-ES parameters
        mean = best_x.copy()
        sigma = 0.5 * np.mean(ub - lb)
        C = np.eye(dim)
        lam = 4 + int(3 * math.log(dim))
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

        cma_budget = int(0.7 * self.budget)
        # Phase 1: CMA-ES
        while evals < cma_budget:
            remaining = cma_budget - evals
            if remaining < lam:
                lam = max(2, remaining)
                if lam < 2:
                    break
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            candidates = []
            for i in range(lam):
                z = rng.randn(dim)
                x = mean + sigma * A @ z
                x = np.clip(x, lb, ub)
                candidates.append(x)
            vals = []
            for x in candidates:
                if evals >= cma_budget:
                    break
                val = func(x)
                vals.append(val)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
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
            except np.linalg.LinAlgError:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2*evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

        # Phase 2: Intensified pattern search
        step_sizes = (ub - lb) * 0.1
        iteration = 0
        while evals < self.budget:
            if iteration % 2 == 0:
                # coordinate search round-robin
                d = (iteration // 2) % dim
                step = step_sizes[d]
                # positive direction
                x_cand = best_x.copy()
                x_cand[d] = np.clip(best_x[d] + step, lb[d], ub[d])
                val = func(x_cand)
                evals += 1
                if val < best_val:
                    best_x = x_cand.copy()
                    best_val = val
                    report_best(best_val, best_x)
                    step_sizes[d] *= 1.5
                    # line search extension
                    while evals < self.budget:
                        new_x = best_x.copy()
                        new_x[d] = np.clip(best_x[d] + step_sizes[d], lb[d], ub[d])
                        new_val = func(new_x)
                        evals += 1
                        if new_val < best_val:
                            best_x = new_x.copy()
                            best_val = new_val
                            report_best(best_val, best_x)
                            step_sizes[d] *= 1.5
                        else:
                            break
                else:
                    # negative direction
                    x_cand = best_x.copy()
                    x_cand[d] = np.clip(best_x[d] - step, lb[d], ub[d])
                    val = func(x_cand)
                    evals += 1
                    if val < best_val:
                        best_x = x_cand.copy()
                        best_val = val
                        report_best(best_val, best_x)
                        step_sizes[d] *= 1.5
                        # line search extension
                        while evals < self.budget:
                            new_x = best_x.copy()
                            new_x[d] = np.clip(best_x[d] - step_sizes[d], lb[d], ub[d])
                            new_val = func(new_x)
                            evals += 1
                            if new_val < best_val:
                                best_x = new_x.copy()
                                best_val = new_val
                                report_best(best_val, best_x)
                                step_sizes[d] *= 1.5
                            else:
                                break
                    else:
                        step_sizes[d] *= 0.7
                step_sizes[d] = max(step_sizes[d], 1e-10)
            else:
                # pattern search random direction
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = np.ones(dim) / np.sqrt(dim)
                else:
                    direction /= norm
                avg_step = np.mean(step_sizes)
                candidate = np.clip(best_x + avg_step * direction, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_x = candidate.copy()
                    best_val = val
                    report_best(best_val, best_x)
                    step_sizes *= 1.5
                    # line search extension along direction
                    while evals < self.budget:
                        new_x = np.clip(best_x + avg_step * direction, lb, ub)
                        new_val = func(new_x)
                        evals += 1
                        if new_val < best_val:
                            best_x = new_x.copy()
                            best_val = new_val
                            report_best(best_val, best_x)
                            step_sizes *= 1.5
                        else:
                            break
                else:
                    step_sizes *= 0.7
                step_sizes = np.maximum(step_sizes, 1e-10)
            iteration += 1
            if evals >= self.budget:
                break

        return best_val, best_x