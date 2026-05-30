import numpy as np
import math

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial point
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)
        # CMA-ES parameters (exploitation-focused)
        sigma_init = 0.3 * np.mean(ub - lb)
        sigma = sigma_init
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
        c1 = 2 / ((dim + 1.3) ** 2 + mueff) * 1.5
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff)) * 1.5
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        # Stagnation tracking
        last_improvement_evals = evals
        stagnation_limit = max(10, int(0.3 * self.budget))
        restart_count = 0
        max_restarts = 3
        # CMA-ES loop
        while evals < self.budget:
            # Check stagnation and restart if needed
            if (evals - last_improvement_evals > stagnation_limit and
                self.budget - evals > 5 and restart_count < max_restarts):
                # Restart with doubled population
                restart_count += 1
                lam = lam * 2
                lam = min(lam, self.budget - evals)
                if lam < 2:
                    lam = max(2, self.budget - evals)
                mu = lam // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)
                mueff = 1.0 / np.sum(weights ** 2)
                cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2 / ((dim + 1.3) ** 2 + mueff) * 1.5
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2) ** 2 + mueff)) * 1.5
                damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
                # Reset mean, covariance, paths
                mean = np.random.uniform(lb, ub)
                sigma = sigma_init
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
                continue
            # Sample population
            try:
                A = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                A = np.eye(dim)
            candidates = []
            for i in range(lam):
                if evals >= self.budget:
                    break
                z = np.random.randn(dim)
                x = mean + sigma * A @ z
                x = np.clip(x, lb, ub)
                candidates.append(x)
            # Evaluate
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
            # Sort
            idx = np.argsort(vals)
            candidates = [candidates[i] for i in idx]
            # Update mean
            old_mean = mean.copy()
            mean = np.sum([w * candidates[i] for i, w in enumerate(weights[:len(weights)])], axis=0)
            mean = np.clip(mean, lb, ub)
            # Update evolution paths
            z_mean = (mean - old_mean) / sigma
            try:
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
            except:
                invsqrtC = np.eye(dim)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
            hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2*evals/lam)) < (1.4 + 2/(dim+1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
            # Update covariance
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            for i in range(mu):
                z = (candidates[i] - old_mean) / sigma
                C += cmu * weights[i] * np.outer(z, z)
            C = (C + C.T) / 2
            # Update step size
            sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            # Adjust lambda for remaining budget
            remaining = self.budget - evals
            if remaining < lam:
                lam = max(2, remaining)
        # Nelder-Mead local search from best_x
        if evals < self.budget:
            n = dim
            simplex = [best_x.copy()]
            for i in range(n):
                step = 0.05 * (ub[i] - lb[i])
                if step == 0:
                    step = 0.05
                point = best_x.copy()
                point[i] += step
                point = np.clip(point, lb, ub)
                simplex.append(point)
            fvals = []
            for x in simplex:
                if evals >= self.budget:
                    break
                val = func(x)
                fvals.append(val)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            if len(fvals) >= n+1:
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                while evals < self.budget:
                    centroid = np.mean(simplex[:-1], axis=0)
                    reflect = centroid + 1.0 * (centroid - simplex[-1])
                    reflect = np.clip(reflect, lb, ub)
                    if evals >= self.budget:
                        break
                    fref = func(reflect)
                    evals += 1
                    if fref < best_val:
                        best_val = fref
                        best_x = reflect.copy()
                        report_best(best_val, best_x)
                    if fref < fvals[0]:
                        expand = centroid + 2.0 * (reflect - centroid)
                        expand = np.clip(expand, lb, ub)
                        if evals >= self.budget:
                            break
                        fexp = func(expand)
                        evals += 1
                        if fexp < best_val:
                            best_val = fexp
                            best_x = expand.copy()
                            report_best(best_val, best_x)
                        if fexp < fref:
                            simplex[-1] = expand
                            fvals[-1] = fexp
                        else:
                            simplex[-1] = reflect
                            fvals[-1] = fref
                    elif fref < fvals[-2]:
                        simplex[-1] = reflect
                        fvals[-1] = fref
                    else:
                        if fref < fvals[-1]:
                            contract = centroid + 0.5 * (reflect - centroid)
                        else:
                            contract = centroid + 0.5 * (simplex[-1] - centroid)
                        contract = np.clip(contract, lb, ub)
                        if evals >= self.budget:
                            break
                        fcont = func(contract)
                        evals += 1
                        if fcont < best_val:
                            best_val = fcont
                            best_x = contract.copy()
                            report_best(best_val, best_x)
                        if fcont < fvals[-1]:
                            simplex[-1] = contract
                            fvals[-1] = fcont
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                if evals >= self.budget:
                                    break
                                val = func(simplex[i])
                                evals += 1
                                fvals[i] = val
                                if val < best_val:
                                    best_val = val
                                    best_x = simplex[i].copy()
                                    report_best(best_val, best_x)
                            if evals >= self.budget:
                                break
                    idx = np.argsort(fvals)
                    simplex = [simplex[i] for i in idx]
                    fvals = [fvals[i] for i in idx]
        return best_val, best_x