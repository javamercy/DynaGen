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
        range_ = ub - lb
        best_val = None
        best_x = None
        evals = 0

        # initial point
        x0 = np.random.uniform(lb, ub)
        best_val = func(x0)
        best_x = x0.copy()
        evals += 1
        report_best(best_val, best_x)

        if self.budget <= 1:
            return best_val, best_x

        # CMA-ES parameters
        lam = max(2, 10 + int(5 * math.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        # reduced learning rates for exploration
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff) * 0.5
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)) * 0.5
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs

        # reserve budget for Nelder-Mead (at least dim+2 evaluations)
        reserve_nm = dim + 2

        while evals < self.budget - reserve_nm:
            # initialize for this restart
            mean = best_x.copy()
            sigma = 0.5 * np.mean(range_)
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            last_improvement_evals = evals
            local_evals = 0

            lam_cur = min(lam, self.budget - evals - reserve_nm)
            if lam_cur < 2:
                lam_cur = max(2, self.budget - evals - reserve_nm)
                if lam_cur < 2:
                    break

            while evals < self.budget - reserve_nm:
                # sample
                try:
                    A = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    A = np.eye(dim)
                candidates = []
                for _ in range(lam_cur):
                    z = np.random.randn(dim)
                    y = mean + sigma * A @ z
                    y = np.clip(y, lb, ub)
                    candidates.append(y)
                # evaluate
                vals = []
                for cand in candidates:
                    if evals >= self.budget - reserve_nm:
                        break
                    val = func(cand)
                    vals.append(val)
                    evals += 1
                    local_evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = cand.copy()
                        last_improvement_evals = evals
                        report_best(best_val, best_x)
                if len(vals) == 0:
                    break
                # sort
                idx = np.argsort(vals)
                candidates = [candidates[i] for i in idx]
                # update mean
                old_mean = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += weights[i] * candidates[i]
                mean = np.clip(mean, lb, ub)
                # update paths
                z_mean = (mean - old_mean) / sigma
                try:
                    invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
                except:
                    invsqrtC = np.eye(dim)
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
                hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2*local_evals/lam_cur)) < (1.4 + 2/(dim+1))
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
                # update covariance
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                for i in range(mu):
                    z = (candidates[i] - old_mean) / sigma
                    C += cmu * weights[i] * np.outer(z, z)
                C = (C + C.T) / 2
                # update sigma
                sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
                if sigma < 1e-10 * np.mean(range_):
                    sigma = 0.1 * np.mean(range_)
                    C = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                # adjust lambda for remaining budget
                lam_cur = min(lam, self.budget - evals - reserve_nm)
                if lam_cur < 2:
                    break
                # restart condition: no improvement for a while
                if evals - last_improvement_evals > 0.2 * self.budget:
                    break
                # also break if we have too few evaluations left for NM
                if self.budget - evals <= reserve_nm:
                    break

            # restart by reinitializing mean to a new random point (but not evaluated yet)
            # The loop will then continue with the new mean in the next iteration
            mean = np.random.uniform(lb, ub)
            # if not enough budget for another restart, break outer while
            if self.budget - evals <= reserve_nm:
                break

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
            if len(fvals) == n+1:
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