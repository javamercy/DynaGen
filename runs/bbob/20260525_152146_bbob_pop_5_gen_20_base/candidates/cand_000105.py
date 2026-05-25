import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        if budget <= 0:
            best_x = np.zeros(dim)
            best_f = float('inf')
            report_best(best_f, best_x)
            return best_f, best_x

        # population size
        lambda_ = max(2, min(budget, 4 + int(3 * np.log(dim))))
        mu = lambda_ // 2

        # weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # strategy parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # initialize dynamic state
        mean = rng.uniform(lb, ub, size=dim)
        sigma = 0.5 * (ub - lb) / np.sqrt(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        eigenval = None
        eigenvec = None

        best_x = mean.copy()
        best_f = func(mean)
        budget -= 1
        report_best(best_f, best_x)

        gen = 0
        stall_counter = 0
        stall_limit = max(5, budget // (2 * lambda_))
        while budget > 0:
            # sample offspring
            if eigenvec is None or eigenval is None:
                eigenval, eigenvec = np.linalg.eigh(C)
            arz = rng.randn(lambda_, dim)
            arx = mean + sigma * (eigenvec * np.sqrt(eigenval)).dot(arz.T).T
            arx = np.clip(arx, lb, ub)
            arf = np.full(lambda_, np.inf)
            for i in range(lambda_):
                if budget <= 0:
                    break
                arf[i] = func(arx[i])
                budget -= 1
                if arf[i] < best_f:
                    best_f = arf[i]
                    best_x = arx[i].copy()
                    report_best(best_f, best_x)
                    stall_counter = 0
            # selection and recombination
            arf_sorted_idx = np.argsort(arf)
            xold = mean.copy()
            mean = np.sum(weights.reshape(-1,1) * arx[arf_sorted_idx[:mu]], axis=0)
            # update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (mean - xold) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen+1))) < 1.4 + 2.0 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - xold) / sigma
            # update covariance
            artmp = (arx[arf_sorted_idx[:mu]] - xold) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp * weights.reshape(-1,1)).T.dot(artmp)
            # step size adaptation
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) / (1 - (1 - cs) ** (2 * (gen+1))) - 1))
            # adapt to bounds: recenter mean if moved out
            mean = np.clip(mean, lb, ub)
            gen += 1
            # check stagnation and restart
            stall_counter += 1
            if stall_counter >= stall_limit and budget >= lambda_:
                # restart: keep best, add perturbation
                pert_std = 0.3 * sigma * (ub - lb)
                new_mean = np.clip(best_x + rng.randn(dim) * pert_std, lb, ub)
                mean = new_mean
                sigma = min(0.5, sigma * 2)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                eigenval = None
                eigenvec = None
                stall_counter = 0
                # evaluate new mean
                if budget > 0:
                    f_new = func(mean)
                    budget -= 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = mean.copy()
                        report_best(best_f, best_x)
            # ensure C is symmetric
            C = (C + C.T) / 2
            # enforce eigenvalues positive
            try:
                eigenval, eigenvec = np.linalg.eigh(C)
            except:
                pass

        return best_f, best_x