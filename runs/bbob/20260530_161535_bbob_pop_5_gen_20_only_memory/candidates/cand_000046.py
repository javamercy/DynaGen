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
        domain_range = ub - lb
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_x = None
        best_f = np.inf
        count = 0

        def evaluate(x):
            nonlocal count, best_x, best_f
            x = np.clip(x, lb, ub)
            f = func(x)
            count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(f, best_x)
            return f

        # Evaluate a random point first
        x0 = rng.uniform(lb, ub, size=dim)
        evaluate(x0)

        # Phase 1: Modified CMA-ES (from exploit_focused_cma_es)
        lam = 4 + int(2 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff) * 1.5
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2) ** 2 + mu_eff)) * 1.5
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        max_restarts = max(1, int(budget / (5 * dim)))
        for restart in range(max_restarts + 1):
            if count >= budget:
                break
            sigma = 0.1 * np.mean(domain_range)
            xmean = rng.uniform(lb, ub, size=dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            invsqrtC = np.eye(dim)
            eigen_eval = 0

            if restart > 0:
                evaluate(xmean)
                if count >= budget:
                    break

            while count + lam <= budget:
                arx = []
                arf = []
                for k in range(lam):
                    z = rng.normal(0, 1, dim)
                    y = B @ (D * z)
                    x = xmean + sigma * y
                    x = np.clip(x, lb, ub)
                    arx.append(x)
                    f = evaluate(x)
                    arf.append(f)
                    if count >= budget:
                        break
                if count >= budget:
                    break

                idx = np.argsort(arf)
                xold = xmean.copy()
                xmean = np.sum(weights[:, None] * np.array(arx)[idx[:mu]], axis=0)

                dmean = xmean - xold
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ dmean / sigma)
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (dmean / sigma)

                C *= (1 - c1 - cmu)
                C += c1 * np.outer(pc, pc)
                for i in range(mu):
                    diff = (np.array(arx)[idx[i]] - xold) / sigma
                    C += cmu * weights[i] * np.outer(diff, diff)

                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / (np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))) - 1))

                if count - eigen_eval > dim:
                    eigen_eval = count
                    C = np.triu(C) + np.triu(C, 1).T
                    D, B = np.linalg.eigh(C)
                    D = np.abs(D)
                    D = np.maximum(D, 1e-30)
                    D = np.sqrt(D)
                    invsqrtC = B @ np.diag(1/D) @ B.T

                if sigma < 1e-8 * np.mean(domain_range):
                    break

        # Phase 2: DE refinement (if budget remains)
        if count < budget and best_x is not None:
            pop_size = max(4, int(2 * np.log(dim)))
            if pop_size < 2:
                pop_size = 2
            # Initialize population around best
            sigma_de = 0.1 * np.mean(domain_range)
            pop = np.array([best_x + rng.normal(0, sigma_de, size=dim) for _ in range(pop_size)])
            pop = np.clip(pop, lb, ub)
            fit = np.full(pop_size, np.inf)
            for i in range(pop_size):
                if count >= budget:
                    break
                fit[i] = evaluate(pop[i])
            # DE/best/1/bin loop
            while count < budget:
                for i in range(pop_size):
                    if count >= budget:
                        break
                    # Choose two distinct random indices not equal to i
                    candidates = [j for j in range(pop_size) if j != i]
                    if len(candidates) < 2:
                        continue
                    r = rng.choice(candidates, size=2, replace=False)
                    # Mutation: best + F * (pop[r1] - pop[r2])
                    F = 0.8
                    mutant = best_x + F * (pop[r[0]] - pop[r[1]])
                    # Crossover with target vector
                    trial = pop[i].copy()
                    j_rand = rng.randint(dim)
                    CR = 0.9
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, lb, ub)
                    val = evaluate(trial)
                    if val < fit[i]:
                        pop[i] = trial
                        fit[i] = val
                        if val < best_f:
                            best_f = val
                            best_x = trial.copy()
                            report_best(best_f, best_x)

        return best_f, best_x