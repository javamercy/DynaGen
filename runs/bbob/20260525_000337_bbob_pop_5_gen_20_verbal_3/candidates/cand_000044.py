import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = self.rng
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        # Evaluate initial point
        best_x = rng.uniform(lb, ub, size=dim)
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)
        
        # Base population size
        lam = 4 + int(3 * np.log(dim))
        lam = max(2, min(lam, budget - evals))
        
        # CMA-ES parameters
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3)**2)
        
        # Initial state
        m = best_x.copy()
        sigma = (ub - lb).mean() / 6.0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        
        # Restart parameters
        max_no_improve = 10 * lam
        no_improve_count = 0
        restart_count = 0
        max_restarts = 3
        
        while evals < budget:
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            # Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            # Generate and reflect
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                x = m + sigma * L @ z
                for j in range(dim):
                    if x[j] < lb[j]:
                        x[j] = lb[j] + (lb[j] - x[j])
                    elif x[j] > ub[j]:
                        x[j] = ub[j] - (x[j] - ub[j])
                pop[i] = np.clip(x, lb, ub)
            # Evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            # Update best
            improved = False
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    report_best(best_value, best_x)
                    improved = True
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += lam_gen
            
            # Update CMA-ES state
            idx = np.argsort(vals)
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # update p_c
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            # update p_s
            try:
                invL = np.linalg.solve(L, np.eye(dim))
            except:
                invL = np.eye(dim)
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            # update sigma
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-12)
            # update C
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
            C += 1e-10 * np.eye(dim)
            
            # Restart condition
            if no_improve_count >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                lam = lam * 2
                lam = min(lam, budget - evals)
                if lam < 2:
                    break
                mu = lam // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
                w = w / np.sum(w)
                mueff = 1.0 / np.sum(w**2)
                cc = 4.0 / (dim + 4.0)
                cs = (mueff + 2.0) / (dim + mueff + 5.0)
                damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
                ccov = 2.0 / ((dim + 1.3)**2)
                m = rng.uniform(lb, ub, size=dim)
                sigma = (ub - lb).mean() / 6.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_count = 0
                
        return best_value, best_x