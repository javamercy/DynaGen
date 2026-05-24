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
        rng = self.rng
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        # Initial population
        pop_size = max(4, min(10 * dim, budget // 2))
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        values = np.array([func(p) for p in pop])
        evals = pop_size
        best_idx = np.argmin(values)
        best_x = pop[best_idx].copy()
        best_value = values[best_idx]
        report_best(best_value, best_x)
        
        # DE parameters with dithering
        F_base = 0.8
        CR_base = 0.9
        
        no_improve_evals = 0
        restart_threshold = int(0.15 * budget)
        local_budget = min(50, max(10, int(0.05 * budget)))
        
        while evals < budget:
            # Check if stagnation for restart
            if no_improve_evals >= restart_threshold and evals < budget:
                # Perturb best solution for restart
                best_x = best_x + 1e-3 * (ub - lb) * rng.randn(dim)
                np.clip(best_x, lb, ub, out=best_x)
                # Reinitialize population around best
                pop = rng.uniform(lb, ub, size=(pop_size, dim))
                pop[0] = best_x
                for i in range(1, pop_size):
                    # Mix with best
                    pop[i] = best_x + 0.1 * rng.uniform(lb, ub, size=dim) * (ub - lb)
                    np.clip(pop[i], lb, ub, out=pop[i])
                values = np.array([func(p) for p in pop])
                evals += pop_size
                for i in range(pop_size):
                    if values[i] < best_value:
                        best_value = values[i]
                        best_x = pop[i].copy()
                        report_best(best_value, best_x)
                no_improve_evals = 0
                continue
            
            # DE generation
            F = F_base + 0.2 * rng.rand()
            CR = CR_base + 0.1 * rng.rand()
            new_pop = np.empty_like(pop)
            for i in range(pop_size):
                # Select three distinct random indices not equal to i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                np.clip(trial, lb, ub, out=trial)
                new_pop[i] = trial
            
            # Evaluate new population
            new_values = np.array([func(p) for p in new_pop])
            evals += pop_size
            improved = False
            for i in range(pop_size):
                if new_values[i] < values[i]:
                    pop[i] = new_pop[i]
                    values[i] = new_values[i]
                    if new_values[i] < best_value:
                        best_value = new_values[i]
                        best_x = new_pop[i].copy()
                        report_best(best_value, best_x)
                        improved = True
            if not improved:
                no_improve_evals += pop_size
            else:
                no_improve_evals = 0
            
            # If budget low, break
            if evals >= budget - local_budget:
                break
        
        # Local refinement with pattern search on best
        remaining = budget - evals
        if remaining > 0:
            step = (ub - lb) / 50.0  # initial step size per dimension
            x = best_x.copy()
            f = best_value
            # Simple coordinate descent with adaptive steps
            for _ in range(min(remaining, local_budget)):
                improved = False
                for d in range(dim):
                    # Try positive step
                    cand = x.copy()
                    cand[d] = min(ub[d], cand[d] + step[d])
                    cand[d] = max(lb[d], cand[d])
                    val = func(cand)
                    evals += 1
                    if val < f:
                        f = val
                        x = cand.copy()
                        improved = True
                        report_best(f, x)
                        continue
                    # Try negative step
                    cand = x.copy()
                    cand[d] = max(lb[d], cand[d] - step[d])
                    cand[d] = min(ub[d], cand[d])
                    val = func(cand)
                    evals += 1
                    if val < f:
                        f = val
                        x = cand.copy()
                        improved = True
                if improved:
                    step *= 1.2  # increase step on success
                else:
                    step *= 0.8  # decrease step on failure
                    step = np.maximum(step, (ub - lb) * 1e-10)
                if evals >= budget:
                    break
            if f < best_value:
                best_value = f
                best_x = x.copy()
                report_best(best_value, best_x)
        
        return best_value, best_x