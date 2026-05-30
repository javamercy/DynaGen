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

        # Reserve budget for local search
        ls_budget = max(2 * dim, budget // 3)
        # Determine population size for DE
        pop_size = max(3 * dim, min(15 + int(dim**0.5), budget // 2))
        pop_size = min(pop_size, budget - ls_budget)
        if pop_size < 2 * dim:
            pop_size = max(2, min(2 * dim, budget - ls_budget))
        if pop_size < 1:
            pop_size = 1

        # Latin Hypercube initial population
        def lhs(n, d):
            intervals = np.linspace(0, 1, n + 1)
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    samples[i, j] = intervals[perm[i]] + rng.uniform(0, 1/n)
            return samples

        lhs_samples = lhs(pop_size, dim)
        pop = lb + (ub - lb) * lhs_samples

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Determine number of DE generations
        remaining_for_de = budget - evals - ls_budget
        max_gen = max(0, remaining_for_de // pop_size) if pop_size > 0 else 0
        max_gen = min(max_gen, 100)

        # Stagnation parameters
        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 1
        stag_counter = 0
        last_best = best_val

        # Adaptive DE loop
        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                if evals >= budget - ls_budget:
                    break
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                if evals >= budget - ls_budget:
                    break
            if evals >= budget - ls_budget:
                break

            if best_val < last_best:
                stag_counter = 0
                last_best = best_val
            else:
                stag_counter += 1

            if stag_counter >= stag_limit and evals < budget - ls_budget:
                n_replace = max(1, pop_size // 3)
                worst_idx = np.argsort(pop_fitness)[-n_replace:]
                for idx in worst_idx:
                    if evals >= budget - ls_budget:
                        break
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fitness[idx] = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stag_counter = 0
                last_best = best_val

        # Local search: randomized direct search
        if evals < budget:
            remaining = budget - evals
            # Generate random orthogonal directions
            # Use QR decomposition of random matrix to get orthonormal basis
            if dim > 0:
                n_dirs = min(dim, max(1, remaining // 2))
                directions = []
                for _ in range(n_dirs):
                    vec = rng.randn(dim)
                    # Orthogonalize against previous directions (Gram-Schmidt)
                    for d in directions:
                        vec = vec - np.dot(vec, d) * d
                    norm = np.linalg.norm(vec)
                    if norm > 1e-10:
                        vec = vec / norm
                        directions.append(vec)
                if len(directions) == 0:
                    directions = [np.ones(dim) / np.sqrt(dim)]

                # Adaptive step size
                step_size = 0.1 * np.linalg.norm(ub - lb) / np.sqrt(dim) if dim > 0 else 0.1
                step_size = max(step_size, 1e-8)
                shrink = 0.5
                expand = 2.0
                success = True
                while evals < budget:
                    if not success:
                        step_size *= shrink
                        if step_size < 1e-10:
                            break
                    improved = False
                    for d in directions:
                        if evals >= budget:
                            break
                        # Try move in positive direction
                        trial = np.clip(best_x + step_size * d, lb, ub)
                        val = func(trial)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                            improved = True
                            step_size *= expand
                            break
                        # Try move in negative direction
                        trial = np.clip(best_x - step_size * d, lb, ub)
                        val = func(trial)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                            improved = True
                            step_size *= expand
                            break
                    if not improved:
                        success = False
                    else:
                        success = True

        return best_val, best_x