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

        # Population size: heuristically based on dimension and budget
        pop_size = min(budget // 2, max(5 * dim, 20))
        pop_size = min(pop_size, budget)
        pop_size = max(pop_size, 4)  # ensure at least 4

        # Latin Hypercube sampling
        lhs = self._latin_hypercube(pop_size, dim, rng)
        pop = lb + (ub - lb) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

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

        # Reserve budget for local search
        local_budget = max(dim, budget // 5)
        gen_budget = budget - local_budget

        max_gen = (gen_budget - pop_size) // pop_size if gen_budget > pop_size else 0
        if max_gen <= 0:
            # Not enough budget for a generation, do random search?
            # But we already have initial pop, just continue with local search
            pass

        stagnation_count = 0
        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 5
        last_best_val = best_val

        # DE main loop
        for gen in range(max_gen):
            if evals >= gen_budget:
                break

            # Adaptive mutation factor: random in [0.5, 1.0]
            F = 0.5 + 0.5 * rng.uniform()
            CR = 0.9

            for i in range(pop_size):
                if evals >= gen_budget:
                    break

                # Select three distinct random indices
                indices = list(range(pop_size))
                indices.remove(i)
                rng.shuffle(indices)
                a, b, c = indices[:3]

                # Mutation: rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
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

            # Stagnation check
            if best_val < last_best_val:
                stagnation_count = 0
                last_best_val = best_val
            else:
                stagnation_count += 1

            if stagnation_count >= stag_limit and evals < gen_budget:
                # Restart worst 30% of population
                worst_indices = np.argsort(pop_fitness)[-max(1, int(0.3 * pop_size)):]
                for idx in worst_indices:
                    if evals >= gen_budget:
                        break
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stagnation_count = 0
                last_best_val = best_val

                # Additionally, perturb the best point
                if evals < gen_budget:
                    perturbation = 0.1 * (ub - lb) * rng.randn(dim)
                    x_pert = np.clip(best_x + perturbation, lb, ub)
                    val = func(x_pert)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_pert.copy()
                        report_best(best_val, best_x)

        # Local search: random perturbation with decaying step size
        step = 0.1 * (ub - lb)
        while evals < budget:
            direction = rng.randn(dim)
            direction = direction / np.linalg.norm(direction)
            step_size = step * (1 - evals / budget)  # decay
            new_x = np.clip(best_x + step_size * direction, lb, ub)
            val = func(new_x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = new_x.copy()
                report_best(best_val, best_x)
                step = step * 1.5  # increase step after improvement
            else:
                step = step * 0.9  # reduce step

        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs