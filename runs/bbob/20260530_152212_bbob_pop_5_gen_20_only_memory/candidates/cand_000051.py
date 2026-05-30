class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 6, 4 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F0 = 0.8
        CR0 = 0.9
        sigma0 = 0.2 * (ub - lb).mean()

        memory_size = 5
        F_memory = [F0] * memory_size
        CR_memory = [CR0] * memory_size
        memory_idx = 0

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        last_improvement_evals = evals

        while evals < budget:
            F = np.random.choice(F_memory)
            CR = np.random.choice(CR_memory)
            success_F = []
            success_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = best_x + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    success_F.append(F)
                    success_CR.append(CR)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals

            if success_F:
                F_memory[memory_idx] = np.mean(success_F)
                CR_memory[memory_idx] = np.mean(success_CR)
                memory_idx = (memory_idx + 1) % memory_size

            # intensified local search using quadratic surrogate
            if evals < budget and best_x is not None:
                n_local = min(budget - evals, int(0.1 * budget) + 5)
                local_evals = 0
                # sample points around best
                n_samples = min(5 * dim, n_local // 2)
                samples = []
                values = []
                for _ in range(n_samples):
                    if evals >= budget:
                        break
                    x = best_x + sigma0 * np.random.randn(dim) * 0.5
                    x = np.clip(x, lb, ub)
                    v = func(x)
                    evals += 1
                    local_evals += 1
                    if v < best_val:
                        best_val = v
                        best_x = x.copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals
                    samples.append(x)
                    values.append(v)
                if len(samples) >= 2 * dim:
                    # fit quadratic: solve least squares
                    X = np.array(samples)
                    y = np.array(values)
                    # add bias column
                    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
                    try:
                        coeff, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                        # optimum of quadratic: -0.5 * inv(A) * b (assuming symmetric A)
                        # Here we have linear term: coeff[:dim], quadratic: 0 (no quadratic term because we only have linear)
                        # Actually we need quadratic terms: we can include pairwise products but that's many.
                        # Instead, use simple gradient descent: step = -0.1 * coeff[:dim]
                        step = -0.1 * coeff[:dim]
                        trial = best_x + step
                        trial = np.clip(trial, lb, ub)
                        if evals < budget:
                            v = func(trial)
                            evals += 1
                            local_evals += 1
                            if v < best_val:
                                # line search further
                                direction = trial - best_x
                                line_length = 0.5
                                for _ in range(3):
                                    if evals >= budget:
                                        break
                                    step = best_x + line_length * direction
                                    step = np.clip(step, lb, ub)
                                    val_ls = func(step)
                                    evals += 1
                                    local_evals += 1
                                    if val_ls < best_val:
                                        best_val = val_ls
                                        best_x = step.copy()
                                        report_best(best_val, best_x)
                                        last_improvement_evals = evals
                                        line_length *= 2
                                    else:
                                        line_length *= 0.5
                                        break
                    except np.linalg.LinAlgError:
                        pass

            # restart if no improvement for 15% of budget
            if evals - last_improvement_evals > max(1, budget // 6) and evals < budget:
                std = 0.05 * (ub - lb)
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    new_x = best_x + np.random.randn(dim) * std
                    new_x = np.clip(new_x, lb, ub)
                    val = func(new_x)
                    evals += 1
                    pop[i] = new_x
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                        last_improvement_evals = evals
                # keep memory intact

        return best_val, best_x