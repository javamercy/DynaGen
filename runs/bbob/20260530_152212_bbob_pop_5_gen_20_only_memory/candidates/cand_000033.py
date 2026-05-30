class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 4, 5 * dim))

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

        # Success-history memory
        memory_size = 5
        F_memory = [F0] * memory_size
        CR_memory = [CR0] * memory_size
        memory_idx = 0

        # Initialization
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

        # Main loop
        while evals < budget:
            # Sample F and CR from memory
            F = np.random.choice(F_memory)
            CR = np.random.choice(CR_memory)
            # DE/best/1/bin
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

            # Update memory with successful parameters
            if success_F:
                F_memory[memory_idx] = np.mean(success_F)
                CR_memory[memory_idx] = np.mean(success_CR)
                memory_idx = (memory_idx + 1) % memory_size

            # Local refinement with adaptive step and line search
            if evals < budget:
                ratio = 1.0 - evals / budget
                sigma = sigma0 * ratio ** 0.5
                n_local = min(20, (budget - evals) // 2 + 1)
                if pop_size > 1:
                    center = pop.mean(axis=0)
                    C = np.cov(pop.T) + 1e-8 * np.eye(dim)
                else:
                    C = np.eye(dim)
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    delta = np.random.multivariate_normal(np.zeros(dim), sigma ** 2 * C)
                    trial = best_x + delta
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        # Line search along direction of improvement
                        direction = trial - best_x
                        line_length = 0.5
                        for _ in range(3):
                            if evals >= budget:
                                break
                            step = best_x + line_length * direction
                            step = np.clip(step, lb, ub)
                            v = func(step)
                            evals += 1
                            if v < best_val:
                                best_val = v
                                best_x = step.copy()
                                report_best(best_val, best_x)
                                line_length *= 2
                            else:
                                line_length *= 0.5
                                break
                        idx = np.random.randint(pop_size)
                        pop[idx] = trial
                        fitness[idx] = val
                        last_improvement_evals = evals
                    else:
                        sigma *= 0.9

            # Restart if no improvement for a while
            if evals - last_improvement_evals > max(1, budget // 10) and evals < budget:
                # Reinitialize population around best with small variance
                std = 0.1 * (ub - lb)
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
                # Reset memory to default values
                F_memory = [F0] * memory_size
                CR_memory = [CR0] * memory_size
                memory_idx = 0

        return best_val, best_x