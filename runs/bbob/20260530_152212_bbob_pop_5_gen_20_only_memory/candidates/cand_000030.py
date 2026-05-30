import numpy as np

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
        # Memory for F and CR
        memory_size = 5
        F_memory = [0.8] * memory_size
        CR_memory = [0.9] * memory_size
        memory_idx = 0

        # Initialize population
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

        # Main loop
        while evals < budget:
            # Sample F and CR from memory with added noise
            F = np.random.choice(F_memory) + 0.1 * np.random.randn()
            CR = np.random.choice(CR_memory) + 0.1 * np.random.randn()
            F = np.clip(F, 0.1, 1.0)
            CR = np.clip(CR, 0.0, 1.0)

            # DE/rand/1/bin
            success_F = []
            success_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
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

            # Update memory
            if success_F:
                F_memory[memory_idx] = np.mean(success_F)
                CR_memory[memory_idx] = np.mean(success_CR)
                memory_idx = (memory_idx + 1) % memory_size

            # Local refinement with exploration
            if evals < budget:
                ratio = 1.0 - evals / budget
                sigma = 0.2 * (ub - lb).mean() * ratio ** 0.5
                n_local = min(20, (budget - evals) // 2 + 1)
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    # Random direction, sometimes large jump
                    if np.random.rand() < 0.2:
                        delta = np.random.uniform(lb - best_x, ub - best_x)
                    else:
                        delta = sigma * np.random.randn(dim)
                    trial = best_x + delta
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        # Line search
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
                        # Replace a random individual
                        idx = np.random.randint(pop_size)
                        pop[idx] = trial
                        fitness[idx] = val

            # Periodic reinitialization of worst members
            if evals < budget and evals % max(1, budget // 10) < pop_size:
                num_repl = max(1, pop_size // 5)
                worst_idx = np.argsort(fitness)[-num_repl:]
                for idx in worst_idx:
                    if evals >= budget:
                        break
                    # Random point or perturbed best
                    if np.random.rand() < 0.5:
                        new_x = np.random.uniform(lb, ub)
                    else:
                        new_x = best_x + 0.2 * np.random.randn(dim) * (ub - lb)
                        new_x = np.clip(new_x, lb, ub)
                    val = func(new_x)
                    evals += 1
                    if val < fitness[idx]:
                        pop[idx] = new_x
                        fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)

        return best_val, best_x