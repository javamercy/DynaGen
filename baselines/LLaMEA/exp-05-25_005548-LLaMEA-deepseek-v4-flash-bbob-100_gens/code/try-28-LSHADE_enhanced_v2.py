import numpy as np

class LSHADE_enhanced_v2:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb) if hasattr(func.bounds, 'lb') else -5.0
        ub = np.array(func.bounds.ub) if hasattr(func.bounds, 'ub') else 5.0
        dim = self.dim
        budget = self.budget

        # Population size (larger initial for diversity)
        NP_init = max(8, int(18 * np.log(dim)) if dim > 1 else 18)
        NP = NP_init
        max_archive = NP
        H = 10  # memory size

        # If budget very small, random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Memory for CR and F
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation tracking
        last_improve_evals = 0
        local_search_trigger = max(100, int(0.1 * budget))

        # Main loop
        while fevals < budget:
            # Linear population reduction (NP from NP_init to 4)
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio (non‑linear decrease)
            ratio = 0.2 * (1 - (1 - remaining_evals / budget) ** 2)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR and F from Cauchy using memory
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Combine pop and archive for r2
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                # Evaluate
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Update archive with parent
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improve_evals = fevals

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update parameter memory
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Stagnation-triggered local search on best solution
            if (fevals - last_improve_evals >= local_search_trigger and
                remaining_evals > 0.1 * budget):
                best_perturbed = False
                for _ in range(min(5, remaining_evals - 1)):
                    sigma = 0.01 * (ub - lb)
                    trial = self.best_x + np.random.normal(0, sigma)
                    trial = np.clip(trial, lb, ub)
                    ft = func(trial)
                    fevals += 1
                    if ft < self.best_f:
                        self.best_f = ft
                        self.best_x = trial.copy()
                        best_perturbed = True
                        last_improve_evals = fevals
                        break
                if best_perturbed:
                    # Replace worst individual with new best
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = self.best_x
                    fitness[worst_idx] = self.best_f

            # Diversity check and partial restart (if budget > 20% left)
            if remaining_evals > 0.2 * budget and fevals > 0.1 * budget:
                mean_pop = np.mean(pop, axis=0)
                std_pop = np.std(pop, axis=0)
                if np.all(std_pop < 1e-3 * (ub - lb)):
                    # Reinitialize 20% worst individuals (keep best)
                    num_reinit = max(1, int(0.2 * NP))
                    reinit_idx = np.argsort(fitness)[-num_reinit:]
                    for idx in reinit_idx:
                        pop[idx] = np.random.uniform(lb, ub)
                        fitness[idx] = func(pop[idx])
                        fevals += 1
                        if fevals >= budget:
                            break
                    # Reset stagnation counter
                    last_improve_evals = fevals

        return self.best_f, self.best_x