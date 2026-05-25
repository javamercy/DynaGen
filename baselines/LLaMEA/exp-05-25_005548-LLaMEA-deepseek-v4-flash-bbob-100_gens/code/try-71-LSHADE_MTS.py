import numpy as np

class LSHADE_MTS:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def _halton(self, n, d, low, high):
        """Generate n low-discrepancy points in [low, high]^d using Halton sequence."""
        primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
                  73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,
                  157,163,167,173,179,181,191,193,197,199,211,223,227,229,
                  233,239,241,251,257,263,269,271,277,281,283,293,307,311,
                  313,317,331,337,347,349,353,359,367,373,379,383,389,397,
                  401,409,419,421,431,433,439,443,449,457,461,463,467,479,
                  487,491,499,503,509,521,523,541]
        if d > len(primes):
            # fallback: generate more primes
            def next_prime(p):
                while True:
                    p += 1
                    for i in range(2, int(p**0.5)+1):
                        if p % i == 0:
                            break
                    else:
                        return p
            while len(primes) < d:
                primes.append(next_prime(primes[-1]))
        result = np.zeros((n, d))
        for i in range(d):
            base = primes[i]
            for j in range(n):
                f = 1.0 / base
                x = 0.0
                k = j
                while k > 0:
                    x += (k % base) * f
                    k //= base
                    f /= base
                result[j, i] = x
        # scramble (simple random shift per dimension)
        for i in range(d):
            result[:, i] = (result[:, i] + np.random.uniform(0, 1, n)) % 1.0
        return low[np.newaxis, :] + result * (high - low)[np.newaxis, :]

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Budget allocation: 80% for LSHADE, 20% for MTS local search
        local_budget = max(50 * dim, int(0.2 * budget))
        main_budget = budget - local_budget

        if main_budget < 20:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Initial population with Halton sequence ----
        NP_init = max(20, int(20 * np.log(dim))) if dim > 1 else 20
        NP = NP_init
        pop = self._halton(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive and success-history memory
        archive = np.empty((0, dim))
        max_archive = NP
        H = 30  # memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0
        success_CR = []
        success_F = []
        success_delta = []

        # ---- Main LSHADE loop ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # Linear population reduction
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # pbest ratio: linear from 0.2 to 0.05
            ratio = 0.2 - 0.15 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_CR = []
            S_F = []
            S_delta = []

            for i in range(NP):
                # Generate CR and F using memory
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = np.clip(CR, 0., 1.)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Mutation: current-to-pbest/1 with archive
                pbest = pop[np.random.choice(pbest_pool)]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)
                combined = np.vstack((pop, archive))
                # ensure r2 is distinct from i and r1
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Boundary handling (reflection + random)
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    S_delta.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update success-history memory (weighted Lehmer mean)
            if S_CR:
                w = np.array(S_delta) / np.sum(S_delta)
                # Lehmer mean for F
                sum_w_F = np.sum(w * np.array(S_F))
                sum_w_F2 = np.sum(w * np.array(S_F) ** 2)
                mean_F = sum_w_F2 / sum_w_F if sum_w_F > 1e-30 else 0.5
                mean_CR = np.sum(w * np.array(S_CR))
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- MTS-LS1 local search ----
        if local_budget > 0:
            x = self.best_x.copy()
            fx = self.best_f
            step = (ub - lb) / 10.0
            step_min = 1e-10 * (ub - lb)
            evals = 0
            improvement = True
            # Each MTS iteration: try all dimensions
            while evals < local_budget and improvement:
                improvement = False
                for i in range(dim):
                    if evals >= local_budget:
                        break
                    # positive perturbation
                    x_plus = x.copy()
                    x_plus[i] += step[i]
                    x_plus = np.clip(x_plus, lb, ub)
                    f_plus = func(x_plus)
                    evals += 1
                    if f_plus < fx:
                        x = x_plus
                        fx = f_plus
                        step[i] *= 2.0
                        improvement = True
                        if fx < self.best_f:
                            self.best_f = fx
                            self.best_x = x.copy()
                        continue
                    # negative perturbation
                    x_minus = x.copy()
                    x_minus[i] -= step[i]
                    x_minus = np.clip(x_minus, lb, ub)
                    f_minus = func(x_minus)
                    evals += 1
                    if f_minus < fx:
                        x = x_minus
                        fx = f_minus
                        step[i] *= 2.0
                        improvement = True
                        if fx < self.best_f:
                            self.best_f = fx
                            self.best_x = x.copy()
                    else:
                        step[i] /= 2.0
                    # Keep step within bounds
                    step[i] = min(step[i], (ub[i]-lb[i])/2.0)
                    step[i] = max(step[i], step_min[i])
            # If any improvement, update best
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()

        return self.best_f, self.best_x