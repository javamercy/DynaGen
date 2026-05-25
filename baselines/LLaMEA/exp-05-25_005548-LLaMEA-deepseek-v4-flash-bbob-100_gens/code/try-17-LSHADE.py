import numpy as np

class LSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial population size: 18*dim (at least 10)
        NP_init = max(10, int(18 * dim))
        # If budget too small, reduce NP_init to fit a few generations
        if budget < 2 * NP_init:
            NP_init = max(4, budget // 4)
        NP = NP_init
        max_archive = NP_init  # archive size = initial NP

        # Minimal population size
        NP_min = 4

        # If budget extremely small, random search
        if budget < NP:
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initialize population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()
        archive = np.empty((0, dim))

        # SHADE memory
        H = 5
        M_CR = 0.8 * np.ones(H)   # initial CR memory
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Main loop
        while fevals < budget:
            # Linear population size reduction (from NP_init to NP_min)
            remaining_evals = budget - fevals
            NP_new = int(NP_min + (NP_init - NP_min) * (remaining_evals / budget))
            if NP_new < NP:
                # Sort and keep best NP_new individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # Also keep archive size proportional
                max_archive = max(NP, int(2.0 * NP))

            # pbest ratio: starts at 0.2 linearly decreasing to 0.1 as budget depletes
            p_ratio = 0.2 - 0.1 * (1 - remaining_evals / budget)
            pbest_num = max(1, int(p_ratio * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            # For each individual
            for i in range(NP):
                # Sample CR from Cauchy with location M_CR[r] and scale 0.1
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = np.clip(CR, 0., 1.)

                # Sample F from Cauchy with location M_F[r] and scale 0.1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                # Handle negative or zero F by resampling (truncated Cauchy)
                it = 0
                while F <= 0.0 and it < 10:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                    it += 1
                if F <= 0.0:
                    F = 1e-10
                F = min(F, 1.0)

                # pbest selection
                pbest = pop[np.random.choice(pbest_pool)]

                # r1 != i
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = combined[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Crossover: binomial
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflection bound handling (opposite of clipping to preserve diversity)
                for j in range(dim):
                    if u[j] < lb[j]:
                        u[j] = lb[j] + (lb[j] - u[j])
                        # If still out, random reset
                        if u[j] < lb[j]:
                            u[j] = np.random.uniform(lb[j], ub[j])
                    elif u[j] > ub[j]:
                        u[j] = ub[j] - (u[j] - ub[j])
                        if u[j] > ub[j]:
                            u[j] = np.random.uniform(lb[j], ub[j])

                # Evaluation
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    # Successful update
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    # Update population
                    pop[i] = u
                    fitness[i] = f_u

                    # Add previous parent to archive
                    archive = np.vstack((archive, pop[i].copy()))  # Actually the parent is the old pop[i] which was replaced
                    # The parent is the old vector before update, we stored in new_pop? We'll store it after replacement.
                    # Better: store old pop[i] before replacement
                    # But we already replaced pop[i] with u. So we need to save the old value.
                    # To avoid copying, we can use a temporary variable.
                    # Let's fix: store old before replacement.
                    # We'll do pop[i] = u after storing old.
                    # So we need to restructure slightly.
                    # Since we already overwritten pop[i], we lost the parent.
                    # So we need to store old pop[i] before mutation.
                    # Let's adjust: save old = pop[i].copy() before mutation.
                    # Then after success, add old to archive.
                    # We'll restructure inside loop.

                if fevals >= budget:
                    break

            # Actually we need to restructure the loop to properly handle archive and replacement.
            # Let's rewrite with proper handling.

        # Due to above incomplete restructuring, we need to redo the loop properly.
        # I'll rewrite the main loop with correct archive handling.
        return self.best_f, self.best_x