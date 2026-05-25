import numpy as np

class JADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        # access bounds (assuming func.bounds.lb, .ub exist)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # population size (commonly 4+3*log(dim))
        NP = max(5, int(4 + 3 * np.log(dim)))
        # ensure budget can support at least one generation
        if budget < NP:
            # fallback to random search
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # initialize population uniformly
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        # best so far
        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # archive for inferior solutions (start empty)
        archive = np.empty((0, dim))

        # initialize parameter means
        mu_F = 0.5
        mu_CR = 0.5
        # list to store successful F, CR values
        F_rec = []
        CR_rec = []

        # main loop
        while fevals < budget:
            # compute pbest_top: top p% (p=0.1)
            p = 0.1
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_idx = sorted_idx[:pbest_num]

            # for each target vector
            new_pop = np.empty((NP, dim))
            new_fitness = np.empty(NP)
            successful_F = []
            successful_CR = []

            for i in range(NP):
                # generate CR_i from normal, F_i from Cauchy
                CR_i = np.clip(np.random.normal(mu_CR, 0.1), 0, 1)
                F_i = np.random.standard_cauchy() * 0.1 + mu_F
                while F_i <= 0:
                    F_i = np.random.standard_cauchy() * 0.1 + mu_F
                F_i = min(F_i, 1.0)

                # choose pbest individual
                pbest = pop[np.random.choice(pbest_idx)]

                # choose r1 != i
                r1 = np.random.choice([j for j in range(NP) if j != i])
                # choose r2 from pop ∪ archive
                combined = np.vstack((pop, archive))
                while True:
                    r2 = np.random.randint(len(combined))
                    if combined[r2] is not pop[r1] and combined[r2] is not pop[i]:
                        break

                # mutation
                v = pop[i] + F_i * (pbest - pop[i]) + F_i * (pop[r1] - combined[r2])
                # binomial crossover
                u = np.where(np.random.rand(dim) < CR_i, v, pop[i])
                # ensure at least one component from v (not strictly needed for JADE)
                j_rand = np.random.randint(dim)
                u[j_rand] = v[j_rand]
                # clamp to bounds
                u = np.clip(u, lb, ub)

                # evaluate trial
                f_u = func(u)
                fevals += 1

                # selection
                if f_u <= fitness[i]:
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    # add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    # keep archive size <= NP
                    if len(archive) > NP:
                        archive = archive[np.random.choice(len(archive), size=NP, replace=False)]
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]

                # update best
                if f_u < self.best_f:
                    self.best_f = f_u
                    self.best_x = u.copy()

                # check budget
                if fevals >= budget:
                    # finish loop early
                    break

            # replace population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # update parameter means if any successful
            if successful_F:
                # use Lehmer mean for F
                F_arr = np.array(successful_F)
                mu_F = np.sum(F_arr**2) / np.sum(F_arr)
                mu_CR = np.mean(successful_CR)

        return self.best_f, self.best_x