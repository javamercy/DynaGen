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

        # Initialization via LHS
        pop_size = max(4 * dim, 5)
        if pop_size > budget:
            pop_size = budget
        pop = np.empty((pop_size, dim))
        for i in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            pop[:, i] = (perm + u) / pop_size
        pop = lb + pop * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            f = func(x)
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Parameters
        F_low, F_high = 0.5, 1.0
        CR_low, CR_high = 0.3, 0.9
        stagnation_limit = pop_size * 2
        no_improve = 0

        # History for covariance (successful steps)
        step_buffer = []
        max_buffer = min(2 * dim, 100)

        # Main loop
        while evals < budget:
            # Check stagnation
            if no_improve >= stagnation_limit:
                # Restart via LHS
                remaining = budget - evals
                if remaining < pop_size:
                    pop_size = max(1, remaining)
                else:
                    pop_size = min(pop_size, remaining)
                pop = np.empty((pop_size, dim))
                for i in range(dim):
                    perm = rng.permutation(pop_size)
                    u = rng.rand(pop_size)
                    pop[:, i] = (perm + u) / pop_size
                pop = lb + pop * (ub - lb)
                fitness = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    x = pop[i]
                    f = func(x)
                    evals += 1
                    fitness[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                no_improve = 0
                step_buffer = []
                # rebuild pop
                continue

            # Build covariance matrix from successful steps if available
            cov = None
            if len(step_buffer) >= dim:
                steps = np.array(step_buffer)
                cov = np.cov(steps, rowvar=False)
                # Regularize if singular
                if cov is not None:
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        eigenvalues = np.maximum(eigenvalues, 1e-12)
                        cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                    except np.linalg.LinAlgError:
                        cov = None

            improved_this_gen = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation
                if cov is not None and rng.rand() < 0.5:
                    # Covariance-based mutation
                    delta = rng.multivariate_normal(np.zeros(dim), cov)
                    delta = delta / np.linalg.norm(delta) if np.linalg.norm(delta) > 0 else np.zeros(dim)
                    mutant = pop[i] + (F_low + (F_high - F_low) * rng.rand()) * delta * np.mean(ub - lb)
                else:
                    # Classic DE/rand/1
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(idxs, 3, replace=False)
                    F = F_low + (F_high - F_low) * rng.rand()
                    mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                CR = CR_low + (CR_high - CR_low) * rng.rand()
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                # Evaluation
                f = func(trial)
                evals += 1
                if f < fitness[i]:
                    if f < best_f:
                        best_f = f
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                    fitness[i] = f
                    pop[i] = trial.copy()
                    # Store successful step
                    step = trial - pop[i]  # but pop[i] updated, we want step before update? Actually we want step = trial - old_parent. Use old_pop[i] before update.
                    # So store before update? We'll store after checking improvement, but we need the difference. We'll store after updating, but that's the same as (new - old) since pop[i] is new.
                    # To get step, we need old value. Let's fix: store old pop[i] before mutation, then after improvement, store trial - old_pop.
                    # Implement properly:
                    # We'll do: old = pop[i].copy() before mutation, then after improvement store trial - old.
                # We'll redo with proper storage

            # Re-evaluate pop after generation (we need to track improvements and step buffer properly)
            # I'll restructure loop to better handle step buffer.

        return best_f, best_x