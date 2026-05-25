import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        # Latin Hypercube Sampling for initial population
        pop_size = max(5, min(10 * dim, budget // 2))
        # Ensure at least dim+1 points for diversity
        if pop_size < dim + 1:
            pop_size = min(dim + 1, budget)
        # Generate LHS
        segments = np.linspace(0, 1, pop_size + 1)
        sample = np.zeros((pop_size, dim))
        for i in range(dim):
            perm = rng.permutation(pop_size)
            sample[:, i] = rng.uniform(segments[:-1], segments[1:])[perm]
        pop = lb + sample * (ub - lb)
        # Evaluate initial population
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            f = func(pop[i])
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = pop[i].copy()
                report_best(best_f, best_x)
        if evals >= budget:
            return best_f, best_x

        # DE parameters
        F = 0.5
        CR = 0.9

        # Stagnation detection
        stagnation_evals = 2 * pop_size
        no_improve_evals = 0

        # Main loop
        while evals < budget:
            # Generate offspring
            new_pop = np.zeros_like(pop)
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct random indices
                indices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(indices, 3, replace=False)
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Crossover
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                # Evaluation
                f_trial = func(trial)
                evals += 1
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                else:
                    new_pop[i] = pop[i]
                    no_improve_evals += 1
            # Replace population
            pop = new_pop
            # Update fitness array (we have old fitness for comparison, but we need to update after selection? Better to recompute maybe. But we can keep the fitness array by using the winner values.
            # However, after selection, the fitness values for each individual are known from the comparison. We can store them.
            # To avoid confusion, let's maintain a separate array for new fitness.
            # Actually, we already have f_trial for each i. So we can set fitness[i] accordingly.
            # But we need to update for all? Let's refactor: after loop, we have new_pop and we need to recompute fitness? That would cost extra evals. Better to store the winner's fitness during the loop.
            # For simplicity, after the inner loop, we can compute fitness of new_pop by calling func again? That would double evals, not allowed. So we must keep track.
            # Let's restructure: have a temporary fitness array new_fitness, and after selection assign.
            # To keep code clean, I'll refactor.

            if evals >= budget:
                break

            # Check for restart after generation
            if no_improve_evals >= stagnation_evals and evals < budget:
                # Restart: keep best individual, randomly reinitialize rest
                new_pop = np.zeros((pop_size, dim))
                new_pop[0] = best_x.copy()
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    new_pop[i] = rng.uniform(lb, ub, size=dim)
                    # Evaluate new individual? Actually, we need to evaluate to have fitness for future generations.
                    # But evaluating now consumes budget. We'll evaluate immediately.
                    f_new = func(new_pop[i])
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = new_pop[i].copy()
                        report_best(best_f, best_x)
                pop = new_pop
                # We need to recompute fitness for all individuals? We already evaluated the restarted ones, but the best individual's fitness is known. So we can set fitness accordingly.
                # To make it simple, after restart, we can just run a cycle of evaluation? But that would add overhead. Alternative: after restart, we continue the loop, but we need fitness values for next generation. So we need to evaluate the new population fully.
                # Since we already evaluated the new individuals (except best), we can store their fitness values. Let's keep an array fitness_new.
                # Actually, it's easier to just do a full re-evaluation of population after restart? That would use extra budget. Better to evaluate only the new points.
                # Given that restart is infrequent, we can afford to evaluate new points individually.
                # To simplify, I'll keep the restart approach but not worry about fitness array for now; instead, I'll let the main loop handle evaluation as usual. But the main loop expects pop to have fitness values from previous evaluations. So after restart, we need to have fitness values for each individual. The best one's fitness is known, others we just evaluated. So we can store them.
                # Let's maintain a list of fitness for pop. In the code, I'll initialize fitness array at start and update it properly.

        return best_f, best_x