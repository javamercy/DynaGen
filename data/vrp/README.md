# VRP Data

These train and test instances are copied from `baselines/truck-sim/testing/vrp` and used by DynaGen's VRP integration.

Each pickle item is a tuple of `(coordinates, distance_matrix, reference)`, where `reference` contains the OR-Tools minimax route baseline (`max_distance`, `routes`, and related fields).
