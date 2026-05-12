from gurobipy import Model, GRB, quicksum


def vrptw(Customers: list[int],
          Demand: list[int],
          LowerTimeWindow: list[int],
          UpperTimeWindow: list[int],
          Distance: list[list[int]],
          ServiceTime: list[int],
          Capacity: int,
          num_vehicles: int,
          M: int):
    """
    Args:
        Customers: Set of customers
        Demand: Demand of customer `i`
        LowerTimeWindow: Lower Bound of the Time Window of customer `i`
        UpperTimeWindow: Upper Bound of the Time Window of customer `i`
        Distance: Distance or cost between pair of customer `i` and customer `j`
        ServiceTime: Service _time at customer `i`
        Capacity: Capacity of the vehicles
        num_vehicles: Number of vehicles available
        M: A large constant
    Returns:
        MinTotalDistance: Minimize the total distance or cost
    """
    model = Model("VRPTW")

    # Add depot as customer 0
    N = len(Customers)
    V = range(num_vehicles)

    # Decision Variables
    x = {}  # Binary variable indicating if vehicle k travels from i to j
    for k in V:
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')

    # Time variable for each customer
    t = {}
    for i in range(N + 1):
        t[i] = model.addVar(name=f't_{i}')

    # Objective: Minimize total distance
    model.setObjective(
        quicksum(Distance[i][j] * x[i, j, k]
                 for k in V
                 for i in range(N + 1)
                 for j in range(N + 1)
                 if i != j),
        GRB.MINIMIZE)

    # Constraints

    # Each customer must be visited exactly once
    for i in range(1, N + 1):
        model.addConstr(
            quicksum(x[i, j, k]
                     for k in V
                     for j in range(N + 1)
                     if i != j) == 1)

    # Flow conservation
    for k in V:
        for h in range(N + 1):
            model.addConstr(
                quicksum(x[i, h, k] for i in range(N + 1) if i != h) ==
                quicksum(x[h, j, k] for j in range(N + 1) if j != h))

    # Vehicle capacity
    for k in V:
        model.addConstr(
            quicksum(Demand[i] * quicksum(x[i, j, k] for j in range(N + 1) if i != j)
                     for i in range(1, N + 1)) <= Capacity)

    # Time windows
    for i in range(1, N + 1):
        model.addConstr(t[i] >= LowerTimeWindow[i])
        model.addConstr(t[i] <= UpperTimeWindow[i])

    # Time consistency
    for k in V:
        for i in range(N + 1):
            for j in range(1, N + 1):
                if i != j:
                    model.addConstr(
                        t[i] + ServiceTime[i] + Distance[i][j] <=
                        t[j] + M * (1 - x[i, j, k]))

    # Each vehicle starts from depot
    for k in V:
        model.addConstr(
            quicksum(x[0, j, k] for j in range(1, N + 1)) <= 1)

    # Solve the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return float('inf')
