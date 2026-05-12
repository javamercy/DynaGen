import pulp


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
    # Initialize the model
    model = pulp.LpProblem("VRPTW", pulp.LpMinimize)

    customer_indices = Customers
    all_customers = set(customer_indices)
    depot = 0  # Assuming depot is indexed by 0
    vehicles = range(num_vehicles)

    # Decision variables
    x = pulp.LpVariable.dicts("route",
                              ((i, j, k) for i in customer_indices for j in customer_indices if i != j for k in
                               vehicles),
                              cat='Binary')

    t = pulp.LpVariable.dicts("time",
                              (i for i in customer_indices),
                              lowBound=0,
                              cat='Continuous')

    # Objective: Minimize total distance
    model += pulp.lpSum(Distance[i][j] * x[i, j, k]
                        for i in customer_indices for j in customer_indices if i != j for k in vehicles)

    # Constraints

    # Each customer is visited exactly once
    for j in customer_indices:
        model += pulp.lpSum(x[i, j, k] for i in customer_indices if i != j for k in vehicles) == 1

    # Flow conservation for each vehicle
    for k in vehicles:
        # Each vehicle starts at the depot
        model += pulp.lpSum(x[depot, j, k] for j in customer_indices if j != depot) <= 1
        # Each vehicle ends at the depot
        model += pulp.lpSum(x[i, depot, k] for i in customer_indices if i != depot) <= 1
        for h in customer_indices:
            if h != depot:
                model += (pulp.lpSum(x[i, h, k] for i in customer_indices if i != h) -
                          pulp.lpSum(x[h, j, k] for j in customer_indices if j != h)) == 0

    # Capacity constraints
    for k in vehicles:
        model += pulp.lpSum(Demand[j] * pulp.lpSum(x[i, j, k] for i in customer_indices if i != j)
                            for j in customer_indices) <= Capacity

    # Time window constraints and Subtour Elimination
    for k in vehicles:
        for i in customer_indices:
            for j in customer_indices:
                if i != j:
                    model += t[i] + ServiceTime[i] + Distance[i][j] - M * (1 - x[i, j, k]) <= t[j]

    for i in customer_indices:
        model += t[i] >= LowerTimeWindow[i]
        model += t[i] <= UpperTimeWindow[i]

    # Depot time window
    model += pulp.lpSum(x[depot, j, k] for j in customer_indices if j != depot for k in vehicles) >= 1
    model += pulp.lpSum(x[i, depot, k] for i in customer_indices if i != depot for k in vehicles) >= 1

    # Solve the model
    solver = pulp.PULP_CBC_CMD()
    model.solve(solver)

    # Check if a feasible solution was found
    if pulp.LpStatus[model.status] == 'Optimal':
        MinTotalDistance = pulp.value(model.objective)
    else:
        MinTotalDistance = -1  # Indicate no feasible solution found

    return MinTotalDistance
