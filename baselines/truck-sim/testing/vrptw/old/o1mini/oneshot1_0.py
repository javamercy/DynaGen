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
        ServiceTime: Service time at customer `i`
        Capacity: Capacity of the vehicles
        num_vehicles: Number of vehicles available
        M: A large constant
    Returns:
        FinishTime: The time when the last truck arrives back at the depot
    """
    # Initialize the problem
    prob = pulp.LpProblem("VRPTW", pulp.LpMinimize)

    # Set of customers excluding depot
    customers = Customers
    depot = 0

    # Decision Variables
    x = {}
    for k in range(num_vehicles):
        for i in customers:
            for j in customers:
                if i != j:
                    x[i,j,k] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat='Binary')

    t = {}
    for i in customers:
        t[i] = pulp.LpVariable(f"t_{i}", lowBound=LowerTimeWindow[i], upBound=UpperTimeWindow[i], cat='Continuous')

    FinishTime = pulp.LpVariable("FinishTime", lowBound=0, cat='Continuous')

    # Objective: Minimize FinishTime
    prob += FinishTime, "Minimize_FinishTime"

    # Constraint 1: Each customer is visited exactly once
    for j in customers:
        if j == depot:
            continue
        prob += pulp.lpSum([x[i,j,k] for k in range(num_vehicles) for i in customers if i != j]) == 1, f"Visit_once_{j}"

    # Constraint 2: Each vehicle starts and ends at the depot
    for k in range(num_vehicles):
        prob += pulp.lpSum([x[depot,j,k] for j in customers if j != depot]) == 1, f"Start_depot_vehicle_{k}"
        prob += pulp.lpSum([x[i, depot, k] for i in customers if i != depot]) == 1, f"End_depot_vehicle_{k}"

    # Constraint 3: Flow conservation for each vehicle
    for k in range(num_vehicles):
        for h in customers:
            if h == depot:
                continue
            prob += (pulp.lpSum([x[i,h,k] for i in customers if i != h]) ==
                     pulp.lpSum([x[h,j,k] for j in customers if j != h])), f"Flow_conservation_{h}_vehicle_{k}"

    # Constraint 4: Time window constraints and subtour elimination
    for k in range(num_vehicles):
        for i in customers:
            if i == depot:
                continue
            for j in customers:
                if j == depot or i == j:
                    continue
                prob += t[j] >= t[i] + ServiceTime[i] + Distance[i][j] - M * (1 - x[i,j,k]), f"Time_{i}_{j}_vehicle_{k}"

    # Constraint 5: Capacity constraints
    for k in range(num_vehicles):
        prob += pulp.lpSum([Demand[j] * pulp.lpSum([x[i,j,k] for i in customers if i != j]) for j in customers if j != depot]) <= Capacity, f"Capacity_vehicle_{k}"

    # Constraint 6: Define FinishTime
    for k in range(num_vehicles):
        for i in customers:
            if i == depot:
                continue
            prob += FinishTime >= t[i] + ServiceTime[i] + Distance[i][depot] * pulp.lpSum([x[i, depot, k]]), f"FinishTime_constraint_vehicle_{k}_customer_{i}"

    # Solve the problem
    prob.solve()

    # Extract FinishTime
    if pulp.LpStatus[prob.status] == 'Optimal':
        return int(pulp.value(FinishTime))
    else:
        return -1  # Indicates no feasible solution found