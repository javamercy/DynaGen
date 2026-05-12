import pulp
from pulp import LpMinimize, LpProblem, LpVariable, LpBinary, LpContinuous, lpSum, value

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
    Solves the Vehicle Routing Problem with Time Windows (VRPTW) using Mixed-Integer Programming.

    Args:
        Customers: List of customer identifiers.
        Demand: List where Demand[i] is the demand of customer i.
        LowerTimeWindow: List where LowerTimeWindow[i] is the earliest time customer i can be served.
        UpperTimeWindow: List where UpperTimeWindow[i] is the latest time customer i can be served.
        Distance: 2D list where Distance[i][j] is the travel time or distance from customer i to customer j.
        ServiceTime: List where ServiceTime[i] is the service time at customer i.
        Capacity: Capacity of each vehicle.
        num_vehicles: Number of vehicles available.
        M: A large constant used for linearization.

    Returns:
        FinishTime: The time when the last truck arrives back at the depot.
    """
    # Define depot as 0
    depot = 0
    all_customers = [depot] + Customers
    customer_indices = {customer: idx+1 for idx, customer in enumerate(Customers)}
    index_to_customer = {idx+1: customer for idx, customer in enumerate(Customers)}
    all_indices = list(range(len(all_customers)))

    # Number of nodes
    n = len(all_customers)

    # Initialize the problem
    prob = LpProblem("VRPTW", LpMinimize)

    # Decision variables
    # x[i][j][k] = 1 if vehicle k travels from i to j
    x = [[[LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)
            for j in all_indices] for k in range(num_vehicles)] for i in all_indices]

    # t[i] = arrival time at node i
    t = [LpVariable(f"t_{i}", lowBound=0, cat=LpContinuous) for i in all_indices]

    # FinishTime variable
    FinishTime = LpVariable("FinishTime", lowBound=0, cat=LpContinuous)

    # Objective: Minimize FinishTime
    prob += FinishTime, "Minimize_Finish_Time"

    # Constraint 1: Each customer is visited exactly once
    for cust_idx in range(1, n):
        prob += lpSum(x[i][cust_idx][k] for i in all_indices for k in range(num_vehicles)) == 1, f"VisitOnce_{cust_idx}"

    # Constraint 2: Flow conservation for each vehicle
    for k in range(num_vehicles):
        for i in all_indices:
            if i == depot:
                continue
            prob += lpSum(x[i][j][k] for j in all_indices) == lpSum(x[j][i][k] for j in all_indices), f"FlowConservation_{i}_Vehicle_{k}"

    # Constraint 3: Each vehicle starts and ends at the depot
    for k in range(num_vehicles):
        # Out of depot
        prob += lpSum(x[depot][j][k] for j in range(1, n)) == 1, f"StartDepot_Vehicle_{k}"
        # Into depot
        prob += lpSum(x[j][depot][k] for j in range(1, n)) == 1, f"EndDepot_Vehicle_{k}"

    # Constraint 4: Time window and precedence constraints
    for k in range(num_vehicles):
        for i in all_indices:
            for j in all_indices:
                if i == j:
                    continue
                prob += t[j] >= t[i] + ServiceTime[i] + Distance[i][j] - M * (1 - x[i][j][k]), f"TimeWindow_{i}_{j}_Vehicle_{k}"

    # Constraint 5: Time windows
    for i in range(1, n):
        prob += t[i] >= LowerTimeWindow[i], f"LowerTimeWindow_{i}"
        prob += t[i] <= UpperTimeWindow[i], f"UpperTimeWindow_{i}"

    # Constraint 6: FinishTime is the maximum return time to depot
    for k in range(num_vehicles):
        prob += t[depot] >= t[j] + Distance[j][depot] + ServiceTime[j] - M * (1 - x[j][depot][k]) for j in range(1, n)

    prob += FinishTime >= t[depot], "SetFinishTime"

    # Constraint 7: Vehicle capacity
    for k in range(num_vehicles):
        prob += lpSum(Demand[j-1] * x[depot][j][k] for j in range(1, n)) <= Capacity, f"Capacity_Vehicle_{k}"

    # Solve the problem
    prob.solve()
    
    # Retrieve the FinishTime
    FinishTime_val = value(FinishTime)

    return FinishTime_val