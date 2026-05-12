"""
Line 90 has a syntax error
"""

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
    n = len(all_customers)  # Total number of nodes including the depot

    # Initialize the problem
    prob = LpProblem("VRPTW", LpMinimize)

    # Decision variables
    # x[i][j][k] = 1 if vehicle k travels from i to j, else 0
    x = [[[LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)
           for j in range(n)] for k in range(num_vehicles)] for i in range(n)]

    # t[i] = arrival time at node i
    t = [LpVariable(f"t_{i}", lowBound=0, cat=LpContinuous) for i in range(n)]

    # FinishTime variable
    FinishTime = LpVariable("FinishTime", lowBound=0, cat=LpContinuous)

    # Objective: Minimize FinishTime
    prob += FinishTime, "Minimize_Finish_Time"

    # Constraint 1: Each customer is visited exactly once
    for cust in range(1, n):
        prob += lpSum(x[i][cust][k] for i in range(n) for k in range(num_vehicles)) == 1, f"VisitOnce_{cust}"

    # Constraint 2: Flow conservation for each vehicle
    for k in range(num_vehicles):
        for i in range(n):
            if i == depot:
                continue
            prob += (lpSum(x[i][j][k] for j in range(n))
                     == lpSum(x[j][i][k] for j in range(n))), f"FlowConservation_{i}_Vehicle_{k}"

    # Constraint 3: Each vehicle starts and ends at the depot
    for k in range(num_vehicles):
        # Out of depot
        prob += lpSum(x[depot][j][k] for j in range(1, n)) == 1, f"StartDepot_Vehicle_{k}"
        # Into depot
        prob += lpSum(x[j][depot][k] for j in range(1, n)) == 1, f"EndDepot_Vehicle_{k}"

    # Constraint 4: Time window and precedence constraints
    for k in range(num_vehicles):
        for i in range(n):
            for j in range(n):
                if i == j or j == depot:
                    continue
                # If vehicle k travels from i to j, then arrival at j >= arrival at i + service_time + travel_time
                prob += (t[j] >= t[i] + ServiceTime[i] + Distance[i][j]
                         - M * (1 - x[i][j][k])), f"TimeWindow_{i}_{j}_Vehicle_{k}"

    # Constraint 5: Time windows at each customer
    for cust in range(1, n):
        prob += t[cust] >= LowerTimeWindow[cust-1], f"LowerTimeWindow_{cust}"
        prob += t[cust] <= UpperTimeWindow[cust-1], f"UpperTimeWindow_{cust}"

    # Constraint 6: Link FinishTime with return to depot
    for k in range(num_vehicles):
        for j in range(1, n):
            # If vehicle k returns from j to depot, then FinishTime >= arrival time at depot after service
            prob += (FinishTime >= t[j] + ServiceTime[j] + Distance[j][depot]
                     - M * (1 - x[j][depot][k])), f"FinishTime_Vehicle_{k}_Customer_{j}"

    # Constraint 7: Vehicle capacity
    for k in range(num_vehicles):
        prob += (lpSum(Demand[j-1] * x[depot][j][k] for j in range(1, n))
                 <= Capacity), f"Capacity_Vehicle_{k}"

    # Solving the problem
    prob.solve()

    # Retrieve the FinishTime
    FinishTime_val = value(FinishTime)

    return FinishTime_val