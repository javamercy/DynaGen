"""
When I run the code I get the following error:
```
Traceback (most recent call last):
  File "/home/koitu/git/truck-sim/prompts/vrptw/testing.py", line 11, in <module>
    output = vrptw(**sample['input'])
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/koitu/git/truck-sim/prompts/vrptw/o1mini/zeroshot1_1.py", line 56, in vrptw
    prob += lpSum(x[i][cust][k] for i in range(n) for k in range(num_vehicles)) == 1, f"VisitOnce_{cust}"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/koitu/.conda/envs/sim/lib/python3.11/site-packages/pulp/pulp.py", line 2249, in lpSum
    return LpAffineExpression().addInPlace(vector)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/koitu/.conda/envs/sim/lib/python3.11/site-packages/pulp/pulp.py", line 860, in addInPlace
    for e in other:
  File "/home/koitu/git/truck-sim/prompts/vrptw/o1mini/zeroshot1_1.py", line 56, in <genexpr>
    prob += lpSum(x[i][cust][k] for i in range(n) for k in range(num_vehicles)) == 1, f"VisitOnce_{cust}"
                  \~\~\~\~^^^^^^
IndexError: list index out of range
```
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
        Customers: List of customer identifiers (excluding the depot).
        Demand: List where Demand[i] is the demand of customer i (0-based indexing).
        LowerTimeWindow: List where LowerTimeWindow[i] is the earliest time customer i can be served.
        UpperTimeWindow: List where UpperTimeWindow[i] is the latest time customer i can be served.
        Distance: 2D list where Distance[i][j] is the travel time or distance from node i to node j.
                 Node 0 is the depot, and nodes 1 to n are customers.
        ServiceTime: List where ServiceTime[i] is the service time at node i.
                     ServiceTime[0] should be 0 for the depot.
        Capacity: Capacity of each vehicle.
        num_vehicles: Number of vehicles available.
        M: A large constant used for linearization.

    Returns:
        FinishTime: The time when the last truck arrives back at the depot.
    """
    # Number of customers
    n = len(Customers)

    # Input Validation
    assert len(Demand) == n, "Length of Demand must be equal to the number of Customers."
    assert len(LowerTimeWindow) == n, "Length of LowerTimeWindow must be equal to the number of Customers."
    assert len(UpperTimeWindow) == n, "Length of UpperTimeWindow must be equal to the number of Customers."
    assert len(ServiceTime) == n + 1, "Length of ServiceTime must be equal to number of Customers plus Depot."
    assert len(Distance) == n + 1, "Distance matrix must have (n+1) rows (including Depot)."
    for row in Distance:
        assert len(row) == n + 1, "Each row in Distance matrix must have (n+1) columns (including Depot)."
    assert ServiceTime[0] == 0, "ServiceTime for Depot (node 0) must be 0."

    # Initialize the problem
    prob = LpProblem("VRPTW", LpMinimize)

    # Decision variables
    # x[k][i][j] = 1 if vehicle k travels from node i to node j, else 0
    x = {}
    for k in range(num_vehicles):
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    x[k, i, j] = LpVariable(f"x_{k}_{i}_{j}", cat=LpBinary)

    # t[i] = arrival time at node i
    t = {}
    for i in range(n + 1):
        t[i] = LpVariable(f"t_{i}", lowBound=0, cat=LpContinuous)

    # FinishTime variable
    FinishTime = LpVariable("FinishTime", lowBound=0, cat=LpContinuous)

    # Objective: Minimize FinishTime
    prob += FinishTime, "Minimize_Finish_Time"

    # Constraint 1: Each customer is visited exactly once
    for cust in range(1, n + 1):
        prob += lpSum(x[k, i, cust] for k in range(num_vehicles) for i in range(n + 1) if i != cust) == 1, f"VisitOnce_{cust}"

    # Constraint 2: Flow conservation for each vehicle
    for k in range(num_vehicles):
        for node in range(n + 1):
            if node == 0:
                # Skip depot for flow conservation (handled separately)
                continue
            prob += (lpSum(x[k, i, node] for i in range(n + 1) if i != node) ==
                     lpSum(x[k, node, j] for j in range(n + 1) if j != node)), f"FlowConservation_Vehicle_{k}_Node_{node}"

    # Constraint 3: Each vehicle starts and ends at the depot
    for k in range(num_vehicles):
        # Out of depot
        prob += lpSum(x[k, 0, j] for j in range(1, n + 1)) == 1, f"StartDepot_Vehicle_{k}"
        # Into depot
        prob += lpSum(x[k, j, 0] for j in range(1, n + 1)) == 1, f"EndDepot_Vehicle_{k}"

    # Constraint 4: Time window and precedence constraints
    for k in range(num_vehicles):
        for i in range(n + 1):
            for j in range(n + 1):
                if i == j or j == 0:
                    continue
                if (k, i, j) in x:
                    prob += (t[j] >= t[i] + ServiceTime[i] + Distance[i][j] - M * (1 - x[k, i, j])), f"TimeWindow_Vehicle_{k}_From_{i}_To_{j}"

    # Constraint 5: Time windows at each customer
    for cust in range(1, n + 1):
        prob += t[cust] >= LowerTimeWindow[cust - 1], f"LowerTimeWindow_{cust}"
        prob += t[cust] <= UpperTimeWindow[cust - 1], f"UpperTimeWindow_{cust}"

    # Constraint 6: Link FinishTime with return to depot
    for k in range(num_vehicles):
        for j in range(1, n + 1):
            prob += (FinishTime >= t[j] + ServiceTime[j] + Distance[j][0] - M * (1 - x[k, j, 0])), f"FinishTime_Vehicle_{k}_From_{j}_to_Depot"

    # Constraint 7: Vehicle capacity
    for k in range(num_vehicles):
        prob += (lpSum(Demand[cust - 1] * x[k, 0, cust] for cust in range(1, n + 1)) <= Capacity), f"Capacity_Vehicle_{k}"

    # Solve the problem
    prob.solve()

    # Retrieve the FinishTime
    FinishTime_val = value(FinishTime)

    return FinishTime_val