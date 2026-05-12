import math
import json
import random

from typing import Dict

from baseline.ortool import vrptw

# TODO: double check this stuff

def generate_vrptw_instance(
        num_customers: int,
        vehicle_capacity: int,
        num_vehicles: int = None,
        demand_range: tuple = (5, 30),
        service_time_range: tuple = (5, 20),
        time_window_range: tuple = (50, 300),
        distance_metric: str = 'euclidean',
        seed: int = None
) -> Dict:
    """
    Generates a VRPTW problem instance.

    Args:
        num_customers (int): Number of customers (excluding depot).
        vehicle_capacity (int): Capacity of each vehicle.
        num_vehicles (int, optional): Number of vehicles. If None, it will be calculated based on total demand and capacity.
        demand_range (tuple): (min_demand, max_demand) for each customer.
        service_time_range (tuple): (min_service_time, max_service_time) for each customer.
        time_window_range (tuple): (min_time_window, max_time_window) for each customer.
        distance_metric (str): 'euclidean' or 'random'. Determines how distances are generated.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Dict: A dictionary containing the VRPTW instance parameters.
    """
    if seed is not None:
        random.seed(seed)

    # Initialize Depot
    depot = 0
    Customers = [depot] + list(range(1, num_customers + 1))

    # Generate customer coordinates if using Euclidean distances
    if distance_metric == 'euclidean':
        # Assign random coordinates within a 100x100 grid
        coordinates = {depot: (0, 0)}  # Depot at (0,0)
        for customer in Customers[1:]:
            x = random.randint(0, 100)
            y = random.randint(0, 100)
            coordinates[customer] = (x, y)

        # Compute Euclidean distance matrix
        Distance = []
        for i in Customers:
            row = []
            for j in Customers:
                xi, yi = coordinates[i]
                xj, yj = coordinates[j]
                distance = math.hypot(xi - xj, yi - yj)
                distance = int(distance)  # Convert to integer
                row.append(distance)
            Distance.append(row)
    elif distance_metric == 'random':
        # Generate random distances between 10 and 100
        Distance = []
        for i in Customers:
            row = []
            for j in Customers:
                if i == j:
                    row.append(0)
                else:
                    distance = random.randint(10, 100)
                    row.append(distance)
            Distance.append(row)
    else:
        raise ValueError("Invalid distance_metric. Choose 'euclidean' or 'random'.")

    # Generate demands (0 for depot)
    Demand = [0]
    for _ in Customers[1:]:
        demand = random.randint(demand_range[0], demand_range[1])
        Demand.append(demand)

    # Generate service times (0 for depot)
    ServiceTime = [0]
    for _ in Customers[1:]:
        service_time = random.randint(service_time_range[0], service_time_range[1])
        ServiceTime.append(service_time)

    # Generate time windows
    LowerTimeWindow = []
    UpperTimeWindow = []
    for _ in Customers:
        if _ == depot:
            LowerTimeWindow.append(0)
            UpperTimeWindow.append(1000)  # Depot has a wide time window
        else:
            a = random.randint(time_window_range[0], time_window_range[1] - 50)
            b = random.randint(a + 30, time_window_range[1])
            LowerTimeWindow.append(a)
            UpperTimeWindow.append(b)

    # Calculate total demand to determine number of vehicles if not specified
    total_demand = sum(Demand)
    if num_vehicles is None:
        num_vehicles = math.ceil(total_demand / vehicle_capacity)

    # Set a large constant M
    M = 1000000

    # Assemble the instance dictionary
    return {
        "Customers": Customers,
        "Demand": Demand,
        "LowerTimeWindow": LowerTimeWindow,
        "UpperTimeWindow": UpperTimeWindow,
        "ServiceTime": ServiceTime,
        "Distance": Distance,
        "Capacity": vehicle_capacity,
        "num_vehicles": num_vehicles,
        "M": M
    }


if __name__ == "__main__":
    instances = []

    for _ in range(20):
        input_instance = generate_vrptw_instance(
            num_customers=8,
            vehicle_capacity=100,
            distance_metric='euclidean')

        instances.append({
                "input": input,
                "output": -1,
            })

    with open('testing.json', 'w') as f:
        json.dump(instances, f)
