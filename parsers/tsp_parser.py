import os
import numpy as np

def parse_tsp_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    name = ""
    dimension = 0
    edge_weight_type = ""
    optimal = None
    coords = []
    
    parsing_coords = False
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue
            
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
            if name.endswith(".tsp"):
                name = name[:-4]
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("OPTIMAL"):
            optimal = float(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            parsing_coords = True
            continue
            
        if parsing_coords:
            parts = line.split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
                
    coords = np.array(coords)
    
    # Compute distance matrix according to TSPLIB standard
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.linalg.norm(coords[i] - coords[j])
                if edge_weight_type == 'EUC_2D':
                    dist_matrix[i, j] = int(d + 0.5)
                else:
                    dist_matrix[i, j] = d
                    
    return {
        'name': name,
        'dimension': dimension,
        'coords': coords,
        'dist_matrix': dist_matrix,
        'optimal': optimal
    }

def get_tsp_dataset(directory):
    dataset = []
    for file in sorted(os.listdir(directory)):
        if file.endswith('.tsp'):
            dataset.append(parse_tsp_instance(os.path.join(directory, file)))
    return dataset
