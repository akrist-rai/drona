"""
Utilities for converting NetworkX graphs to PyTorch Geometric format.
"""

import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data


def networkx_to_pyg(G):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Extracts:
    - Node features: speed_limit, lanes, length, highway type, etc.
    - Edge connectivity
    - Edge features: length, maxspeed, etc.
    
    Args:
        G: NetworkX graph (from OSMnx)
    
    Returns:
        pyg_data: PyTorch Geometric Data object
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Create node-to-index mapping
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Extract node features
    node_features = []
    feature_keys = ['x', 'y', 'highway', 'lanes', 'maxspeed', 'length']
    
    for node in nodes:
        node_data = G.nodes[node]
        features = []
        
        # Coordinates
        if 'x' in node_data and 'y' in node_data:
            features.extend([node_data['x'], node_data['y']])
        else:
            features.extend([0.0, 0.0])
        
        # Highway type (one-hot encoding)
        highway = node_data.get('highway', 'unknown')
        highway_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'unknown']
        highway_onehot = [1.0 if highway == ht else 0.0 for ht in highway_types]
        features.extend(highway_onehot)
        
        # Lanes (normalized)
        lanes = node_data.get('lanes', 1)
        if isinstance(lanes, list):
            lanes = lanes[0] if lanes else 1
        features.append(float(lanes) / 4.0)  # Normalize to [0, 1]
        
        # Max speed (normalized)
        maxspeed = node_data.get('maxspeed', 50)
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0] if maxspeed else 50
        if isinstance(maxspeed, str):
            try:
                maxspeed = float(maxspeed.replace('mph', '').replace('km/h', '').strip())
            except:
                maxspeed = 50
        features.append(float(maxspeed) / 120.0)  # Normalize to [0, 1]
        
        # Length (if available)
        length = node_data.get('length', 100.0)
        features.append(float(length) / 1000.0)  # Normalize to km
        
        node_features.append(features)
    
    # Convert to tensor
    node_feat_tensor = torch.tensor(node_features, dtype=torch.float32)
    
    # Extract edge connectivity
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        
        # Add both directions (undirected graph)
        edge_index.append([u_idx, v_idx])
        edge_index.append([v_idx, u_idx])
        
        # Edge features: length, maxspeed
        length = data.get('length', 100.0)
        maxspeed = data.get('maxspeed', 50.0)
        if isinstance(maxspeed, (list, str)):
            try:
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                maxspeed = float(str(maxspeed).replace('mph', '').replace('km/h', '').strip())
            except:
                maxspeed = 50.0
        
        edge_attr.append([length / 1000.0, maxspeed / 120.0])
        edge_attr.append([length / 1000.0, maxspeed / 120.0])
    
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    
    # Create PyG Data object
    pyg_data = Data(
        x=node_feat_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        num_nodes=num_nodes
    )
    
    return pyg_data


def extract_node_features(G, node):
    """
    Extract features for a single node.
    
    Args:
        G: NetworkX graph
        node: Node ID
    
    Returns:
        features: Dictionary of features
    """
    node_data = G.nodes[node]
    return {
        'x': node_data.get('x', 0.0),
        'y': node_data.get('y', 0.0),
        'highway': node_data.get('highway', 'unknown'),
        'lanes': node_data.get('lanes', 1),
        'maxspeed': node_data.get('maxspeed', 50),
        'length': node_data.get('length', 100.0)
    }

