"""
NewtonGraphMamba: Graph-Mamba Model for Route Prediction

This model combines:
1. Graph Neural Network (GNN) encoder for spatial graph structure
2. Mamba SSM for efficient sequence modeling
3. Vehicle embedding for multi-agent learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from mamba_ssm import Mamba


class GraphEncoder(nn.Module):
    """Encodes graph structure into node embeddings."""
    
    def __init__(self, node_features, d_model, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer: node_features -> d_model
        self.convs.append(GCNConv(node_features, d_model))
        
        # Hidden layers: d_model -> d_model
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(d_model, d_model))
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes (optional)
        Returns:
            node_embeddings: [num_nodes, d_model]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.norm(x)
        return x


class VehicleEmbedding(nn.Module):
    """Embeds vehicle IDs into feature vectors."""
    
    def __init__(self, num_vehicles, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_vehicles, d_model)
        
    def forward(self, vehicle_id):
        """
        Args:
            vehicle_id: [batch_size]
        Returns:
            vehicle_emb: [batch_size, d_model]
        """
        return self.embedding(vehicle_id)


class NewtonGraphMamba(nn.Module):
    """
    Graph-Mamba model for route prediction.
    
    Architecture:
    1. Graph Encoder: Converts graph structure to node embeddings
    2. Vehicle Embedding: Embeds vehicle IDs
    3. Mamba Decoder: Processes sequences with graph context
    """
    
    def __init__(
        self,
        num_nodes,
        node_features,
        d_model=256,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        num_vehicles=1000,
        graph_encoder_layers=2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            node_features=node_features,
            d_model=d_model,
            num_layers=graph_encoder_layers
        )
        
        # Vehicle embedding
        self.vehicle_embedding = VehicleEmbedding(
            num_vehicles=num_vehicles,
            d_model=d_model
        )
        
        # Node embedding (for route nodes)
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_nodes)
        
        # Graph memory projection (for combining graph context)
        self.graph_memory_proj = nn.Linear(d_model, d_model)
        
    def forward_encoder(self, graph_data):
        """
        Encode graph structure.
        
        Args:
            graph_data: PyG Data object with x, edge_index
        
        Returns:
            graph_memory: Node embeddings [num_nodes, d_model]
        """
        x = graph_data.x
        edge_index = graph_data.edge_index
        
        # Encode graph
        graph_memory = self.graph_encoder(x, edge_index)
        
        return graph_memory
    
    def forward_decoder(
        self,
        sequence_input,
        vehicle_id,
        graph_memory=None
    ):
        """
        Decode sequence with graph context.
        
        Args:
            sequence_input: Route sequence [batch_size, seq_len] (node indices)
            vehicle_id: Vehicle IDs [batch_size]
            graph_memory: Pre-computed graph embeddings [num_nodes, d_model]
                          If None, will need to be computed separately
        
        Returns:
            logits: Next node predictions [batch_size, seq_len, num_nodes]
        """
        batch_size, seq_len = sequence_input.shape
        
        # Embed sequence nodes
        # [batch_size, seq_len, d_model]
        seq_emb = self.node_embedding(sequence_input)
        
        # Embed vehicle IDs
        # [batch_size, d_model]
        vehicle_emb = self.vehicle_embedding(vehicle_id)
        
        # Add vehicle context to each timestep
        # [batch_size, seq_len, d_model]
        vehicle_emb_expanded = vehicle_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = seq_emb + vehicle_emb_expanded
        
        # If graph_memory is provided, incorporate it
        if graph_memory is not None:
            # For each node in sequence, add its graph embedding
            # [batch_size, seq_len, d_model]
            graph_context = graph_memory[sequence_input]
            graph_context_proj = self.graph_memory_proj(graph_context)
            x = x + graph_context_proj
        
        # Process through Mamba layers
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            # Mamba expects [batch, seq_len, d_model]
            residual = x
            x = mamba_layer(x)
            x = layer_norm(x + residual)
        
        # Project to output space
        logits = self.output_proj(x)  # [batch_size, seq_len, num_nodes]
        
        return logits
    
    def forward(self, graph_data, sequence_input, vehicle_id):
        """
        Full forward pass: encode graph + decode sequence.
        
        Args:
            graph_data: PyG Data object
            sequence_input: [batch_size, seq_len]
            vehicle_id: [batch_size]
        
        Returns:
            logits: [batch_size, seq_len, num_nodes]
        """
        # Encode graph
        graph_memory = self.forward_encoder(graph_data)
        
        # Decode sequence
        logits = self.forward_decoder(
            sequence_input=sequence_input,
            vehicle_id=vehicle_id,
            graph_memory=graph_memory
        )
        
        return logits

