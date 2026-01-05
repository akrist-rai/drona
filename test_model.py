"""
Simple test script to verify model loading and inference works.
"""

import torch
from newton_graphmamba import NewtonGraphMamba
from mamba_simple import replace_mamba_with_simple


def test_model():
    """Test model initialization and forward pass."""
    print("Testing NewtonGraphMamba model...")
    
    # Model config
    config = {
        'num_nodes': 1000,
        'node_features': 6,
        'd_model': 128,
        'n_layers': 2,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2
    }
    
    # Initialize model
    print("1. Initializing model...")
    model = NewtonGraphMamba(**config)
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Replace with CPU-compatible Mamba
    print("2. Replacing Mamba layers with CPU-compatible version...")
    model = replace_mamba_with_simple(model)
    print("   ✓ Mamba layers replaced")
    
    # Create dummy graph data
    print("3. Creating dummy graph data...")
    from torch_geometric.data import Data
    num_nodes = config['num_nodes']
    num_edges = 2000
    
    # Random node features
    x = torch.randn(num_nodes, config['node_features'])
    
    # Random edge connectivity
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    graph_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    print(f"   ✓ Graph created: {num_nodes} nodes, {num_edges} edges")
    
    # Test forward pass
    print("4. Testing forward pass...")
    batch_size = 4
    seq_len = 10
    sequence_input = torch.randint(0, num_nodes, (batch_size, seq_len))
    vehicle_id = torch.randint(0, 100, (batch_size,))
    
    model.eval()
    with torch.no_grad():
        # Encode graph
        graph_memory = model.forward_encoder(graph_data)
        print(f"   ✓ Graph encoded: {graph_memory.shape}")
        
        # Decode sequence
        logits = model.forward_decoder(
            sequence_input=sequence_input,
            vehicle_id=vehicle_id,
            graph_memory=graph_memory
        )
        print(f"   ✓ Sequence decoded: {logits.shape}")
        
        # Full forward
        logits_full = model(graph_data, sequence_input, vehicle_id)
        print(f"   ✓ Full forward pass: {logits_full.shape}")
    
    print("\n✅ All tests passed!")
    print("\nModel is ready for:")
    print("  - Training on Kaggle (use mamba-ssm)")
    print("  - Inference on CPU (use mamba_simple)")


if __name__ == '__main__':
    test_model()

