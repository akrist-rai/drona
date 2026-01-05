"""
CPU Inference API for Newton-GraphMamba

This API loads a model trained on Kaggle GPU and runs inference on CPU
using the pure PyTorch Mamba implementation (mamba_simple).
"""

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from pathlib import Path
import json

from newton_graphmamba import NewtonGraphMamba
from mamba_simple import replace_mamba_with_simple


app = Flask(__name__)
model = None
graph_data = None
device = "cpu"


def load_model(model_path, graph_path=None):
    """
    Load trained model and convert Mamba layers to CPU-compatible version.
    
    Args:
        model_path: Path to saved model (.pt file)
        graph_path: Path to saved graph data (.pt file)
    """
    global model, graph_data
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Default config (adjust based on your training)
        config = {
            'num_nodes': 50000,
            'node_features': 6,
            'd_model': 256,
            'n_layers': 4,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2
        }
    
    # Initialize model
    model = NewtonGraphMamba(**config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Replace Mamba layers with CPU-compatible version
    model = replace_mamba_with_simple(model)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Load graph data if provided
    if graph_path and Path(graph_path).exists():
        print(f"Loading graph from {graph_path}...")
        graph_data = torch.load(graph_path, map_location=device)
        print(f"Graph loaded: {graph_data.num_nodes} nodes")
    else:
        print("Warning: No graph data provided. Graph memory will be None.")
    
    return model, graph_data


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'graph_loaded': graph_data is not None,
        'device': device
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict next nodes in route.
    
    Request JSON:
    {
        "route": [node_id1, node_id2, ..., node_idN],  # Current route sequence
        "vehicle_id": 123,  # Vehicle ID (optional, defaults to 0)
        "top_k": 5  # Number of top predictions to return (optional)
    }
    
    Response JSON:
    {
        "predictions": [
            {"node_id": 1234, "probability": 0.85},
            {"node_id": 5678, "probability": 0.12},
            ...
        ],
        "next_node": 1234  # Most likely next node
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Parse input
        route = data.get('route', [])
        vehicle_id = data.get('vehicle_id', 0)
        top_k = data.get('top_k', 5)
        
        if not route:
            return jsonify({'error': 'Route sequence is required'}), 400
        
        # Convert to tensor
        route_tensor = torch.tensor([route], dtype=torch.long).to(device)
        vehicle_id_tensor = torch.tensor([vehicle_id], dtype=torch.long).to(device)
        
        # Forward pass
        with torch.no_grad():
            # Get graph memory if available
            graph_memory = None
            if graph_data is not None:
                graph_memory = model.forward_encoder(graph_data)
            
            # Predict
            logits = model.forward_decoder(
                sequence_input=route_tensor,
                vehicle_id=vehicle_id_tensor,
                graph_memory=graph_memory
            )
            
            # Get predictions for last timestep
            last_logits = logits[0, -1, :]  # [num_nodes]
            probabilities = torch.softmax(last_logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))
            
            # Format response
            predictions = [
                {
                    'node_id': int(idx.item()),
                    'probability': float(prob.item())
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
            response = {
                'predictions': predictions,
                'next_node': int(top_indices[0].item()),
                'confidence': float(top_probs[0].item())
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    """
    Predict full sequence continuation.
    
    Request JSON:
    {
        "route": [node_id1, node_id2, ...],
        "vehicle_id": 123,
        "length": 10  # Number of steps to predict
    }
    
    Response JSON:
    {
        "predicted_route": [node_id1, node_id2, ..., node_idN],
        "probabilities": [prob1, prob2, ..., probN]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        route = data.get('route', [])
        vehicle_id = data.get('vehicle_id', 0)
        length = data.get('length', 10)
        
        if not route:
            return jsonify({'error': 'Route sequence is required'}), 400
        
        predicted_route = route.copy()
        probabilities = []
        
        # Get graph memory if available
        graph_memory = None
        if graph_data is not None:
            with torch.no_grad():
                graph_memory = model.forward_encoder(graph_data)
        
        # Predict step by step
        for _ in range(length):
            route_tensor = torch.tensor([predicted_route], dtype=torch.long).to(device)
            vehicle_id_tensor = torch.tensor([vehicle_id], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model.forward_decoder(
                    sequence_input=route_tensor,
                    vehicle_id=vehicle_id_tensor,
                    graph_memory=graph_memory
                )
                
                last_logits = logits[0, -1, :]
                probabilities_t = torch.softmax(last_logits, dim=-1)
                
                # Get most likely next node
                next_node = torch.argmax(probabilities_t).item()
                prob = probabilities_t[next_node].item()
                
                predicted_route.append(next_node)
                probabilities.append(prob)
        
        response = {
            'predicted_route': predicted_route,
            'probabilities': probabilities,
            'original_length': len(route),
            'predicted_length': length
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Newton-GraphMamba Inference API')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model (.pt file)')
    parser.add_argument('--graph_path', type=str, default=None,
                        help='Path to graph data (.pt file)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to bind to')
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path, args.graph_path)
    
    # Start server
    print(f"\nStarting inference API on {args.host}:{args.port}")
    print("Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Predict next node")
    print("  POST /predict_sequence - Predict route continuation")
    print("\n")
    
    app.run(host=args.host, port=args.port, debug=False)

