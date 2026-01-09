🧊 Project Newton: ST-Mamba Architecture
📖 The Core Concept: "The Data Cube"

In high-performance traffic prediction, we treat our data not as a simple list, but as a 4-Dimensional Hypercube.

To process this efficiently on a GPU, we visualize the data being loaded into SRAM (fast memory) and processed along different axes. The model acts as a "Scanner" that looks at the cube from two different angles: Space and Time.
The Tensor Shape

    Shape: [Batch, Time, Nodes, Features]

    🧊 The Cube: Represents one batch of simulations.

    📏 Axis X (Nodes): The spatial map (Road Network).

    📏 Axis Y (Time): The sequence history (Traffic Flow).

    📏 Axis Z (Features): The data depth (Speed, Density).

⚙️ The Architecture: "Slice & Skewer"

The ST-Mamba Block interleaves spatial and temporal processing by mathematically "rotating" this cube in memory.
The Process Flowchart
Code snippet

graph TD
    Input[Input Cube<br/>Batch, Time, Nodes, Features] --> Load[Load to SRAM]
    
    subgraph "Pass 1: Spatial Mixing (GNN)"
        Load --"Slice by Time"--> SliceT[Time Slices]
        SliceT --> GNN[GNN: Nodes talk to Neighbors]
        GNN --"Compile Slices"--> Cube1[Spatially Mixed Cube]
    end
    
    subgraph "The Pivot"
        Cube1 --"Transpose / Rotate"--> Rotate[Rotate Dimensions<br/>(Swap Time & Nodes)]
    end
    
    subgraph "Pass 2: Temporal Mixing (Mamba)"
        Rotate --"Slice by Node"--> SkewerN[Node Skewers]
        SkewerN --> Mamba[Mamba: Past talks to Future]
        Mamba --"Compile Skewers"--> Cube2[Temporally Mixed Cube]
    end
    
    Cube2 --> RotateBack[Rotate Back]
    RotateBack --> Final[Final Judgement Layer]
    Final --> Output[Predicted Traffic]

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#9f9,stroke:#333,stroke-width:2px
    style Rotate fill:#ff9,stroke:#333,stroke-dasharray: 5 5

🧠 Detailed Walkthrough
1. The Setup (SRAM Loading)

We load the raw traffic data into GPU memory. At this stage, the data is just numbers; the model doesn't yet know how "Node A" relates to "Node B" or how "Time 1" relates to "Time 2."
2. Pass 1: "Slicing the Bread" (Spatial / GNN)

    The View: We lock the Time Axis. We slice the cube into thin sheets (like sliced bread). Each sheet is a snapshot of the city at one specific second.

    The Action: The GNN (Graph Neural Network) looks at one sheet at a time.

    The Logic: "If Node A is congested on this sheet, pass that message to Neighbor B."

    Result: The cube is reassembled. It looks the same, but now every voxel knows about its physical neighbors.

3. The Pivot: "The Transpose"

    The Challenge: Computers read memory linearly. To process Time efficiently, we need to line up the data chronologically.

    The Action: We mathematically rotate the cube 90 degrees (Transpose).

        Old View: [Time, Nodes]

        New View: [Nodes, Time]

4. Pass 2: "Skewering the Cube" (Temporal / Mamba)

    The View: We lock the Node Axis. We treat the cube as a bundle of long "skewers" or tubes. Each skewer is the complete history of one single intersection.

    The Action: Mamba runs along the length of each skewer.

    The Logic: "Traffic was rising at T=1 and T=2, so it will peak at T=3."

    Result: The cube is reassembled. Now every voxel knows about its historical context.

5. Final Judgement

The cube now contains "Spatio-Temporal Embeddings." We pass this rich data to a final Linear Layer (The Judge) to predict the exact speed for the next timestep.
💻 Code Structure (Conceptual)
Python

class STMambaBlock(nn.Module):
    def forward(self, x_cube, adj_matrix):
        # x_cube: [Batch, Time, Nodes, Features]
        
        # === PASS 1: SPATIAL (GNN) ===
        # View: Process all nodes for a given time
        # We merge Batch and Time to treat them as independent static graphs
        B, T, N, F = x_cube.shape
        x_slices = x_cube.view(B * T, N, F)
        x_spatial = self.gnn(x_slices, adj_matrix)
        
        # === THE PIVOT (Rotate) ===
        # Reshape to group by Node, so Mamba sees a sequence of Time
        # New Shape: [Batch * Nodes, Time, Features]
        x_rotated = x_spatial.view(B, T, N, F).permute(0, 2, 1, 3).reshape(B * N, T, F)
        
        # === PASS 2: TEMPORAL (Mamba) ===
        # View: Process the time axis for a given node
        x_temporal = self.mamba(x_rotated)
        
        # === ROTATE BACK ===
        # Restore original shape [Batch, Time, Nodes, Features]
        x_final = x_temporal.view(B, N, T, F).permute(0, 2, 1, 3)
        
        return x_final


