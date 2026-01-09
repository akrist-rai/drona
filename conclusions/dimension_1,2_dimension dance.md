The Implementation: The "Dimension Dance"

This is the hardest part to code.

    GNNs want: [Batch * Time, Nodes, Features] (They treat every timestamp as a separate graph).

    Mamba wants: [Batch * Nodes, Time, Features] (It treats every node as a separate timeline).

To interleave them, you must constantly reshape (transpose) the data inside the block.
