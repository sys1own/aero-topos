import numpy as np

class TopologicalArena:
    """
    Coordinate matrix arena for tracking node-pair interactions in the evolutionary optimizer.
    Manages a 2D adjacency matrix and active node registry for tensor contraction operations.
    """
    def __init__(self, max_coordinate_slots: int = 64):
        self.max_slots = max_coordinate_slots
        self.arena = np.zeros((max_coordinate_slots, max_coordinate_slots), dtype=np.float64)
        self.active_nodes = {}

    def register_node_pair(self, i: int, j: int, weight: float = 1.0) -> None:
        """Register a node pair interaction in the arena matrix with bounds checking."""
        if i < 0 or i >= self.max_slots or j < 0 or j >= self.max_slots:
            return  # Gracefully ignore out-of-bounds to protect isolation boundaries
        self.arena[i, j] = weight
        self.active_nodes[(i, j)] = weight

class GoITopologicalArena:
    """
    Upgraded Phase III Coordinate Matrix tracking open sub-sets,
    subobject classification partitions, and block-sparse solvers.
    """
    def __init__(self, num_ports=8):
        self.P = num_ports
        # Pre-allocated structural routing matrices
        self.M = np.eye(self.P, dtype=np.float64)
        self.U = np.zeros((self.P, self.P), dtype=np.float64)
        
    def configure_block_sparse_partition(self):
        """
        Splits the graph space into internal variables (I) and boundary/routing rings (B)
        to enable optimized Schur complement matrix factorizations.
        """
        # For an 8x8 space: first half acts as internal, second half as boundary interfaces
        midpoint = self.P // 2
        internal_indices = np.arange(0, midpoint)
        boundary_indices = np.arange(midpoint, self.P)
        return internal_indices, boundary_indices

    def set_wire_permutation(self, Custom_M):
        assert Custom_M.shape == (self.P, self.P)
        self.M = Custom_M.copy()

    def set_routing_operator(self, Custom_U):
        assert Custom_U.shape == (self.P, self.P)
        self.U = Custom_U.copy()
