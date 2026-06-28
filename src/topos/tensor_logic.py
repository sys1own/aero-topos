import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

class TensorLogicEngine:
    """
    Tensor deduction engine for logical inference via matrix multiplication.
    Safely casts JAX device arrays to NumPy for compatibility with validation assertions.
    """
    def __init__(self):
        pass

    def execute_deduction(self, tA, tB):
        """
        Execute tensor deduction via matrix multiplication.
        Converts JAX device arrays to NumPy to avoid type mismatch errors in validation.
        If tA[i,j]=1 and tB[j,k]=1, then result[i,k]=1 (transitive closure step).
        """
        # Safely convert JAX device arrays to NumPy arrays
        A = np.asarray(tA)
        B = np.asarray(tB)
        return np.dot(A, B)

class GeometryOfInteractionSolver:
    """
    Implements Girard's GoI execution formula EX(M, U) = (I - U*M)^{-1} * U
    and computes exact analytical metamorphic gradients of the loss with respect
    to the rule matrix U without requiring full, explicit backpropagation.
    """
    def __init__(self, edge_dimension: int):
        self.dim = edge_dimension

    @staticmethod
    @jit
    def execute_goi_wave(M: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the forward wave step: EX(M, U) = (I - U * M)^(-1) * U.
        M is the block-sparse Hashimoto directed-edge adjacency matrix.
        U is the dynamic routing rule matrix.
        """
        I = jnp.eye(U.shape[0], dtype=U.dtype)
        inv_term = jnp.linalg.inv(I - jnp.dot(U, M))
        return jnp.dot(inv_term, U)

    @staticmethod
    @jit
    def compute_metamorphic_gradients(M: jnp.ndarray, U: jnp.ndarray, 
                                     loss_grad_out: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the analytical gradient dL/dU of the routing rule matrix U.
        Letting X = (I - U * M)^(-1), the exact derivative is computed as:
        grad_U = X^T * loss_grad_out * (I + M * EX_current)^T
        This avoids explicit backpropagation through recursive solver steps.
        """
        I = jnp.eye(U.shape[0], dtype=U.dtype)
        inv_term = jnp.linalg.inv(I - jnp.dot(U, M))
        EX = jnp.dot(inv_term, U)
        
        inv_trans = jnp.conjugate(inv_term).T
        right_factor = I + jnp.dot(jnp.conjugate(M).T, jnp.conjugate(EX).T)
        grad_U = jnp.dot(inv_trans, jnp.dot(loss_grad_out, right_factor))
        return grad_U
