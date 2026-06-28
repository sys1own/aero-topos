import jax
# FORCE 64-BIT PRECISION PROFILE GLOBALLY BEFORE ALL MATRIX ALLOCATIONS
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from src.topos.sheaf import SimplicialSheaf
from src.topos.tensor_logic import GeometryOfInteractionSolver
from src.topos.infinity import UnivalentCompiler

def run_phase_iv_intelligence_ceiling():
    print("==============================================================================")
    print("[Aero Topos Phase IV] Evaluating Homotopical Intelligence Structural Limits")
    print("==============================================================================")
    
    # 1. Evaluate Simplicial Sheaf Complex Connections
    sheaf = SimplicialSheaf(num_vertices=10, num_edges=12)
    mock_edges = np.array([[0, 1], [1, 2], [2, 3]])
    involution = sheaf.construct_directed_edge_space(mock_edges)
    print(f"[+] Simplicial Complex Deployed. Involution Operator Matrix Shape: {involution.shape}")

    # 2. Higher-Order Univalence Invariant Verification Passage
    compiler = UnivalentCompiler(dimension=3)
    L1_basis = jnp.eye(3, dtype=jnp.complex128)
    # Introducing a unitarily phase-shifted equivalent target space algorithm
    L2_basis = jnp.eye(3, dtype=jnp.complex128) * jnp.exp(1j * jnp.pi / 4)
    
    eigenvalues = compiler.compute_souriau_invariant(L1_basis, L2_basis)
    print(f"[+] Univalent Check: Computational Equivalence Unitary Roots:\n    {np.round(eigenvalues, 4)}")

    # 3. Geometry of Interaction Wave Propagation & Gradient Extraction
    dim = 6
    solver = GeometryOfInteractionSolver(edge_dimension=dim)
    M = jnp.roll(jnp.eye(dim), 1, axis=1)
    U = jnp.eye(dim) * 0.15
    
    EX = solver.execute_goi_wave(M, U)
    mock_loss_grad = jnp.ones_like(EX) * 0.01
    dU = solver.compute_metamorphic_gradients(M, U, mock_loss_grad)
    
    print(f"\n[+] Metamorphic Inversion Wave Computation Complete.")
    print(f"    - Execution Operator Field Shape: {EX.shape}")
    print(f"    - Exact Analytical Metamorphic Gradient (dL/dU) Norm: {jnp.linalg.norm(dU):.6f}")
    print("==============================================================================")

if __name__ == "__main__":
    run_phase_iv_intelligence_ceiling()
