import unittest
import numpy as np
from src.topos.arena import TopologicalArena
from src.topos.tensor_logic import TensorLogicEngine
from src.topos.homeostasis import BiochemicalHomeostasis
from src.topos.compiler import ToposCompiler

class TestAeroToposCore(unittest.TestCase):

    def setUp(self):
        # Establish a clean, isolated testing space for each run
        self.entities = {"A": 0, "B": 1, "C": 2}
        self.arena = TopologicalArena(max_coordinate_slots=10)
        self.tensor_engine = TensorLogicEngine()
        self.repair_engine = BiochemicalHomeostasis(self.arena)
        self.compiler = ToposCompiler(self.entities)

    def test_tensor_contraction_deduction(self):
        """Validates that logical deduction executes accurately as a matrix contraction."""
        # Setup mock relational matrices: A->B and B->C
        tA = np.zeros((3, 3))
        tB = np.zeros((3, 3))
        tA[0, 1] = 1
        tB[1, 2] = 1

        # Execute multi-linear contraction (Einstein summation notation)
        result = self.tensor_engine.execute_deduction(tA, tB)
        
        # The intersection at (0, 2) must be positively deduced
        self.assertEqual(result[0, 2], 1)
        self.assertEqual(result[0, 1], 0)

    def test_biochemical_self_repair_homeostasis(self):
        """Verifies that the engine routes around corrupted hardware coordinates in-place."""
        # Register an active task weight at coordinates (4, 5)
        self.arena.register_node_pair(4, 5, 99)
        self.assertIn((4, 5), self.arena.active_nodes)

        # Simulate a sudden hardware sector corruption event at slot 4
        self.repair_engine.register_hardware_fault(4)

        # Homeostasis check: The corrupted coordinate must be cleared
        self.assertEqual(self.arena.arena[4, 5], 0)
        
        # The weight must be dynamically re-routed to the pre-allocated redundant coordinate (5, 5)
        self.assertEqual(self.arena.arena[5, 5], 99)

    def test_zero_allocation_matrix_bounds(self):
        """Confirms that registering node parameters stays strictly inside pre-allocated spaces."""
        self.arena.register_node_pair(0, 1, 5)
        # Verify slot assignment matches physical index exactly without dynamic expansion arrays
        self.assertEqual(self.arena.arena[0, 1], 5)
        
        # Out of bounds coordinates must be gracefully ignored to protect isolation boundaries
        self.arena.register_node_pair(99, 99, 5)
        self.assertNotIn((99, 99), self.arena.active_nodes)

    def test_homeostasis_fault_tolerance(self):
        """Verifies that compiler token conflicts and chain mismatches are handled via homeostasis."""
        # Initialize compiler with homeostasis
        self.compiler_with_homeostasis = ToposCompiler(self.entities, self.repair_engine)
        
        # Test 1: Token conflict (unknown entity "D")
        knowledge_base_with_conflict = [
            ("parent", "Grandma", "Mom"),
            ("parent", "Mom", "D")  # D is not in self.entities
        ]
        rule = "parent(X,Y) & parent(Y,Z) -> grandparent(X,Z)"
        
        # This must compile without raising exceptions
        tA, tB = self.compiler_with_homeostasis.lower_rule_to_tensors(rule, knowledge_base_with_conflict)
        self.assertIsNotNone(tA)
        self.assertIsNotNone(tB)
        
        # Test 2: Chain mismatch (parent(X,Y) & parent(W,Z))
        invalid_rule = "parent(X,Y) & parent(W,Z) -> grandparent(X,Z)"
        tA_invalid, tB_invalid = self.compiler_with_homeostasis.lower_rule_to_tensors(invalid_rule, [("parent", "Grandma", "Mom"), ("parent", "Mom", "Child")])
        self.assertIsNotNone(tA_invalid)
        self.assertIsNotNone(tB_invalid)

if __name__ == "__main__":
    unittest.main()

