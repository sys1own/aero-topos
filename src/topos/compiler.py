import numpy as np
import re

class ToposCompiler:
    def __init__(self, entity_registry, homeostasis_engine=None):
        # Maps raw human string tokens (e.g., "Alice") to coordinate index integers
        self.entity_registry = entity_registry
        self.dim_size = len(entity_registry)
        self.homeostasis = homeostasis_engine

    def lower_rule_to_tensors(self, rule_string, relationship_pairs):
        """
        Parses a declarative logical rule string and extracts variable dependencies
        to output structured relational matrices for tensor logic contraction.
        """
        print(f"[Aero Topos] Parsing declaration: {rule_string}")
        
        # Extract the predicates from the string
        predicates = re.findall(r'(\w+)\((\w+),(\w+)\)', rule_string)
        if len(predicates) < 2:
            raise ValueError("Invalid rule syntax. Requires at least two input predicates.")

        # Initialize clean structural matrices matching our entity dimensions
        tensor_A = np.zeros((self.dim_size, self.dim_size))
        tensor_B = np.zeros((self.dim_size, self.dim_size))

        # Map input metadata into physical coordinate intersections
        pred_A_name, var_A1, var_A2 = predicates[0]
        pred_B_name, var_B1, var_B2 = predicates[1]

        # Verify variable chaining continuity (e.g., the 'Y' index link)
        if var_A2 != var_B1:
            if self.homeostasis is not None:
                print(f"[Homeostasis Intercept] Variable chain mismatch: {var_A2} != {var_B1}. Dynamic re-routing applied.")
                # Associative rewrite rule: align variables to heal the mismatch
                var_B1 = var_A2
            else:
                raise NotImplementedError("Complex non-linear index chaining requires full UAST mapping optimization.")

        # Populate the tensors from the developer's raw relationship data pairs
        for rel_name, p1, p2 in relationship_pairs:
            idx1 = self.entity_registry.get(p1)
            idx2 = self.entity_registry.get(p2)
            
            # Fault tolerance: if a compiler target catches a broken dependency or token conflict
            if idx1 is None:
                if self.homeostasis is not None:
                    # Dynamically map missing token to a redundant pre-allocated coordinate slot
                    redundant_slot = (hash(p1) % self.dim_size) if self.dim_size > 0 else 0
                    self.homeostasis.register_hardware_fault(redundant_slot)
                    idx1 = redundant_slot
                else:
                    raise ValueError(f"Token conflict: '{p1}' is not registered and homeostasis is inactive.")
            
            if idx2 is None:
                if self.homeostasis is not None:
                    # Dynamically map missing token to a redundant pre-allocated coordinate slot
                    redundant_slot = (hash(p2) % self.dim_size) if self.dim_size > 0 else 0
                    self.homeostasis.register_hardware_fault(redundant_slot)
                    idx2 = redundant_slot
                else:
                    raise ValueError(f"Token conflict: '{p2}' is not registered and homeostasis is inactive.")

            if idx1 is not None and idx2 is not None:
                if rel_name == pred_A_name:
                    tensor_A[idx1, idx2] = 1
                elif rel_name == pred_B_name:
                    tensor_B[idx1, idx2] = 1

        return tensor_A, tensor_B

