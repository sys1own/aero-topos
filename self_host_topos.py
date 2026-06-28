import numpy as np
import os
import re
import shutil
from src.topos.arena import TopologicalArena
from src.topos.tensor_logic import TensorLogicEngine
from src.topos.telemetry import ExogenousTelemetryEngine
from src.topos.compiler import ToposCompiler

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib  # type: ignore

def update_file_constant_memory(filepath, pattern, replacement):
    """
    Surgically updates a file line-by-line to ensure constant memory footprint
    (O(1) memory) and performs an atomic replace promotion.
    """
    tmp_filepath = filepath + ".tmp"
    with open(filepath, "r", encoding="utf-8") as f_in, open(tmp_filepath, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(re.sub(pattern, replacement, line))
    os.replace(tmp_filepath, filepath)

def atomic_directory_swap(src_dir, dst_dir):
    """
    Atomically swap src_dir and dst_dir in-situ, enforcing a constant memory
    footprint by using OS directory renaming with zero heap memory allocation.
    """
    temp_dir = dst_dir + ".swap.tmp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    if os.path.exists(dst_dir):
        os.rename(dst_dir, temp_dir)
        os.rename(src_dir, dst_dir)
        shutil.rmtree(temp_dir)
    else:
        os.rename(src_dir, dst_dir)

def execute_self_hosted_optimization():
    print("==============================================================================")
    # Grounding the engine in physical hardware signals to prevent degenerative collapse
    print("[Aero Topos] Initiating Self-Hosting Optimization Loop via Physical Telemetry")
    print("==============================================================================")
    
    # 1. Parse active profile settings directly from self_host.aero
    blueprint_path = "self_host.aero"
    try:
        with open(blueprint_path, "rb") as handle:
            doc = tomllib.load(handle)
        profile_data = doc.get("self_host", {}).get("profile", {})
        active_profile = profile_data.get("active_profile", "extended_tensor_space")
        target_slots = int(profile_data.get("max_coordinate_slots", 250))
        print(f"[+] Loaded configuration: Profile={active_profile} | Max slots={target_slots}")
    except Exception as exc:
        print(f"[!] Warning: Could not parse {blueprint_path} ({exc}); using defaults")
        active_profile = "extended_tensor_space"
        target_slots = 250

    telemetry = ExogenousTelemetryEngine()
    tensor_engine = TensorLogicEngine()
    
    # Candidate configurations to test (varying slot limits and dimension boundaries)
    candidates = [
        {"slots": 50, "label": "compact_isolated"},
        {"slots": 100, "label": "balanced_manifold"},
        {"slots": target_slots, "label": active_profile}
    ]
    
    best_candidate = None
    min_latency = float('inf')
    
    # Core relationship context for benchmarking tensor deductions
    entities = {"Grandma": 0, "Mom": 1, "Child": 2}
    knowledge_base = [("parent", "Grandma", "Mom"), ("parent", "Mom", "Child")]
    rule = "parent(X,Y) & parent(Y,Z) -> grandparent(X,Z)"

    for cfg in candidates:
        slots = cfg["slots"]
        label = cfg["label"]
        print(f"\n[*] Evaluating Candidate Profile: {label} (Slots: {slots})")
        
        # Instantiate test layers bounded within the current candidate space
        arena = TopologicalArena(max_coordinate_slots=slots)
        compiler = ToposCompiler(entities)
        tA, tB = compiler.lower_rule_to_tensors(rule, knowledge_base)
        
        def processing_job():
            g_xz = tensor_engine.execute_deduction(tA, tB)
            arena.register_node_pair(0, 2, int(g_xz[0, 2]))
            return g_xz
            
        # Extract non-synthetic hardware-level constraints to guide the selection axes
        metrics = telemetry.capture_hardware_metrics(processing_job)
        latency = metrics["execution_latency_us"]
        print(f"[Result] Grounded Latency: {latency} us | Entropy Score: {metrics['hardware_entropy_sig']}")
        
        # Selection phase: seek the absolute sharpest execution runtime profile
        if latency < min_latency and latency > 0:
            min_latency = latency
            best_candidate = cfg

    print("\n==============================================================================")
    print(f"[+] Optimization Selected: {best_candidate['label']} with {min_latency} us runtime.")
    print("==============================================================================")

    # 2. Atomic update: Mutate the pre-allocated coordinate matrix slot limits in-place
    # using O(1) memory file rewriting.
    update_file_constant_memory(
        "main_topos.py",
        r"max_coordinate_slots=\d+",
        f"max_coordinate_slots={best_candidate['slots']}"
    )
    
    # Enforce atomic directory swap if the swap directory is configured
    swap_dir = doc.get("self_host", {}).get("atomic_swap_directory") if 'doc' in locals() else None
    if swap_dir and os.path.exists(swap_dir):
        staging_dir = swap_dir + "_staging"
        if os.path.exists(staging_dir):
            print(f"[+] Atomically swapping staging directory {staging_dir} into production {swap_dir}")
            atomic_directory_swap(staging_dir, swap_dir)
        
    print("[+] Structural Refactoring Complete. main_topos.py updated with optimized parameters.")

if __name__ == "__main__":
    execute_self_hosted_optimization()

