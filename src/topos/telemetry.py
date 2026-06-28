import time
import numpy as np

class ExogenousTelemetryEngine:
    def __init__(self):
        pass

    def capture_hardware_metrics(self, evaluation_job):
        """
        Wraps an evaluation job, captures execution metrics, and returns status.
        Returns dict with execution_latency_us, hardware_entropy_sig, and status.
        """
        start_time = time.perf_counter_ns()
        try:
            result = evaluation_job()
            end_time = time.perf_counter_ns()

            # Compute metrics expected by the evolutionary optimizer
            execution_latency_us = (end_time - start_time) // 1000

            # Compute hardware entropy signature from hash of result shape/data
            result_array = np.asarray(result) if result is not None else np.array([0.0])
            hardware_entropy_sig = float(abs(hash(result_array.tobytes()[:64] if result_array.nbytes >= 64 else result_array.tobytes())) % (2**31))

            return {
                "execution_latency_us": execution_latency_us,
                "hardware_entropy_sig": hardware_entropy_sig,
                "status": "PASSED"
            }
        except Exception as e:
            end_time = time.perf_counter_ns()
            return {
                "execution_latency_us": (end_time - start_time) // 1000,
                "hardware_entropy_sig": 0.0,
                "status": "FAILED",
                "error": str(e)
            }

    def execute_monitored_solver(self, solver_callback, *args, **kwargs):
        start_time = time.perf_counter_ns()
        EX, rho = solver_callback(*args, **kwargs)
        end_time = time.perf_counter_ns()
        
        latency_us = (end_time - start_time) // 1000
        # Calculate regularization adjustments based on spectral properties
        regularization_adjustment = 0.0 if rho < 1.0 else (rho - 1.0) + 1e-4
        
        return EX, {
            "latency_us": latency_us,
            "spectral_radius": float(rho),
            "dynamic_regularization_scaling": regularization_adjustment
        }
