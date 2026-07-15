/// AeroVM bytecode hook for `$fn_name` (node $index).
pub fn $hook(input: &[f64]) -> Result<Vec<f64>, String> {
    let mut handle = module().lock().map_err(|e| format!("lock poisoned: {e}"))?;
    handle.ensure_loaded();

    #[cfg(feature = "aero_vm_link")]
    { unsafe { dispatch_node($index, input) } }

    // Differential-verification stub: bit-identical to the preserved legacy
    // implementation. Production replaces this with `dispatch_node`.
    #[cfg(not(feature = "aero_vm_link"))]
    { $stub_expr }
}
