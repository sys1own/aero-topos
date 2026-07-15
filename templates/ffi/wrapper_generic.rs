pub fn $name(input: &[f64]) -> Vec<f64> {
    // Hot path delegated to Aero bytecode via FFI node hook.
    aero_ffi::$hook(&input)
        .expect("AeroVM invocation failed")
}
