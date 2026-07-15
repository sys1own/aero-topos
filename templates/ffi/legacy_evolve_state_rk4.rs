pub fn ${name}_legacy(input: &[f64]) -> Vec<f64> {
    // Unpack: [dt, potential_len, potential..., state...]
    if input.len() < 3 {
        return vec![];
    }
    let dt = input[0];
    let pot_len = input[1] as usize;
    let potential = &input[2..2 + pot_len];
    let state = &input[2 + pot_len..];
    ${name}_impl(state, dt, potential)
}

fn ${name}_impl(state: &[f64], dt: f64, potential: &[f64]) -> Vec<f64> {
    let n = state.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut result = vec![0.0; n];
    for i in 0..n {
        k1[i] = -potential[i % potential.len()] * state[i];
    }
    for i in 0..n {
        k2[i] = -potential[i % potential.len()] * (state[i] + 0.5 * dt * k1[i]);
    }
    for i in 0..n {
        k3[i] = -potential[i % potential.len()] * (state[i] + 0.5 * dt * k2[i]);
    }
    for i in 0..n {
        k4[i] = -potential[i % potential.len()] * (state[i] + dt * k3[i]);
    }
    for i in 0..n {
        result[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    result
}
