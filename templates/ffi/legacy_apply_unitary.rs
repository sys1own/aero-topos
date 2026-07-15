pub fn ${name}_legacy(input: &[f64]) -> Vec<f64> {
    // Unpack: [dim, coupling, state...]
    if input.len() < 3 {
        return vec![];
    }
    let dim = input[0] as usize;
    let coupling = input[1];
    let state = &input[2..];
    ${name}_impl(state, dim, coupling)
}

fn ${name}_impl(state: &[f64], dim: usize, coupling: f64) -> Vec<f64> {
    let n = state.len();
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            let phase = ((i * j) as f64 * coupling).cos();
            let amplitude = ((i as f64 + j as f64) / dim as f64).sin();
            result[i] += state[j] * phase * amplitude;
        }
        let row_norm: f64 = (0..n)
            .map(|k| ((i * k) as f64 * coupling).cos().powi(2))
            .sum::<f64>()
            .sqrt();
        if row_norm > 1e-15 {
            result[i] /= row_norm;
        }
    }
    result
}
