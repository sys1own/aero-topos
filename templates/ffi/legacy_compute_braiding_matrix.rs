pub fn ${name}_legacy(input: &[f64]) -> Vec<f64> {
    // Unpack: [dim, charge]
    if input.len() < 2 {
        return vec![];
    }
    let dim = input[0] as usize;
    let charge = input[1];
    let matrix = ${name}_impl(dim, charge);
    matrix.into_iter().flatten().collect()
}

fn ${name}_impl(dim: usize, charge: f64) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut val = 0.0;
            for k in 0..dim {
                val += (k as f64 * charge).sin()
                    * ((i + k) as f64 * 0.1).cos()
                    * ((j + k) as f64 * 0.1).exp().min(1e6);
            }
            matrix[i][j] = val / (dim as f64);
        }
    }
    matrix
}
