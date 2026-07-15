pub fn ${name}_legacy(input: &[f64]) -> Vec<f64> {
    // Unpack: [twist, nrows, ncols, flat_lattice...]
    if input.len() < 3 {
        return vec![];
    }
    let twist = input[0];
    let nrows = input[1] as usize;
    let ncols = input[2] as usize;
    let flat = &input[3..];
    let lattice: Vec<Vec<f64>> = flat.chunks(ncols).take(nrows).map(|c| c.to_vec()).collect();
    let result = ${name}_impl(&lattice, twist);
    vec![result]
}

fn ${name}_impl(lattice: &[Vec<f64>], twist: f64) -> f64 {
    let mut invariant = 0.0;
    for row in lattice.iter() {
        for val in row.iter() {
            invariant += val.sin() * twist.cos();
            invariant *= 1.0 + val.abs() * 0.001;
        }
    }
    invariant
}
