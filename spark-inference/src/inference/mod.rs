use ndarray::Array2;

pub mod sam;
pub mod yolo;

pub(super) fn linear_interpolate(input: Array2<f32>, new_shape: (usize, usize)) -> Array2<f32> {
    let (old_height, old_width) = input.dim();
    let (new_height, new_width) = new_shape;
    let mut output = Array2::<f32>::zeros((new_height, new_width));

    for i in 0..new_height {
        for j in 0..new_width {
            // Mapping new coordinates to old coordinates
            let x = (j as f32) / (new_width as f32) * (old_width as f32 - 1.0);
            let y = (i as f32) / (new_height as f32) * (old_height as f32 - 1.0);

            let x0 = x.floor() as usize;
            let x1 = x.ceil() as usize;
            let y0 = y.floor() as usize;
            let y1 = y.ceil() as usize;

            let p00 = input[[y0, x0]];
            let p01 = input[[y0, x1]];
            let p10 = input[[y1, x0]];
            let p11 = input[[y1, x1]];

            // Interpolation weights
            let dx = x - x0 as f32;
            let dy = y - y0 as f32;

            // Bilinear interpolation formula
            let interpolated_value = p00 * (1.0 - dx) * (1.0 - dy)
                + p01 * dx * (1.0 - dy)
                + p10 * (1.0 - dx) * dy
                + p11 * dx * dy;

            output[[i, j]] = interpolated_value;
        }
    }

    output
}

pub(super) fn sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}
