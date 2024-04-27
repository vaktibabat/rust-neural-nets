use ndarray::Array2;

pub mod mnist;

pub struct Dataset {
    pub data: Array2<f64>,
    pub target: Array2<f64>,
}
