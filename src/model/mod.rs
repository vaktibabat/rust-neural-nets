use ndarray::{Array2, ArrayView2};

use crate::parsing::Dataset;

pub mod neural_net;

pub trait Model {
    fn fit(&mut self, dataset: &Dataset) -> Vec<(usize, f64)>;
    fn predict(&self, instance: &ArrayView2<f64>) -> Array2<f64>;
}
