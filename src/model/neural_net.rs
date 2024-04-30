use crate::parsing::{mnist::parse_dataset, Dataset};
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::distributions::{Distribution, Uniform};

use super::Model;

/// Represents a neural net
pub struct NeuralNet {
    pub layers: Vec<(Array2<f64>, Array1<f64>)>, // Each layer holds a weight matrix and a bias vector
    pub num_epochs: usize,                       // Training hyperparams
    pub batch_size: usize,
    pub learning_rate: f64,
    pub activation_function: ActivationFunction,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    LeakyReLU,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum InitMethod {
    Default,
    Xavier,
}

impl NeuralNet {    
    /// Construct a new neural net according to the specified hyperparams
    pub fn new(
        layer_structure: Vec<usize>,
        num_epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        activation_function: ActivationFunction,
        init_method: InitMethod,
    ) -> NeuralNet {
        let layers = match init_method {
            InitMethod::Default => init_layers_default(&layer_structure),
            InitMethod::Xavier => init_layers_xavier(&layer_structure)
        };

        NeuralNet {
            layers,
            num_epochs,
            batch_size,
            learning_rate,
            activation_function,
        }
    }

    // Perform a forward pass of the network on some input.
    // Returns the outputs of the hidden layers, and the non-activated outputs of the hidden layers (used for backprop)
    fn forward(&self, inputs: &ArrayView2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut hidden = vec![];
        let mut hidden_linear = vec![];
        // The first layer is a passthrough layer, so it outputs whatever its input is
        hidden.push(inputs.to_owned());

        // We iterate for every layer
        let mut it = self.layers.iter().peekable();

        // Iterate over the layers
        while let Some(layer) = it.next() {
            // The output of the layer without applying the activation function
            let lin_output = hidden.last().unwrap().dot(&layer.0) + &layer.1;
            // The real output of the layer - If the layer is a hidden layer, we apply the activation function
            // and otherwise (this is the output layer) the output is the same as the linear output
            let real_output = lin_output.map(|x| match it.peek() {
                Some(_) => activation(&self.activation_function, *x),
                None => *x,
            });

            hidden.push(real_output);
            hidden_linear.push(lin_output);
        }

        (hidden, hidden_linear)
    }

    /// Calculate the gradients using backprop and perform a GD step
    fn backward_and_update(
        &mut self,
        hidden: Vec<Array2<f64>>,
        hidden_linear: Vec<Array2<f64>>,
        grad: Array2<f64>,
    ) {
        // The gradient WRT the current layer
        let mut grad_help = grad;

        for idx in (0..self.layers.len()).rev() {
            // If we aren't at the last layer, we need to change the gradient
            if idx != self.layers.len() - 1 {
                let step_mat = hidden_linear[idx].map(|x| delta_activation(&self.activation_function, *x));
                grad_help = grad_help * step_mat;
            }

            // Gradient WRT the weights in the current layer
            let weight_grad = hidden[idx].t().dot(&grad_help);
            // Gradient WRT the biases in the current layer
            let bias_grad = &grad_help.mean_axis(Axis(0)).unwrap();

            // Perform GD step
            let new_weights = &self.layers[idx].0 - self.learning_rate * weight_grad;
            let new_biases = &self.layers[idx].1 - self.learning_rate * bias_grad;

            // Update the helper variable
            grad_help = grad_help.dot(&self.layers[idx].0.t());

            self.layers[idx] = (new_weights, new_biases);
        }
    }
}

impl Model for NeuralNet {
    /// Fit the model to the dataset
    /// Return the model loss as a function of time (used for plotting)
    fn fit(&mut self, dataset: &Dataset, test_path: Option<&str>) -> Vec<(usize, f64)> {
        // Used for writing the debug output
        let mut losses = vec![];

        for num_epoch in 0..self.num_epochs {
            // Get a batch of instances and their targets
            for (input_batch, target_batch) in dataset
                .data
                .axis_chunks_iter(Axis(0), self.batch_size)
                .zip(dataset.target.axis_chunks_iter(Axis(0), self.batch_size))
            {
                let (hidden, hidden_linear) = self.forward(&input_batch);

                let scores = hidden.last().unwrap();
                let mut predictions = Array::zeros((0, scores.ncols()));

                // Construct softmax matrix
                for row in scores.axis_iter(Axis(0)) {
                    predictions.push_row(softmax(row).view()).unwrap();
                }

                // Gradient is initialized to the gradient of the loss WRT the output layer
                let grad = predictions - target_batch;

                self.backward_and_update(hidden, hidden_linear, grad);
            }

            if let Some(path) = test_path {
                let loss = test_loss(&self, path);

                losses.push((num_epoch, loss));
            }
        }

        losses
    }

    /// Predict the probabities for a set of instances - each instance is a row in "inputs"
    fn predict(&self, inputs: &ArrayView2<f64>) -> Array2<f64> {
        let (hidden, _) = self.forward(inputs);
        let scores = hidden.last().unwrap();
        // Construct the softmax
        let mut predictions = Array::zeros((0, scores.ncols()));

        for row in scores.axis_iter(Axis(0)) {
            predictions.push_row(softmax(row).view()).unwrap();
        }

        predictions
    }
}

fn activation(name: &ActivationFunction, z: f64) -> f64 {
    match name {
        ActivationFunction::ReLU => z.max(0f64),
        ActivationFunction::Sigmoid => (1f64 + (-z).exp()).recip(),
        ActivationFunction::Tanh => (z.exp() - (-z).exp()) / (z.exp() + (-z).exp()),
        ActivationFunction::Linear => 3f64*z + 1f64,
        ActivationFunction::LeakyReLU => z.max(0.01 * z),
    }
}

fn delta_activation(name: &ActivationFunction, z: f64) -> f64 {
    match name {
        ActivationFunction::ReLU => if z > 0f64 {1f64} else {0f64},
        ActivationFunction::Sigmoid => activation(name, z) * (1f64 - activation(name, z)),
        ActivationFunction::Tanh => 1f64 - activation(name, z) * activation(name, z),
        ActivationFunction::Linear => 3f64,
        ActivationFunction::LeakyReLU => if z > 0f64 {1f64} else {0.01f64},
    }
}

fn init_layers_default(layer_structure: &Vec<usize>) -> Vec<(Array2<f64>, Array1<f64>)> {
    let mut layers = vec![];
    let mut rng = rand::thread_rng();
    // Weights are initialized from a uniform distribiution
    let distribution = Uniform::new(-0.3, 0.3);

    for i in 0..layer_structure.len() - 1 {
        // Random matrix of the weights between this layer and the next layer
        let weights = Array::zeros((layer_structure[i], layer_structure[i + 1]))
            .map(|_: &f64| distribution.sample(&mut rng));
        // Bias vector between this layer and the next layer. Init'd to ondes
        let bias = Array::ones(layer_structure[i + 1]);

        layers.push((weights, bias));
    }

    layers
}

fn init_layers_xavier(layer_structure: &Vec<usize>) -> Vec<(Array2<f64>, Array1<f64>)> {
    let mut layers = vec![];
    let mut rng = rand::thread_rng();

    for i in  0..layer_structure.len() - 1 {
        let boundary = 6f64.sqrt() / (layer_structure[i] + layer_structure[i + 1]) as f64;
        let dist = Uniform::new(-boundary, boundary);
        
        let weights = Array::zeros((layer_structure[i], layer_structure[i + 1]))
        .map(|_: &f64| dist.sample(&mut rng));
        let bias = Array::zeros(layer_structure[i + 1]);

        layers.push((weights, bias));

    }

    layers
}

/// Softmax function - Convert scores into a probability distribution
fn softmax(scores: ArrayView1<f64>) -> Array1<f64> {
    let max = scores.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
    // We use a numerical trick where we shift the elements by the max, because otherwise
    // We would have to compute the exp of very large values which wraps to NaN
    let shift_scores = scores.map(|x| x - max);
    let sum: f64 = shift_scores.iter().map(|x| x.exp()).sum();

    (0..scores.len())
        .map(|x| shift_scores[x].exp() / sum)
        .collect()
}

/// Calculate the cross-entropy loss on a given batch
fn cross_entropy(predictions: &Array2<f64>, target: ArrayView2<f64>) -> f64 {
    let total: f64 = predictions
        .axis_iter(Axis(0))
        .zip(target.axis_iter(Axis(0)))
        .map(|(actual_row, target_row)| target_row.dot(&actual_row.map(|x| x.log2())))
        .sum();

    -1f64 * (1f64 / predictions.nrows() as f64) * total
}

fn test_loss(model: &NeuralNet, test_path: &str) -> f64 {
    let test_dataset = parse_dataset(test_path);
    let predictions = model.predict(&test_dataset.data.view());
    
    let target = test_dataset.target;

    cross_entropy(&predictions, target.view())
}