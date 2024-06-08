pub mod model;
pub mod parsing;

use clap::Parser;
use json::object;
use model::neural_net::{ActivationFunction, InitMethod};
use model::{neural_net, Model};
use ndarray::Axis;
use parsing::mnist;
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path of the training dataset
    #[arg(short, long)]
    train_path: String,

    /// The path of the validation dataset
    #[arg(short, long)]
    validation_path: String,

    /// Network structure, e.g. [784, 500, 300, 10]
    #[arg(short, long, value_parser, num_args = 2.., value_delimiter = ' ')]
    network_structure: Vec<usize>,

    /// Learning rate of the network
    #[arg(short, long, default_value_t = 0.01)]
    learning_rate: f64,

    /// Batch size of the network
    #[arg(short, long, default_value_t = 50)]
    batch_size: usize,

    /// Number of epochs to train the network for
    /// If this parameter is not provided, early stopping is used instead
    /// And you also need to specify a tolerance
    #[arg(short, long, default_value = None)]
    num_epochs: Option<usize>,

    /// Debug mode (save loss in a "time     loss" format)
    #[arg(short, long, default_value = None)]
    debug_path: Option<String>,

    /// Activation function used by the network
    #[arg(short, long, default_value = None)]
    activation_function: ActivationFunction,

    /// Weight initialization method
    #[arg(short, long, default_value = None)]
    initialization: InitMethod,

    /// Tolerance for early stopping
    #[arg(short, long, default_value_t = 0.0001)]
    epsilon: f64,
    
    /// Whether or not to export the model's weights
    /// Weights are exported in JSON format
    #[arg(short, long, default_value = None)]
    weight_path: Option<String>,
}

/// Test the model on the validation set
pub fn test_model(path: &str, model: &neural_net::NeuralNet) {
    let dataset = mnist::parse_dataset(path);
    let predictions = model.predict(&dataset.data.view());

    let mut num_mistakes = 0;

    for (prediction, target_row) in predictions
        .axis_iter(Axis(0))
        .zip(dataset.target.axis_iter(Axis(0)))
    {
        let predict_digit = prediction
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;
        let actual_digit = target_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;

        if predict_digit != actual_digit {
            num_mistakes += 1;
        }
    }

    println!("The number of mistakes is {}", num_mistakes);
}

/// Write the losses to a debug file
fn write_losses(debug_path: &str, losses: Vec<(usize, f64)>) -> std::io::Result<()> {
    let mut file = File::create(debug_path)?;

    for (x, y) in losses {
        file.write_all(format!("{}    {}\n", x, y).as_bytes())?;
    }

    Ok(())
}

/// Write the weights of the model in JSON formats
/// The keys are e.g. W0, b0, W1, b1. The values are provided in an array of the weights
fn write_weights(weight_path: &str, model: &neural_net::NeuralNet) -> std::io::Result<()> {
    let mut data = object! {};
    let mut file = File::create(weight_path)?;

    for (i, weight) in model.layers.iter().enumerate() {
        let w: Vec<f64> = weight.0.iter().map(|x| *x).collect();
        let b: Vec<f64> = weight.1.iter().map(|x| *x).collect();
        let w_key = format!("W{}", i);
        let b_key = format!("b{}", i);

        data[w_key] = w.into();
        data[b_key] = b.into();
    }

    file.write_all(data.dump().as_bytes())?;

    Ok(())
}

fn main() {
    let args = Args::parse();

    let dataset = mnist::parse_dataset(&args.train_path);
    let mut neural_net = neural_net::NeuralNet::new(
        args.network_structure,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.activation_function,
        args.initialization,
        args.epsilon,
    );

    let losses = neural_net.fit(&dataset, &args.validation_path);

    if let Some(debug_path) = args.debug_path {
        let _ = write_losses(&debug_path, losses);
    }

    if let Some(weight_path) = args.weight_path {
        let _ = write_weights(&weight_path, &neural_net);
    }

    test_model(&args.validation_path, &neural_net);
}
