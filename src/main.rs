pub mod model;
pub mod parsing;

use clap::Parser;
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
    #[arg(short, long, default_value_t = 11)]
    num_epochs: usize,

    /// Debug mode (save loss in a "time     loss" format)
    #[arg(short, long, default_value = None)]
    debug_path: Option<String>,
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

fn main() {
    let args = Args::parse();

    let dataset = mnist::parse_dataset(&args.train_path);
    let mut neural_net = neural_net::NeuralNet::new(
        args.network_structure,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
    );

    let losses = neural_net.fit(&dataset);

    if let Some(debug_path) = args.debug_path {
        let _ = write_losses(&debug_path, losses);
    }

    test_model(&args.validation_path, &neural_net);
}
