use clap::{Parser, Subcommand};
use std::io;

// Import functionality from examples
mod examples {
    pub mod convert {
        include!("../examples/convert.rs");
    }
    pub mod benchmark {
        include!("../examples/benchmark.rs");
    }
}
use examples::convert::{ConvertCli, run_convert as convert_run_convert};
use examples::benchmark::{BenchmarkArgs, run_benchmark as benchmark_run_benchmark};

#[derive(Parser)]
#[command(name = "mzpeak_prototyping")]
#[command(about = "A tool for converting and benchmarking mass spectrometry data")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert a single mass spectrometry file to mzpeak format
    Convert(ConvertCli),
    /// Benchmark conversion of all supported files in a directory
    Benchmark(BenchmarkArgs),
}


fn main() -> io::Result<()> {
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Convert(cli_args) => convert_run_convert(&cli_args.filename, cli_args.convert_args),
        Commands::Benchmark(args) => benchmark_run_benchmark(args),
    }
}

