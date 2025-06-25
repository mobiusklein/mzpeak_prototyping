use std::{
    io, path, time
};

use mzdata::prelude::SpectrumLike;
use mzpeak_prototyping::MzPeakReader;
use clap::Parser;
use env_logger;

#[derive(Parser)]
struct App {
    #[arg()]
    filename: path::PathBuf
}

fn main() -> io::Result<()> {
    env_logger::init();
    let args = App::parse();
    let start = time::Instant::now();
    let reader = MzPeakReader::new(args.filename)?;
    log::info!("Opened in {:0.2} seconds", start.elapsed().as_secs_f64());
    let mut i = 0;
    let mut points = 0;
    for spec in reader {
        if i % 100 == 0 {
            log::info!("Read spectrum {i}");
        }
        i += 1;
        let arrays = spec.raw_arrays().unwrap();
        match arrays.mzs() {
            Ok(arr) => {
                points += arr.len();
            }
            Err(e) => {
                eprintln!("Failed to retrieve arrays for spectrum {}: {e}", spec.index());
            }
        }
    }
    let dur = start.elapsed();
    eprintln!("Read {i} spectra and {points} points. Elapsed: {} seconds", dur.as_secs_f64());
    Ok(())
}