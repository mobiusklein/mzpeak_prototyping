use std::io;

use clap::Parser;
use mzdata::prelude::*;
use mzpeaks::{coordinate::SimpleInterval, CoordinateRange};

#[derive(clap::Parser)]
struct App {
    #[arg()]
    filename: String,

    #[arg(short, long, default_value="10.0-21.0")]
    time_range: CoordinateRange<f32>,

    #[arg(short, long, default_value="623.0-625.0")]
    mz_range: CoordinateRange<f64>,

    #[arg(short, long, default_value="0.8-1.2")]
    im_range: CoordinateRange<f64>,

}

fn main() -> io::Result<()> {
    let args = App::parse();
    let mut reader = mzdata::MZReader::open_path(
        args.filename
    )?;

    let start = std::time::Instant::now();

    let time_range = SimpleInterval::new(args.time_range.start.unwrap(), args.time_range.end.unwrap());
    let mz_range = SimpleInterval::new(args.mz_range.start.unwrap(), args.mz_range.end.unwrap());
    let im_range = SimpleInterval::new(args.im_range.start.unwrap(), args.im_range.end.unwrap());

    let it = reader.start_from_time(time_range.start as f64)?;
    while let Some(spec) = it.next() {
        if let Some(arrays) = spec.arrays.as_ref() {
            let mzs = arrays.mzs()?;
            let ints = arrays.intensities()?;
            let time = spec.start_time();
            let index = spec.index();
            if let Ok((ims, _)) = arrays.ion_mobility() {
                for (mz, (int, im)) in mzs.iter().zip(ints.iter().zip(ims.iter())) {
                    if mz_range.contains(mz) && im_range.contains(im) {
                        println!("{index}\t{time}\t{mz}\t{int}\t{im}");
                    }
                }
            } else {
                for (mz, int) in mzs.iter().zip(ints.iter()) {
                    if mz_range.contains(mz) {
                        println!("{index}\t{time}\t{mz}\t{int}");
                    }
                }
            }

        }
        if spec.start_time() > time_range.end {
            break
        }
    }
    let end = std::time::Instant::now();
    eprintln!("{} seconds elapsed", (end - start).as_secs_f64());
    Ok(())
}