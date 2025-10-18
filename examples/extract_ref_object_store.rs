use clap::Parser;
use futures::StreamExt;
use mzdata::io::AsyncRandomAccessSpectrumIterator;
use mzdata::mzpeaks::coordinate::{CoordinateRange, SimpleInterval, Span1D};
use mzdata::prelude::*;
use std::{io, sync::Arc};

use object_store::ObjectStore;

#[derive(clap::Parser)]
struct App {
    #[arg()]
    filename: String,

    #[arg(short, long, default_value = "10.0-21.0")]
    time_range: CoordinateRange<f32>,

    #[arg(short, long, default_value = "623.0-625.0")]
    mz_range: CoordinateRange<f64>,

    #[arg(short, long, default_value = "0.8-1.2")]
    im_range: CoordinateRange<f64>,

    #[arg(short='l', long)]
    ms_level_range: Option<CoordinateRange<u8>>
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> io::Result<()> {
    env_logger::init();
    let args = App::parse();
    let start = std::time::Instant::now();
    let (store, path) = object_store::parse_url(&args.filename.parse().unwrap()).unwrap();
    let store = Arc::new(store);
    let meta = store.head(&path).await?;

    let handle = object_store::buffered::BufReader::new(store, &meta);

    let mut reader = mzdata::io::AsyncMZReader::open_read_seek(handle).await?;
    eprintln!(
        "Opening reader took {} seconds",
        start.elapsed().as_secs_f64()
    );

    let time_range =
        SimpleInterval::new(args.time_range.start.unwrap(), args.time_range.end.unwrap());
    let mz_range = SimpleInterval::new(args.mz_range.start.unwrap(), args.mz_range.end.unwrap());
    let im_range = SimpleInterval::new(args.im_range.start.unwrap(), args.im_range.end.unwrap());
    let ms_level_range = args.ms_level_range.map(|r| {
        SimpleInterval::new(
            r.start.unwrap_or_default() as u8,
            r.end.map(|v| v as u8).unwrap_or(u8::MAX),
        )
    }).unwrap_or(SimpleInterval::new(0, u8::MAX));

    reader.start_from_time(time_range.start as f64).await?;
    let mut it = reader.as_stream();
    let mut k = 0;
    while let Some(spec) = it.next().await {
        k += 1;
        if !ms_level_range.contains(&spec.ms_level()) {
            if spec.start_time() > time_range.end && !spec.start_time().is_close(&time_range.end) {
                break;
            }
            continue;
        }
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
            break;
        }
    }
    eprintln!(
        "{} seconds elapsed, read {k} spectra",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
