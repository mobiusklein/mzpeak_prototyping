use arrow::{
    array::{AsArray, Float32Array, Float64Array, UInt64Array},
};

use clap::Parser;
use mzdata::mzpeaks::coordinate::{CoordinateRange, SimpleInterval, Span1D};
use std::io;

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
}

fn main() -> io::Result<()> {
    env_logger::init();
    let args = App::parse();
    let start = std::time::Instant::now();

    let mut reader = mzpeak_prototyping::reader::MzPeakReader::new(&args.filename)?;
    // reader.load_all_spectrum_metadata()?;

    eprintln!("Opening archive took {} seconds", start.elapsed().as_secs_f64());

    let has_ion_mobility = reader.metadata.spectrum_array_indices.has_ion_mobility();

    let time_range = SimpleInterval::new(
        args.time_range.start.unwrap_or(0.0) as f32,
        args.time_range.end.unwrap_or(f64::INFINITY) as f32,
    );

    let mz_range = SimpleInterval::new(
        args.mz_range.start.unwrap_or(0.0),
        args.mz_range.end.unwrap_or(f64::INFINITY),
    );

    let im_range = SimpleInterval::new(
        args.im_range.start.unwrap_or(0.0),
        args.im_range.end.unwrap_or(f64::INFINITY),
    );

    let (it, time_index) = reader.extract_peaks(
        time_range,
        Some(mz_range),
        None
    )?;

    let query_range_end = std::time::Instant::now();
    eprintln!("{} seconds elapsed reading indices", (query_range_end - start).as_secs_f64());

    for batch in it {
        let batch = batch.unwrap();
        let root = batch.column(0).as_struct();
        let indices: &UInt64Array = root.column(0).as_any().downcast_ref().unwrap();
        let intensities: &Float32Array = root.column(2).as_any().downcast_ref().unwrap();

        macro_rules! iter {
            ($mzs:expr, $ims:expr, $mz_range:expr, $im_range:expr) => {
                let it = indices.iter().flatten().zip(
                    $mzs.iter().flatten()
                ).zip(
                    intensities.iter().flatten()
                );

                let mut last_index = 0;
                let mut last_time = 0.0;

                if $ims.is_some() {
                    for (((index, mz), intensity), im) in it.zip($ims.unwrap().iter().flatten()) {
                        if $im_range.contains(&im) {
                            if last_index != index {
                                last_time = time_index[&index];
                                last_index = index;
                            }
                            println!("{index}\t{last_time}\t{mz}\t{intensity}\t{im}");
                        }
                    }
                } else {
                    for ((index, mz), intensity) in it {
                        {
                            if last_index != index {
                                last_time = time_index[&index];
                                last_index = index;
                            }
                            println!("{index}\t{last_time}\t{mz}\t{intensity}");
                        }
                    }
                }

                // if started && !time_index.contains_key(&indices.values().last().unwrap()) {
                //     break;
                // }
            };
        }

        if has_ion_mobility {
            if let Some(mzs) = root.column(1).as_any().downcast_ref::<Float64Array>() {
                if let Some(ims) = root.column(3).as_any().downcast_ref::<Float64Array>() {
                    iter!(mzs, Some(ims), SimpleInterval::new(mz_range.start as f64, mz_range.end as f64), SimpleInterval::new(im_range.start as f64, im_range.end as f64));
                } else if let Some(ims) = root.column(3).as_any().downcast_ref::<Float32Array>() {
                    iter!(mzs, Some(ims), SimpleInterval::new(mz_range.start as f64, mz_range.end as f64), SimpleInterval::new(im_range.start as f32, im_range.end as f32));
                } else {
                    iter!(mzs, Option::<Float64Array>::None, mz_range, im_range);
                }
            } else if let Some(mzs) = root.column(1).as_any().downcast_ref::<Float32Array>() {
                if let Some(ims) = root.column(3).as_any().downcast_ref::<Float64Array>() {
                    iter!(mzs, Some(ims), SimpleInterval::new(mz_range.start as f32, mz_range.end as f32), SimpleInterval::new(im_range.start as f64, im_range.end as f64));
                } else if let Some(ims) = root.column(3).as_any().downcast_ref::<Float32Array>() {
                    iter!(mzs, Some(ims), SimpleInterval::new(mz_range.start as f32, mz_range.end as f32), SimpleInterval::new(im_range.start as f32, im_range.end as f32));
                } else {
                    iter!(mzs, Option::<Float64Array>::None, SimpleInterval::new(mz_range.start as f32, mz_range.end as f32), im_range);
                }
            } else {
                unimplemented!()
            }
        } else {
            if let Some(mzs) = root.column(1).as_any().downcast_ref::<Float64Array>() {
                iter!(mzs, Option::<Float64Array>::None, SimpleInterval::new(mz_range.start as f64, mz_range.end as f64), im_range);
            } else if let Some(mzs) = root.column(1).as_any().downcast_ref::<Float32Array>() {
                iter!(mzs, Option::<Float64Array>::None, SimpleInterval::new(mz_range.start as f32, mz_range.end as f32), im_range);
            } else {
                unimplemented!()
            }
        }
    }
    let end = std::time::Instant::now();
    eprintln!("{} seconds elapsed", (end - query_range_end).as_secs_f64());
    Ok(())
}
