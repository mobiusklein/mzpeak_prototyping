use arrow::{
    array::{AsArray, PrimitiveArray},
    datatypes::{Float32Type, Float64Type, UInt64Type},
};

use clap::Parser;
use mzdata::mzpeaks::coordinate::{SimpleInterval, Span1D, CoordinateRange};
use parquet::{
    self,
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
        },
    },
};
use std::{
    fs, io::{self, Seek}
};

use mzpeak_prototyping::index::{
    read_point_mz_index,
    read_point_spectrum_index,
    spectrum_index_range_for_time_range,
};


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

    let mut handle = fs::File::open(
        args.filename
    )?;

    let start = std::time::Instant::now();
    let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(
        handle.try_clone()?,
        ArrowReaderOptions::new().with_page_index(true),
    )?;
    let point_spectrum_index = read_point_spectrum_index(&reader).unwrap();
    let point_mz_index = read_point_mz_index(&reader).unwrap();

    let time_range = SimpleInterval::new(args.time_range.start.unwrap() as f32, args.time_range.end.unwrap() as f32);
    let mz_range = SimpleInterval::new(args.mz_range.start.unwrap(), args.mz_range.end.unwrap());
    let im_range = SimpleInterval::new(args.im_range.start.unwrap(), args.im_range.end.unwrap());

    let index_range = spectrum_index_range_for_time_range(&mut handle, time_range)?;
    let query_range_end = std::time::Instant::now();
    eprintln!("{} seconds elapsed reading indices", (query_range_end - start).as_secs_f64());

    handle.seek(io::SeekFrom::Start(0))?;

    let start = std::time::Instant::now();

    let index_selection = point_spectrum_index.row_selection_overlaps(&index_range);
    let mz_selection = point_mz_index.row_selection_overlaps(&mz_range);

    let projection = ProjectionMask::columns(
        reader.parquet_schema(),
        [
            "point.spectrum_index",
            "point.mz",
            "point.intensity",
            "point.im",
        ],
    );

    let reader = reader
    .with_row_selection(
        index_selection.intersection(&mz_selection)
    ).with_projection(projection)
    .build()?;

    for batch in reader.flatten() {
        // eprintln!("{}", batch.num_rows());
        let point = batch.column(0).as_struct();
        let spectrum_index_array: &PrimitiveArray<UInt64Type> = point.column(0).as_primitive();

        if spectrum_index_array.is_empty() { continue; }
        if spectrum_index_array.value(spectrum_index_array.len() - 1) < index_range.start {
            continue;
        }

        let mz_array: &PrimitiveArray<Float64Type> = point.column(1).as_primitive();
        let intensity_array: &PrimitiveArray<Float32Type> = point.column(2).as_primitive();
        let im_array: &PrimitiveArray<Float64Type> = point.column(3).as_primitive();
        let it = spectrum_index_array.iter().zip(
            mz_array.iter().zip(
                intensity_array.iter().zip(
                    im_array.iter()
                )
            )
        );

        for (spectrum_index, (mz, (intensity, im))) in it {
            if let (Some(spectrum_index), Some(mz), Some(intensity), Some(im)) = (spectrum_index, mz, intensity, im) {
                if index_range.contains(&spectrum_index) && mz_range.contains(&mz) && im_range.contains(&im) {
                    println!("{spectrum_index}\t{mz}\t{intensity}\t{im}");
                }
            }
        }
    }
    let end = std::time::Instant::now();
    eprintln!("{} seconds elapsed", (end - start).as_secs_f64());
    Ok(())
}