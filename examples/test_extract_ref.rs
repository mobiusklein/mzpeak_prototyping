use std::{io, env};

use mzdata::prelude::*;
use mzpeaks::coordinate::SimpleInterval;



fn main() -> io::Result<()> {
    let filename = env::args().skip(1).next().unwrap();

    let mut reader = mzdata::MZReader::open_path(
        filename
    )?;

    let start = std::time::Instant::now();

    let time_range = SimpleInterval::new(10.0, 21.0);
    let mz_range = SimpleInterval::new(623.0, 625.0);
    let im_range = SimpleInterval::new(0.8, 1.2);

    let it = reader.start_from_time(time_range.start)?;
    while let Some(spec) = it.next() {
        if let Some(arrays) = spec.arrays.as_ref() {
            let mzs = arrays.mzs()?;
            let ints = arrays.intensities()?;
            let (ims, _) = arrays.ion_mobility()?;

            for (mz, (int, im)) in mzs.iter().zip(ints.iter().zip(ims.iter())) {
                if mz_range.contains(mz) && im_range.contains(im) {
                    println!("{}\t{mz}\t{int}\t{im}", spec.index());
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