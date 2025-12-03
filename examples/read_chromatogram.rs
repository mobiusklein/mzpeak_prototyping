use mzdata::{prelude::*, spectrum::ArrayType};
use mzpeak_prototyping::MzPeakReader;
use std::{
    env,
    io,
    path::PathBuf,
};

fn fetch(path: &PathBuf, index: usize) -> io::Result<()> {
    let mut reader = MzPeakReader::new(path)?;
    let chrom = reader.get_chromatogram(index).unwrap();
    println!("Chrom: {}", chrom.id());

    let arrays = &chrom.arrays;
    for (k, v) in arrays.iter() {
        println!("{k}: {:?}", v.data_len());
    }

    let mut writer = io::stdout().lock();
    let times = arrays.get(&ArrayType::TimeArray).unwrap().to_f64()?;
    let ints = arrays.intensities()?;
    for (time, i) in times.iter().zip(ints.iter()) {
        writeln!(writer, "{time}\t{i}")?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    env_logger::init();
    let mut args = env::args().skip(1);

    let path = args.next().map(|p| PathBuf::from(p)).unwrap();
    let index: usize = args.next().and_then(|v| v.parse().ok()).unwrap();

    fetch(&path, index)
}
