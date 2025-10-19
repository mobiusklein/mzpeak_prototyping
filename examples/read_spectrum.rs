use mzdata::{io::mgf::MGFWriter, prelude::SpectrumLike};
use mzpeak_prototyping::MzPeakReader;
use std::{
    env,
    io::{self, prelude::*},
    path::PathBuf,
};

fn fetch(path: &PathBuf, index: usize) -> io::Result<()> {
    let mut reader = MzPeakReader::new(path)?;
    let mut spec = reader.get_spectrum(index).unwrap();
    spec.pick_peaks(1.0).unwrap();

    let writer = io::stdout().lock();
    let mut writer = MGFWriter::new(writer);
    writer.write(&spec)?;
    drop(writer);

    let mut writer = io::stdout().lock();
    writeln!(writer, "Raw Data:")?;
    let arrays = spec.raw_arrays().unwrap();
    let mzs = arrays.mzs()?;
    let ints = arrays.intensities()?;
    for (mz, i) in mzs.iter().zip(ints.iter()) {
        writeln!(writer, "{mz}\t{i}")?;
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
