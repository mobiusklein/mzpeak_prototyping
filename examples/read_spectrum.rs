use std::{io, path::PathBuf, env};
use mzdata::io::mgf::MGFWriter;
use mzpeak_prototyping::MzPeakReader;


fn fetch(path: &PathBuf, index: usize) -> io::Result<()> {
    let mut reader = MzPeakReader::new(path)?;
    let mut spec = reader.get_spectrum(index)?;
    spec.pick_peaks(1.0).unwrap();

    let writer = io::stdout().lock();
    let mut writer = MGFWriter::new(writer);
    writer.write(&spec)?;
    drop(writer);
    Ok(())
}

fn main() -> io::Result<()> {
    env_logger::init();
    let mut args = env::args().skip(1);

    let path = args.next().map(|p| PathBuf::from(p)).unwrap();
    let index: usize = args.next().and_then(|v| v.parse().ok()).unwrap();

    fetch(&path, index)
}
