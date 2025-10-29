use std::{
    fs, io,
    path::PathBuf,
    sync::{Arc, mpsc::sync_channel},
    thread,
};

use clap::Parser;
use parquet::arrow::ArrowWriter;
use serde::{Deserialize, Serialize};
use serde_arrow;

use mzdata::mzsignal::PeakPicker;
use mzdata::{self, io::MZReader, prelude::*, spectrum::SignalContinuity};
use mzpeak_prototyping::MzPeakReader;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
struct PeakError {
    spectrum_index: usize,
    spec_mz: f64,
    diff: f64,
    numpress_diff: f64,
    intensity: f32,
}

impl PeakError {
    fn new(spectrum_index: usize, spec_mz: f64, diff: f64, numpress_diff: f64, intensity: f32) -> Self {
        Self {
            spectrum_index,
            spec_mz,
            diff,
            numpress_diff,
            intensity,
        }
    }
}

#[derive(Parser)]
struct App {
    #[arg()]
    mzpeak_filename: PathBuf,
    #[arg()]
    ref_filename: PathBuf,
    #[arg()]
    outfile: PathBuf,
}

fn numpress_peaks(
    raw_arrays: &mzdata::spectrum::BinaryArrayMap,
) -> Vec<mzdata::mzsignal::FittedPeak> {
    let mut raw_mzs = raw_arrays
        .get(&mzdata::spectrum::ArrayType::MZArray)
        .unwrap()
        .clone();
    raw_mzs
        .store_compressed(mzdata::spectrum::bindata::BinaryCompressionType::NumpressLinear)
        .unwrap();
    let raw_mzs = raw_mzs.to_f64().unwrap();
    let intensities = raw_arrays.intensities().unwrap();
    let picker = PeakPicker::default();
    let mut peaks = Vec::new();
    picker
        .discover_peaks(&raw_mzs, &intensities, &mut peaks)
        .unwrap();
    peaks
}

fn main() -> io::Result<()> {
    env_logger::init();
    let args = App::parse();

    let ref_reader = MZReader::open_path(&args.ref_filename)?;
    let mp_reader = MzPeakReader::new(&args.mzpeak_filename)?;
    let n = ref_reader.len();
    let mut n_peaks = 0;

    let fields: Vec<arrow::datatypes::FieldRef> =
        serde_arrow::schema::SchemaLike::from_type::<PeakError>(Default::default()).unwrap();
    let schema = Arc::new(arrow::datatypes::Schema::new(fields.clone()));
    let mut writer = ArrowWriter::try_new(fs::File::create(&args.outfile)?, schema.clone(), None)?;

    let (mp_send, mp_recv) = sync_channel(100);
    let (ref_send, ref_recv) = sync_channel(100);

    let mp_read = thread::spawn(move || {
        let picker = PeakPicker::default();

        for mut spec in mp_reader {
            let arrays = spec.raw_arrays().unwrap();
            let mut acc = Vec::new();
            picker
                .discover_peaks(
                    &arrays.mzs().unwrap(),
                    &arrays.intensities().unwrap(),
                    &mut acc,
                )
                .unwrap();

            spec.pick_peaks(1.0).unwrap();
            mp_send.send((spec, acc)).unwrap();
        }
    });

    let ref_read = thread::spawn(move || {
        let picker = PeakPicker::default();
        for mut spec in ref_reader {
            let arrays = spec.raw_arrays().unwrap();
            let mut acc = Vec::new();
            picker
                .discover_peaks(
                    &arrays.mzs().unwrap(),
                    &arrays.intensities().unwrap(),
                    &mut acc,
                )
                .unwrap();
            spec.pick_peaks(1.0).unwrap();
            ref_send.send((spec, acc)).unwrap();
        }
    });

    let cmpr = thread::spawn(move || -> io::Result<()> {
        let mut errors = Vec::new();
        for (i, ((spec, spec_peaks), (ref_spec, ref_peaks))) in
            mp_recv.into_iter().zip(ref_recv.into_iter()).enumerate()
        {
            if i % 1000 == 0 {
                log::info!(
                    "Working on spectrum {i}/{n} ({:0.2}%), {n_peaks} peaks processed so far.",
                    (i as f64 / n as f64 * 100.0)
                );
            }
            if spec.signal_continuity() != SignalContinuity::Profile {
                continue;
            }

            let raw_arrays: &mzdata::spectrum::BinaryArrayMap = ref_spec.raw_arrays().unwrap();
            let numpress_peaks = numpress_peaks(raw_arrays);

            // let spec_peaks = spec.peaks.as_ref().unwrap();
            // let ref_peaks = ref_spec.peaks.as_ref().unwrap();
            n_peaks += spec_peaks.len();

            assert_eq!(
                spec_peaks.len(),
                ref_peaks.len(),
                "{}/{} {} level {} did not have the same number of peaks, {} != {}",
                spec.id(),
                ref_spec.id(),
                spec.index(),
                spec.ms_level(),
                spec_peaks.len(),
                ref_peaks.len()
            );

            for ((up, rp), np) in spec_peaks
                .iter()
                .zip(ref_peaks.iter())
                .zip(numpress_peaks.iter())
            {
                let e = PeakError::new(spec.index(), rp.mz, rp.mz - up.mz, rp.mz - np.mz, rp.intensity);
                errors.push(e);
                if (rp.mz - up.mz).abs() > 0.001 {
                    println!(
                        "{}: {up:?} vs {rp:?} differ by {}",
                        spec.index(),
                        rp.mz - up.mz
                    );
                }
            }

            if errors.len() > 10_000 {
                let batch = serde_arrow::to_record_batch(&fields, &errors).unwrap();
                writer.write(&batch)?;
                errors.clear();
            }
        }
        if !errors.is_empty() {
            let batch = serde_arrow::to_record_batch(&fields, &errors).unwrap();
            writer.write(&batch)?;
            errors.clear();
        }
        writer.finish()?;
        Ok(())
    });

    mp_read.join().unwrap();
    ref_read.join().unwrap();

    cmpr.join().unwrap()?;
    Ok(())
}
