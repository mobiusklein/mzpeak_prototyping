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

#[derive(Debug, Default, Serialize, Deserialize)]
struct Histogram {
    edges: Vec<f64>,
    counts: Vec<u64>,
}

impl Histogram {
    fn from_range(start: f64, end: f64, step: f64) -> Self {
        let edges = mzdata::mzsignal::gridspace(start, end, step);
        let counts = vec![0; edges.len()];
        Self { edges, counts }
    }

    fn add(&mut self, val: f64) {
        match self.edges.binary_search_by(|probe| probe.total_cmp(&val)) {
            Ok(i) => {
                self.counts[i] += 1;
            }
            Err(i) => {
                if i < self.edges.len() {
                    self.counts[i] += 1;
                } else {
                    self.counts[self.edges.len() - 1] += 1;
                }
            }
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

fn main() -> io::Result<()> {
    env_logger::init();
    let args = App::parse();

    let ref_reader = MZReader::open_path(&args.ref_filename)?;
    let mp_reader = MzPeakReader::new(&args.mzpeak_filename)?;
    let n = ref_reader.len();
    let mut n_peaks = 0;

    let fields: Vec<arrow::datatypes::FieldRef> =
        serde_arrow::schema::SchemaLike::from_type::<Histogram>(Default::default()).unwrap();
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
        let mut errors = Histogram::from_range(-1e-5, 1e-5, 1e-12);
        for (i, ((spec, spec_peaks), (ref_spec, ref_peaks))) in mp_recv.into_iter().zip(ref_recv.into_iter()).enumerate() {
            if i % 1000 == 0 {
                log::info!(
                    "Working on spectrum {i}/{n} ({:0.2}%), {n_peaks} peaks processed so far.",
                    (i as f64 / n as f64 * 100.0)
                );
            }
            if spec.signal_continuity() != SignalContinuity::Profile {
                continue;
            }

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

            for (up, rp) in spec_peaks.iter().zip(ref_peaks.iter()) {
                errors.add(rp.mz - up.mz);
                if (rp.mz - up.mz).abs() > 0.001 {
                    println!(
                        "{}: {up:?} vs {rp:?} differ by {}",
                        spec.index(),
                        rp.mz - up.mz
                    );
                }
            }
        }
        let batch = serde_arrow::to_record_batch(&fields, &[errors]).unwrap();
        writer.write(&batch)?;
        writer.finish()?;
        Ok(())
    });

    mp_read.join().unwrap();
    ref_read.join().unwrap();

    cmpr.join().unwrap()?;
    Ok(())
}
