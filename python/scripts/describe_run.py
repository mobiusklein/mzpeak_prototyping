import click
from mzpeak import MzPeakFile


@click.command()
@click.argument('path', type=click.Path(exists=True, readable=True))
def main(path):
    handle = MzPeakFile(path)
    click.echo(f"Opened {path}")
    click.echo(
        f"Detected {len(handle)} spectra with {handle.spectrum_metadata.num_spectrum_points or '?'} data points"
    )

    click.echo(f"Has peak data? {bool(handle.spectrum_peak_data)}")

    for key, value in handle.spectrum_data.array_index.items():
        click.echo(f"Found spectrum array: {value['array_name']} with key {key}")

    click.echo(f"Has chromatogram data? {bool(handle.chromatogram_metadata)}")
    if handle.chromatogram_data:
        for key, value in handle.chromatogram_data.array_index.items():
            click.echo(f"Found chromatogram array: {value['array_name']} with key {key}")



if __name__ == '__main__':
    main.main()