import click
import time
import logging

from mzpeak import MzPeakFile

logger = logging.getLogger("read_mzpeak")

@click.command()
@click.argument("path", type=click.Path(exists=True, readable=True))
def main(path):
    logging.basicConfig(
        level=logging.INFO,
        stream=click.get_text_stream("stderr"),
        format="%(asctime)s | %(levelname)-6s | %(name)-9s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    reader = MzPeakFile(path)
    n_points = 0
    start = time.monotonic()
    it = iter(reader)
    for i, spec in enumerate(it):
        n_points += len(spec['m/z array'])
        if i % 5000 == 0:
            logger.info(f"Read spectrum {i}, {n_points} points read so far")

    end = time.monotonic()
    logger.info(f"Read {n_points} points from {path} over {i} spectra in {end - start:0.3f} seconds")


if __name__ == "__main__":
    main.main()