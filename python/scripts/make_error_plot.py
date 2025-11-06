from pathlib import Path

import polars as pl
import numpy as np
import matplotlib
import click

matplotlib.use("agg")

from matplotlib import pyplot as plt
import seaborn as sb


@click.command("make_error_plot")
@click.argument("delta_path", type=click.Path(path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("-t", "--dataset-title", default=None, help="The name of this dataset")
@click.option(
    "--bin-max", type=float, default=0.003, help="The m/z error maximum bin value"
)
@click.option(
    "--num-bins",
    type=int,
    default=400,
    help="The number of m/z error bins to create between 0 and `--bin-max`",
)
@click.option(
    "--has-delta-encoding",
    is_flag=True,
    help="Indicate that the dataset includes delta encoding chunked data",
)
@click.option(
    "--has-null-marking/--no-null-marking",
    is_flag=True,
    default=False,
    help="Indicate that the dataset includes null marked sparse data",
)
def main(
    delta_path: Path,
    output_path: Path,
    dataset_title: str,
    bin_max: float = 0.003,
    num_bins: int = 400,
    has_delta_encoding: bool = False,
    has_null_marking: bool = True,
):
    if dataset_title is None:
        dataset_title = delta_path.name.split(".")[0]
    delta_df = pl.scan_parquet(delta_path)
    x = delta_df.select("diff").collect()
    y = delta_df.select("numpress_diff").collect()
    bins = np.linspace(0, bin_max, num_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=180)
    ax1.hist(x["diff"].abs(), bins=bins, edgecolor="black", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.xaxis.set_major_locator(plt.FixedLocator(bins, nbins=5))
    ax1.xaxis.set_major_formatter(lambda x, _: "%0.3g" % x)
    sb.despine(ax=ax1)
    ax1.set_xlabel(r"$\delta$ peak m/z ")
    ax1.set_ylabel("count")
    encodings = []
    if has_null_marking:
        encodings.append("null marking")
    if has_delta_encoding:
        encodings.append("delta encoding")
    title_encoding = " and ".join(encodings)
    if title_encoding:
        ax1.set_title(f"Absolute error of {title_encoding}\nprofile on centroid m/z")
    else:
        ax1.set_title("Absolute error of plain encoding profile on centroid m/z")
    ax1.text(
        0.7,
        0.8,
        f"median={x['diff'].abs().median():0.4g}",
        transform=ax1.transAxes,
        ha="left",
    )
    ax1.text(
        0.7,
        0.75,
        rf"$\mu$={x['diff'].abs().mean():0.4g}",
        transform=ax1.transAxes,
        ha="left",
    )
    ax1.text(
        0.7,
        0.7,
        rf"$\sigma$={x['diff'].abs().std():0.4g}",
        transform=ax1.transAxes,
        ha="left",
    )
    ax1.text(
        0.7,
        0.65,
        rf"max={x['diff'].abs().max():0.4g}",
        transform=ax1.transAxes,
        ha="left",
    )

    ax2.hist(y["numpress_diff"].abs(), bins=bins, edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.xaxis.set_major_locator(plt.FixedLocator(bins, nbins=5))
    ax2.xaxis.set_major_formatter(lambda x, _: "%0.3g" % x)
    sb.despine(ax=ax2)
    ax2.set_xlabel(r"$\delta$ peak m/z ")
    ax2.set_title(
        "Absolute error of numpress recoding of\noriginal profile spectra on centroid m/z"
    )
    ax2.text(
        0.7,
        0.8,
        f"median={y['numpress_diff'].abs().median():0.4g}",
        transform=ax2.transAxes,
        ha="left",
    )
    ax2.text(
        0.7,
        0.75,
        rf"$\mu$={y['numpress_diff'].abs().mean():0.4g}",
        transform=ax2.transAxes,
        ha="left",
    )
    ax2.text(
        0.7,
        0.7,
        rf"$\sigma$={y['numpress_diff'].abs().std():0.4g}",
        transform=ax2.transAxes,
        ha="left",
    )
    ax2.text(
        0.7,
        0.65,
        rf"max={y['numpress_diff'].abs().max():0.4g}",
        transform=ax2.transAxes,
        ha="left",
    )

    fig.suptitle(
        f"Error of encoding of m/z on centroid relative to original signal for\n{dataset_title}",
        y=1.1,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main.main()
