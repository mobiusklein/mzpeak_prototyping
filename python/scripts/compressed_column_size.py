import click
import humanize

from mzpeak import MzPeakFile

@click.command
@click.option("-c", "--chromatograms", is_flag=True)
@click.argument('path')
@click.argument("column_path")
def main(path: str, column_path: str, chromatograms: bool):
    archive = MzPeakFile(path)
    if chromatograms:
        if column_path.startswith(("point", "chunk")):
            meta = archive.chromatogram_data.meta
        else:
            meta = archive.chromatogram_metadata.meta
    else:
        if column_path.startswith(("point", 'chunk')):
            meta = archive.spectrum_data.meta
        else:
            meta = archive.spectrum_metadata.meta

    z = 0
    zu = 0
    for i in range(0, meta.num_row_groups):
        rg = meta.row_group(i)
        for j in range(meta.num_columns):
            col_idx = rg.column(j)
            if col_idx.path_in_schema == column_path or (col_idx.path_in_schema == column_path + '.list.item'):
                z += col_idx.total_compressed_size
                zu += col_idx.total_uncompressed_size
                break
        else:
            raise click.ClickException(
                f"Column {column_path} was not found in {meta.schema}"
            )
    print(
        f"Compressed Size: {humanize.naturalsize(z, format='%.3f')} over {meta.num_row_groups} row groups"
    )
    print(f"Decompressed Size: {humanize.naturalsize(zu, format='%.3f')}")

if __name__ == '__main__':
    main.main()