import click
import pyarrow as pa
from mzpeak import MzPeakFile

@click.command
@click.argument('path')
@click.argument('table')
def main(path: str, table: str):
    archive = MzPeakFile(path)
    if table == 'point':
        handle = archive.spectrum_data.handle
    else:
        handle = archive.spectrum_metadata.handle
    schema: pa.Schema = handle.schema_arrow
    subfield = schema.field(table)
    fields = [(0, subfield)]
    while fields:
        level, field = fields.pop()
        subfields = field.flatten()[::-1]
        if len(subfields) == 1:
            f = subfields[0]
            if hasattr(f.type, 'value_field'):
                print("    " * level, 'list', f.name)
                fields.append((level + 1, f.type.value_field))
            else:
                print('    ' * level, f.name, f.type, f"nullable={f.nullable}")
        else:
            for f in subfields:
                fields.append((level + 1, f))
            print("    " * level, field.name, 'struct', f"nullable={field.nullable}")


if __name__ == '__main__':
    main.main()