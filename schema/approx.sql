
CREATE TYPE param_value AS (
    'integer' integer,
    'float' numeric,
    'bool' boolean,
    'string' varchar
);

CREATE TYPE parameter AS (
    'name' varchar,
    accession varchar,
    'value' param_value,
    unit varchar,
);


CREATE TABLE spectrum (
    'index' bigserial,
    id varchar,
    ms_level smallint,
    'time' real,
    polarity smallint,
    mz_signal_continuity varchar,
    spectrum_type varchar,
    number_of_data_points int,
    parameters parameter[],
    data_procesing_ref int,
    number_of_auxiliary_arrays int,
    auxiliary_arrays auxiliary_array[],
    mz_delta_model numeric[],
    total_ion_current numeric,
    base_peak_mz numeric,
    base_peak_intensity numeric,
    lowest_observed_mz numeric,
    highest_observed_mz numeric,
    ...
);

CREATE TABLE selected_ion (
    spectrum_index bigint,
    precursor_index bigint,
    selected_ion_mz numeric,
    charge_state int,
    intensity numeric,
    parameters parameter[],
    ...
);

CREATE TYPE scan_window AS (
    lower_limit numeric,
    upper_limit numeric,
    unit varchar,
    parameters parameter[]
);

CREATE TABLE scan (
    spectrum_index bigint,
    scan_start_time numeric,
    preset_scan_configuration int,
    filter_string varchar,
    ion_injection_time numeric,
    instrument_configuration_ref int,
    parameters parameter[],
    scan_windows scan_window[],
    ...
);


CREATE TABLE point (
    spectrum_index bigint,
    mz numeric,
    intensity numeric,
    ...
);