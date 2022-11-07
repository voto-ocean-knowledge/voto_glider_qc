import xarray as xr
import numpy as np
import ioos_qc
from ioos_qc.config import Config
from ioos_qc.qartod import aggregate
from ioos_qc.streams import XarrayStream
from ioos_qc.results import collect_results, CollectedResult
import datetime
from pathlib import Path

pyglider_cotede_var_names = {
    "pressure": "PRES",
    "temperature": "TEMP",
    "salinity": "PSAL",
    "depth": "DEPTH",
}

temperature_vars = ["temperature"]
salinity_vars = ["salinity", "conductivity"]
cond_temp_vars = ["potential_density", "density", "potential_temperature"]
secondary_vars = ["oxygen_concentration", "chlorophyll"]
vars_to_flag = temperature_vars + salinity_vars + cond_temp_vars + secondary_vars

temp_config = {
    "temperature": {
        "qartod": {
            "gross_range_test": {"suspect_span": [0, 30], "fail_span": [-2.5, 40]},
            "spike_test": {"suspect_threshold": 2.0, "fail_threshold": 6.0},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}

salinity_config = {
    "salinity": {
        "qartod": {
            "gross_range_test": {"suspect_span": [5, 30], "fail_span": [2, 41]},
            "spike_test": {"suspect_threshold": 0.3, "fail_threshold": 0.9},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}

tempsal_config = {**temp_config, **salinity_config}

oxygen_config = {
    "oxygen_concentration": {
        "qartod": {
            "gross_range_test": {"suspect_span": [0, 350], "fail_span": [0, 500]},
            "spike_test": {"suspect_threshold": 10, "fail_threshold": 50},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}


chl_config = {
    "chlorophyll": {
        "qartod": {
            "gross_range_test": {"suspect_span": [-1, 10], "fail_span": [-1, 15]},
            "spike_test": {"suspect_threshold": 1, "fail_threshold": 5},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}


def apply_ioos_flags(ds, config):
    c = Config(config)
    qc = XarrayStream(ds, lon="longitude", lat="latitude")
    runner = list(qc.run(c))
    results = collect_results(runner, how='list')
    agg = CollectedResult(
        stream_id='',
        package='qartod',
        test='qc_rollup',
        function=aggregate,
        results=aggregate(results),
        tinp=qc.time(),
        data=ds
    )
    flag_vals = agg.results
    call = c.calls
    proc_record = str(call)
    return flag_vals, proc_record


def flag_ioos(ds):
    # extract ioos flags for these variables
    temp_flags, temp_flag_comment = apply_ioos_flags(ds, temp_config)
    temp_flagged_prop = 100 * sum(np.logical_and(temp_flags > 1, temp_flags < 9)) / len(temp_flags)
    print(f"Flagged {temp_flagged_prop.round(5)} % of temperature as bad")
    sal_flags, sal_flag_comment = apply_ioos_flags(ds, salinity_config)
    sal_flagged_prop = 100 * sum(np.logical_and(sal_flags > 1, sal_flags < 9)) / len(sal_flags)
    print(f"Flagged {sal_flagged_prop.round(5)} % of salinity as bad")
    cond_temp_flags, cond_temp_flag_comment = apply_ioos_flags(ds, tempsal_config)
    combi_flagged_prop = 100 * sum(np.logical_and(cond_temp_flags > 1, cond_temp_flags < 9)) / len(cond_temp_flags)
    print(f"Flagged {combi_flagged_prop.round(5)} % of values derived from temperature and salinity as bad")
    oxy_flags, oxy_flag_comment = apply_ioos_flags(ds, oxygen_config)
    oxy_flagged_prop = 100 * sum(np.logical_and(oxy_flags > 1, oxy_flags < 9)) / len(oxy_flags)
    print(f"Flagged {oxy_flagged_prop.round(5)} % of oxygen as bad")
    chl_flags, chl_flag_comment = apply_ioos_flags(ds, chl_config)
    chl_flagged_prop = 100 * sum(np.logical_and(chl_flags > 1, chl_flags < 9)) / len(chl_flags)
    print(f"Flagged {chl_flagged_prop.round(5)} % of chlorophyll as bad")
    # Apply flags and add comment
    for name_pyglider in vars_to_flag:
        if name_pyglider in temperature_vars:
            flag = temp_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD https://github.com/ioos/ioos_qc Version: " \
                           f"{ioos_qc.__version__}. Threshold values from EuroGOOS DATA-MEQ Working Group (2010)" \
                           f" Recommendations for in-situ data Near Real Time Quality Control [Version 1.2]. EuroGOOS" \
                           f", 23pp. DOI http://dx.doi.org/10.25607/OBP-214. Using config: {temp_flag_comment} "
        elif name_pyglider in salinity_vars:
            flag = sal_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD https://github.com/ioos/ioos_qc Version: " \
                           f"{ioos_qc.__version__}. Threshold values from EuroGOOS DATA-MEQ Working Group (2010)" \
                           f" Recommendations for in-situ data Near Real Time Quality Control [Version 1.2]. EuroGOOS" \
                           f", 23pp. DOI http://dx.doi.org/10.25607/OBP-214. Using config: {sal_flag_comment} "
        elif name_pyglider in cond_temp_vars:
            flag = cond_temp_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD https://github.com/ioos/ioos_qc Version: " \
                           f"{ioos_qc.__version__}. Threshold values from EuroGOOS DATA-MEQ Working Group (2010)" \
                           f" Recommendations for in-situ data Near Real Time Quality Control [Version 1.2]. EuroGOOS" \
                           f", 23pp. DOI http://dx.doi.org/10.25607/OBP-214. Using config: {cond_temp_flag_comment} "
        elif name_pyglider == "oxygen_concentration":
            flag = cond_temp_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD https://github.com/ioos/ioos_qc Version: " \
                           f"{ioos_qc.__version__}. Using config: {oxy_flag_comment} "
        elif name_pyglider == "chlorophyll":
            flag = cond_temp_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD https://github.com/ioos/ioos_qc Version: " \
                           f"{ioos_qc.__version__}. Using config: {chl_flag_comment} "
        else:
            print(f"no flags found for {name_pyglider}")
            continue

        ds[f"{name_pyglider}_qc"].values = flag
        ds[f"{name_pyglider}_qc"].attrs["comment"] = ioos_comment
        ds[f"{name_pyglider}_qc"].attrs["quality_control_set"] = 1
    return ds


def flag_oxygen(ds):
    oxy_meta_str = ds.oxygen
    import ast
    oxy_meta = ast.literal_eval(oxy_meta_str)
    cal_date = datetime.date.fromisoformat(oxy_meta["calibration_date"])
    if "coda" in oxy_meta["make_model"] and cal_date < datetime.date(2022, 6, 30):
        # These early batches of codas were improperly calibrated
        print("bad legato")
        pre_flags = ds["oxygen_concentration_qc"].values
        sus_flags = np.ones(len(pre_flags), dtype=int) * 3
        ds["oxygen_concentration_qc"].values = np.maximum(pre_flags, sus_flags)
        original_comment = ds["oxygen_concentration_qc"].attrs["comment"]
        bad_oxy_comment = "Oxygen optode improperly calibrated during this deployment. Data may be recoverable."
        if "qartod" in original_comment.lower():
            comment = f"{bad_oxy_comment} {original_comment}"
        else:
            comment = bad_oxy_comment
        ds["oxygen_concentration_qc"].attrs["comment"] = comment
        ds["oxygen_concentration_qc"].attrs["quality_control_set"] = 1
    return ds


def flagger(ds):
    for name in vars_to_flag:
        if name not in list(ds):
            print(f"{name} not found in ds. Skipping")
            continue
        flag = ds[name].copy()
        flag_values = np.empty(len(flag.values), dtype=int)
        flag_values[:] = 2
        flag.values = flag_values
        parent_attrs = flag.attrs
        attrs = {
            'ioos_qc_module': 'qartod',
            "quality_control_conventions": "IOOS QARTOD standard flags",
            "quality_control_set": 0,
            "valid_min": 1,
            "valid_max": 9,
            "flag_values": [1, 2, 3, 4, 9],
            'flag_meanings': 'GOOD, UNKNOWN, SUSPECT, FAIL, MISSING',
            "long_name": f"quality control flags for {parent_attrs['long_name']}",
            "standard_name": f"{parent_attrs['standard_name']}_flag",
            "comment": "No QC applied to this variable"}
        flag.attrs = attrs
        ds[f"{name}_qc"] = flag
    ds = flag_ioos(ds)
    ds = flag_oxygen(ds)
    ds.attrs["processing_level"] = "L1. Quality control flags"
    ds.attrs["disclaimer"] = "Data, products and services from VOTO are provided 'as is' without any warranty as" \
                             " to fitness for a particular purpose."
    return ds


if __name__ == '__main__':
    ds_path = Path("/home/callum/Downloads/glider_data/CABLE.nc")
    ds_in = xr.open_dataset(ds_path)
    ds_in = flagger(ds_in)
    ds_path_parts = list(ds_path.parts)
    fn, extension = ds_path_parts[-1].split(".")
    fn_out = fn + "_flag"
    ds_path_parts[-1] = f"{fn_out}.{extension}"
    ds_out_path = Path("/".join(ds_path_parts))
    ds_in.to_netcdf(ds_out_path)
