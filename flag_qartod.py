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
secondary_vars = ["oxygen_concentration"]
vars_to_flag = temperature_vars + salinity_vars + cond_temp_vars + secondary_vars




default_config = {
    "chlorophyll": {
        "qartod": {
            "gross_range_test": {"suspect_span": [0, 10], "fail_span": [-1, 20]},
            "spike_test": {"suspect_threshold": 0.5, "fail_threshold": 1},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    },
    "temperature": {
        "qartod": {
            "gross_range_test": {"suspect_span": [0, 30], "fail_span": [-5, 50]},
            "spike_test": {"suspect_threshold": 0.5, "fail_threshold": 1},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}

temp_config = {
    "temperature": {
        "qartod": {
            "gross_range_test": {"suspect_span": [0, 30], "fail_span": [-5, 50]},
            "spike_test": {"suspect_threshold": 0.5, "fail_threshold": 1},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}

salinity_config = {
    "salinity": {
        "qartod": {
            "gross_range_test": {"suspect_span": [5, 30], "fail_span": [0, 40]},
            "spike_test": {"suspect_threshold": 0.5, "fail_threshold": 1},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}

tempsal_config = {
    "temperature": {
        "qartod": {
            "gross_range_test": {"suspect_span": [0, 30], "fail_span": [-5, 50]},
            "spike_test": {"suspect_threshold": 0.5, "fail_threshold": 1},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    },
    "salinity": {
        "qartod": {
            "gross_range_test": {"suspect_span": [5, 30], "fail_span": [0, 40]},
            "spike_test": {"suspect_threshold": 0.5, "fail_threshold": 1},
            "location_test": {"bbox": [10, 50, 25, 60]},
        }
    }
}




def apply_ioos_flags(ds, config):
    c = Config(config)
    qc = XarrayStream(ds, lon="longitude", lat="latitude")
    runner = list(qc.run(c))
    results = collect_results(runner, how='list')
    for res in results:
        print(res.test)
        print(f"result {res.results}")
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
    # Combined flags are maximum flag of temperature and salinity data
    cond_temp_flags, cond_temp_flag_comment = apply_ioos_flags(ds, tempsal_config)
    combi_flagged_prop = 100 * sum(np.logical_and(cond_temp_flags > 1, cond_temp_flags < 9)) / len(cond_temp_flags)
    print(f"Flagged {combi_flagged_prop.round(5)} % of values derived from temperature and salinity as bad")
    # Apply flags and add comment
    for name_pyglider in vars_to_flag:
        if name_pyglider in temperature_vars:
            flag = temp_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD Version: " \
                           f"{ioos_qc.__version__}. Using cfg: {temp_flag_comment} "
        elif name_pyglider in salinity_vars:
            flag = sal_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD Version: " \
                           f"{ioos_qc.__version__}. Using cfg: {sal_flag_comment} "
        elif name_pyglider in cond_temp_vars:
            flag = cond_temp_flags
            ioos_comment = f"Quality control flags from IOOS QC QARTOD Version: " \
                           f"{ioos_qc.__version__}. Using cfg: {cond_temp_flag_comment} "
        else:
            continue

        ds[f"{name_pyglider}_quality_control"].values = flag
        ds[f"{name_pyglider}_quality_control"].attrs["comment"] = ioos_comment
        ds[f"{name_pyglider}_quality_control"].attrs["quality_control_set"] = 1
    return ds


def flag_oxygen(ds):
    oxy_meta_str = ds.oxygen
    import ast
    oxy_meta = ast.literal_eval(oxy_meta_str)
    cal_date = datetime.date.fromisoformat(oxy_meta["calibration_date"])
    if "coda" in oxy_meta["make_model"] and cal_date < datetime.date(2022, 6, 30):
        # These early batches of codas were improperly calibrated
        print("bad legato")
        ds["oxygen_concentration_quality_control"].values[:] = 3
        ds["oxygen_concentration_quality_control"].attrs["comment"] = "Oxygen optode improperly calibrated during " \
                                                                      "this deployment. Data may be recoverable"
        ds["oxygen_concentration_quality_control"].attrs["quality_control_set"] = 1
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
                 "standard_name": f"{parent_attrs['long_name']}_flag",
                 "comment": "No QC applied to this variable"}
        flag.attrs = attrs
        ds[f"{name}_quality_control"] = flag
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
