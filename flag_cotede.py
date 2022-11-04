import xarray as xr
import numpy as np
import cotede
import datetime
import pandas as pd
from pathlib import Path


class DummyDataset(object):
    """Minimalist data object that contains data and attributes. From CoTeDe notebooks
    """

    def __init__(self):
        """Two dictionaries to store the data and attributes
        """
        self.attrs = {}
        self.data = {}

    def __getitem__(self, key):
        """Return the requested item from the data
        """
        return self.data[key]

    def keys(self):
        """Show the available variables in data
        """
        return self.data.keys()

    @property
    def attributes(self):
        """Temporary requirement while Gui is refactoring CoTeDe. This will be soon unnecessary
        """
        return self.attrs


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


def flag_cotede(ds, cotede_config="eurogoos"):
    # Prepare dataset that CoTeDe will understand
    mydata = DummyDataset()
    for name_pyglider, name_cotede in pyglider_cotede_var_names.items():
        mydata.data[name_cotede] = ds[name_pyglider].values

    dt = ds.time.mean().values
    mydata.attrs['datetime'] = pd.DatetimeIndex([dt])[0].to_pydatetime()
    mydata.attrs['latitude'] = ds.latitude.mean()
    mydata.attrs['longitude'] = ds.longitude.mean()
    pqced = cotede.qc.ProfileQC(mydata, cfg=cotede_config)
    cotede_comment = f"Quality control flags from CoTeDe: Castelão, G. P., (2020). A Framework to Quality Control" \
                     f" Oceanographic Data." \
                     f" Journal of Open Source Software, 5(48), 2063, https://doi.org/10.21105/joss.02063 Version: " \
                     f"{cotede.__version__}. Using cfg: {cotede_config} EuroGOOS DATA-MEQ Working Group (2010)" \
                     f" Recommendations for in-situ data Near Real Time Quality Control [Version 1.2]. EuroGOOS," \
                     f" 23pp. DOI http://dx.doi.org/10.25607/OBP-214"
    # extract CoTeDe flags for these variables
    temp_flags = pqced.flags["TEMP"]["overall"]
    temp_flagged_prop = 100 * sum(np.logical_and(temp_flags > 1, temp_flags < 9)) / len(temp_flags)
    print(f"Flagged {temp_flagged_prop.round(5)} % of temperature as bad")
    sal_flags = pqced.flags["PSAL"]["overall"]
    sal_flagged_prop = 100 * sum(np.logical_and(sal_flags > 1, sal_flags < 9)) / len(sal_flags)
    print(f"Flagged {sal_flagged_prop.round(5)} % of salinity as bad")
    # Combined flags are maximum flag of temperature and salinity data
    cond_temp_flags = np.max(np.array((temp_flags, sal_flags)), 0)
    combi_flagged_prop = 100 * sum(np.logical_and(cond_temp_flags > 1, cond_temp_flags < 9)) / len(cond_temp_flags)
    print(f"Flagged {combi_flagged_prop.round(5)} % of values derived from temperature and salinity as bad")
    # Apply flags and add comment
    for name_pyglider in vars_to_flag:
        if name_pyglider in temperature_vars:
            flag = temp_flags
        elif name_pyglider in salinity_vars:
            flag = sal_flags
        elif name_pyglider in cond_temp_vars:
            flag = cond_temp_flags
        else:
            continue
        ds[f"{name_pyglider}_quality_control"].values = flag
        ds[f"{name_pyglider}_quality_control"].attrs["comment"] = cotede_comment
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
        flag = ds[name].copy()
        flag_values = np.empty(len(flag.values), dtype=int)
        flag_values[:] = 0
        flag.values = flag_values
        parent_attrs = flag.attrs
        attrs = {"quality_control_conventions": "IMOS standard flags",
                 "quality_control_set": 0,
                 "valid_min": 0,
                 "valid_max": 9,
                 "flag_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 "flag_meanings": ", ".join(["no_qc_performed", "good_data", "probably_good_data",
                                             "bad_data_that_are_potentially_correctable", "bad_data", "value_changed",
                                             "not_used", "not_used", "interpolated_values", "missing_values"]),
                 "long_name": f"quality control flags for {parent_attrs['long_name']}",
                 "standard_name": f"{parent_attrs['long_name']}_flag",
                 "comment": "No QC applied to this variable"}
        flag.attrs = attrs
        ds[f"{name}_quality_control"] = flag
    ds = flag_cotede(ds)
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
