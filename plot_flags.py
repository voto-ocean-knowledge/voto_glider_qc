from pathlib import Path
import xarray as xr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from flag_qartod import flagger


def plot_qc(ds):
    vars = list(ds)
    time = ds["time"]
    for var_name in vars:
        if var_name[-2:] == "qc":
            flag = ds[var_name]
            var = var_name[:-3]
            data = ds[var]
            meaning = np.empty(len(time), dtype=object)
            meaning[:] = "UNKNOWN"
            meaning[flag == 1] = "GOOD"
            meaning[flag == 9] = "MISSING"
            meaning[flag == 3] = "SUSPECT"
            meaning[flag == 4] = "FAIL"
            df = pd.DataFrame({"time": time, var: data, "flag": flag, "depth": ds.depth, "quality control": meaning})
            fig1 = px.line(df, x="time", y=var)
            fig1.update_traces(line=dict(color='rgba(50,50,50,0.2)'))
            fig2 = px.scatter(df, x="time", y=var, color="quality control", size="flag",
                              hover_data=['quality control'], symbol="flag",
                              color_discrete_sequence=["red", "green", "blue"])

            fig3 = go.Figure(data=fig1.data + fig2.data)
            fig3.write_html(f"figures/time_{var}.html")
            fig = px.scatter(df, x="time", y="depth", color=var, size="flag",
                             hover_data=['quality control'], symbol="flag")
            fig.write_html(f"figures/depth_{var}.html")


if __name__ == '__main__':
    ds_path = Path("/home/callum/Downloads/glider_data/CABLE.nc")
    ds_in = xr.open_dataset(ds_path)
    ds_in = flagger(ds_in)
    plot_qc(ds_in)
