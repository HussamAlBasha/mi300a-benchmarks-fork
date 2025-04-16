import pandas
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

sns.set_theme()

dtname = {"u64": "UINT64", "f64": "FP64"}
szname = {"s10": "1K", "s20": "1M"}

dfs = [pandas.read_csv(f"{dt}-{unit}-{sz}.csv", names=["GPU threads", "CPU threads", "speedup"]).assign(unit=unit.upper(), case=f"{szname[sz]}-{dtname[dt]}") for unit in ("cpu", "gpu") for dt, sz in [("u64", "s10"), ("u64", "s20"), ("f64", "s20")]]

df = pandas.concat(dfs, ignore_index=True)

print(df)

#plt.figure(figsize=(6.5,6.25), layout="constrained")
#hax, cax = plt.subplots(1, 2)

fg = sns.FacetGrid(df, col="case", row="unit", height=2.5, aspect=0.85)

cbar_ax = fg.fig.add_axes([1.0,0.10, 0.015, 0.8])
#cbar_ax = fg.fig.add_axes([0.1,1.05, 0.8, 0.015])

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index="CPU threads", columns="GPU threads", values="speedup")
    h = sns.heatmap(d, **kwargs)
    h.tick_params(axis="y", rotation=0)

#fg.map_dataframe(draw_heatmap, vmin=0.3, vmax=1.5, center=1, cmap="PiYG")#, cbar=False)
#fg.map_dataframe(draw_heatmap, vmin=0.3, vmax=1.5, center=1, cmap="RdBu", cbar=False)
fg.map_dataframe(draw_heatmap, vmin=0.1, vmax=1.5, center=1, cmap="RdBu", cbar=False)

#fg.map_dataframe(draw_heatmap, norm=matplotlib.colors.LogNorm(vmin=0.2, vmax=2), center=1, cmap="PiYG")#, cbar=False)

#f = fg.map_dataframe(draw_heatmap, cbar_kws={"norm": matplotlib.colors.LogNorm(vmin=0.1, vmax=2)}, center=1, cmap="PiYG", cbar_ax=cbar_ax)

#fg.map_dataframe(draw_heatmap)#, cbar=False)

fg.set_titles(template="{row_name}-{col_name}")

plt.figure()
#sns.heatmap([[0.1,2.0]], vmin=0.3, vmax=1.5, center=1, cmap="RdBu", cbar_ax=cbar_ax)
#sns.heatmap([[0.1,2.0]], vmin=0.3, vmax=1.2, center=1, cmap="RdBu", cbar_ax=cbar_ax, cbar_kws=dict(orientation="horizontal"))
#sns.heatmap([[0.1,2.0]], vmin=0.1, vmax=1.2, center=1, cmap="RdBu", cbar_ax=cbar_ax, cbar_kws=dict(orientation="horizontal"))
sns.heatmap([[0.1,2.0]], vmin=0.1, vmax=1.2, center=1, cmap="RdBu", cbar_ax=cbar_ax)
#plt.figure()
#sns.heatmap([[0.1,2.0]], norm=matplotlib.colors.LogNorm(vmin=0.2, vmax=2), center=1, cmap="PiYG", cbar_ax=cbar_ax)

#plt.colorbar(cax=cbar_ax)

#sns.heatmap(df, annot=True)

#plt.show()
fg.fig.savefig("hybrid3.pdf", bbox_extra_artists=(cbar_ax,), bbox_inches="tight")
