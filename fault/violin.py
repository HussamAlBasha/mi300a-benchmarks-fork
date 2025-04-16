from pandas import DataFrame
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

sns.set_theme()

recs = []
labels = ("GPU Major", "GPU Minor", "1CPU", "12CPU")
dats = ("fault", "minor", "cpu", "omp12")

for cat in range(len(dats)):
    dat = np.loadtxt(f"{dats[cat]}-1.dat")
    nodat = np.loadtxt(f"no{dats[cat]}-1.dat")

    base = np.mean(nodat)
    for el in dat:
        recs.append((labels[cat], (el-base)/1000))

df = DataFrame.from_records(recs, columns=("label", "lat"))

plt.figure(figsize=(5,2.), layout="constrained")

sns.violinplot(data=df, x="label", y="lat")

plt.ylim(0)

plt.xlabel(None)
plt.ylabel("Latency (us)")

#plt.show()
plt.savefig("fault_lat_violin.pdf")
