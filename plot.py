import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np
plt
pd

data = pd.read_csv("data.csv", names=["kre", "kim", "cre", "cim"])
data["abs"] = np.abs(data["cre"] + 1j * data["cim"])

pivot = data.pivot(index="kim", columns="kre", values="abs")
print(pivot)
plt.contourf(pivot.columns, pivot.index, np.log10(pivot.values))
plt.colorbar()
plt.show()
