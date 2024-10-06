import numpy as np
import remsol
from remsol import Polarization as pol

slab = remsol.MultiLayer(
    [
        remsol.Layer(1, 1),
        remsol.Layer(2, 0.6),
        remsol.Layer(1, 1),
        # remsol.Layer(2, 0.6),
        # remsol.Layer(1, 1),
    ]
)

np.pi

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    field = slab.field(2.0 * np.pi / 1.55, pol.TE, 0)
    for component in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        comp = getattr(field, component)
        plt.plot(field.x, np.real(comp) + np.imag(comp), label=component)
    plt.legend()
    plt.show()
