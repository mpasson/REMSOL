import multilayer_solver as msl
import numpy as np
from multilayer_solver import Polarization as pol

slab = msl.MultiLayer(
    [
        msl.Layer(1, 1),
        msl.Layer(2, 0.6),
        msl.Layer(1, 1),
        # msl.Layer(2, 0.6),
        # msl.Layer(1, 1),
    ]
)

np.pi

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    field = slab.field(2.0 * np.pi / 1.55, pol.TM, 0)
    for component in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        comp = getattr(field, component)
        plt.plot(field.x, np.real(comp) + np.imag(comp), label=component)
    plt.legend()
    plt.show()
