"""
This file contains the software for a python multilayer slabsolver

There are 2 classes:
    - Multilayer: Class containing the physical software
    - MultiLayerSolver: Class containing the solver buffer for easy use
"""

import hashlib
import pickle
import warnings

import numpy as np
from scipy import optimize as opt
from scipy.integrate import simpson
from scipy.signal import find_peaks


def hash_call(*args) -> str:
    """Creates an hash from the arguments

    Args:
        *args: tuple of the arguments

    Returns:
        str: hash
    """
    bytes = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
    hasher = hashlib.sha256()
    hasher.update(bytes)
    return hasher.hexdigest()


class Multilayer:
    """Class finding the guided modes of a general multilayer

    This class assumes units where c=1, that angular frequency and wavevector have the same units.
    """

    Z0 = 376.730
    max_depth = 5

    def __init__(self, n_list=None, d_list=None):
        """Creator

        Args:
            n_list (list): List of floats: refractive indexes of the multilayer
            d_list (list): List of floats: thicknesses of the multilayer

        Returns:
            None.

        """
        self.n_list = n_list
        self.d_list = d_list

        # self.root_method = "lm"
        self.root_method = "hybr"
        self.root_options = {
            "xtol": 1e-9,
        }
        self.plot_points = 1001
        self.last_call_hash = None
        self.res = {}

    @staticmethod
    def T_matrix_TE(n1, n2, om, k):
        """Creates T matrix of the interface between two dielectric for TE polarization

        Args:
            n1 (float): Index of first dielectric.
            n2 (float): Index of second dielectric.
            om (float): Angular frequency.
            k (float): Wavevector parallel to the interface (perpendicular to propagation)

        Returns:
            array: 2x2 T-matrix

        """
        k1 = np.sqrt((1 + 0j) * ((om * n1) ** 2 - k**2))
        k2 = np.sqrt((1 + 0j) * ((om * n2) ** 2 - k**2))
        C = np.zeros((2, 2), dtype=complex)
        C[0, 1] = k2 - k1
        C[1, 0] = k2 - k1
        C[0, 0] = k2 + k1
        C[1, 1] = k2 + k1
        return C / (2.0 * k2)

    @staticmethod
    def T_matrix_TM(n1, n2, om, k):
        """Creates T matrix of the interface between two dielectric for TM polarization

        Args:
            n1 (float): Index of first dielectric.
            n2 (float): Index of second dielectric.
            om (float): Angular frequency.
            k (float): Wavevector parallel to the interface (perpendicular to propagation)

        Returns:
            array: 2x2 T-matrix

        """
        k1 = np.sqrt((1 + 0j) * ((om * n1) ** 2 - k**2))
        k2 = np.sqrt((1 + 0j) * ((om * n2) ** 2 - k**2))
        C = np.zeros((2, 2), dtype=complex)
        C[0, 1] = -(n1**2) * k2 + n2**2 * k1
        C[1, 0] = -(n1**2) * k2 + n2**2 * k1
        C[0, 0] = n1**2 * k2 + n2**2 * k1
        C[1, 1] = n1**2 * k2 + n2**2 * k1
        return C / (2.0 * k1 * n2 * n2)

    @staticmethod
    def T_prop(n, om, k, d):
        """Creates propagation T matrix

        Args:
            n (float): Index of dielectric.
            om (float): Angular frequency.
            k (float): Wavevector parallel to the interface (perpendicular to propagation)
            d (float): Index of second dielectric.

        Returns:
            array: 2x2 T-matrix

        """
        w = np.sqrt((1 + 0j) * ((om * n) ** 2 - k**2)) * d
        C = np.zeros((2, 2), dtype=complex)
        C[0, 0] = np.exp((0 + 1j) * w)
        C[1, 1] = np.exp(-(0 + 1j) * w)
        return C

    def approximate_zeros(self, func, om_start, depth=0):
        if depth > self.max_depth:
            return []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            guess = [func(om) for om in om_start]
        # plt.plot(om_start, guess)
        # plt.show()
        zero_crossings = np.where(np.diff(np.sign(np.real(guess))))[0]
        om_guess = list(0.5 * (om_start[zero_crossings] + om_start[zero_crossings + 1]))
        peaks = find_peaks(guess)[0]
        peaks_value = [func(om) for om in om_start[peaks]]
        for i, val in zip(peaks, peaks_value):
            if val < 0.0:
                new_start = np.linspace(om_start[i - 1], om_start[i + 1], 51)
                om_guess += self.approximate_zeros(func, new_start, depth + 1)
        dips = find_peaks(-np.asarray(guess))[0]
        dips_value = [func(om) for om in om_start[dips]]
        for i, val in zip(dips, dips_value):
            if val > 0.0:
                new_start = np.linspace(om_start[i - 1], om_start[i + 1], 51)
                om_guess += self.approximate_zeros(func, new_start, depth + 1)
        return list(np.sort(om_guess))

    def generalT(self, k, om, d_list=None, n_list=None, pol="TE"):
        """Creates T matrix of the multilayer for TE or TM polarization

        Args:
            k (TYPE): Parallel wavevector.
            om (TYPE): Angular frequency.
            d_list (list, optional): List of floats. Thicknesses of the layers. Defaults to None (self.d_list).
            n_list (TYPE, optional): List of floats. Indexes of the layers. Defaults to None (self.n_list).
            pol (str, optional): Polarization string: 'TE' or 'TM'. Defaults to 'TE'.

        Returns:
            array: 2x2 T-matrix

        """
        d_list = self.d_list if d_list is None else d_list
        n_list = self.n_list if n_list is None else n_list
        INT_func = self.T_matrix_TE if pol == "TE" else self.T_matrix_TM
        T = INT_func(n_list[0], n_list[1], om, k)
        for n1, d1, n2 in zip(n_list[1:-1], d_list[1:-1], n_list[2:]):
            T = np.dot(self.T_prop(n1, om, k, d1), T)
            T = np.dot(INT_func(n1, n2, om, k), T)
        return T

    def max_guided_modes(self, om, d_list=None, n_list=None):
        d_list = self.d_list if d_list is None else d_list
        n_list = self.n_list if n_list is None else n_list
        n_core = max(n_list)
        n_clad = min(n_list[0], n_list[-1])
        t = sum(d_list[1:-1])
        V = np.sqrt((n_core**2.0 - n_clad**2.0) * (0.5 * om * t) ** 2.0)
        return int(2.0 * V / np.pi)

    def _func_to_optimize(self, om, d_list=None, n_list=None, pol="TE"):
        """Creates function to be used for root finding

        Args:
            om (TYPE): Angular frequency.
            d_list (list, optional): List of floats. Thicknesses of the layers. Defaults to None (self.d_list).
            n_list (TYPE, optional): List of floats. Indexes of the layers. Defaults to None (self.n_list).
            pol (str, optional): Polarization string: 'TE' or 'TM'. Defaults to 'TE'.

        Returns:
            function: function returning t22 element of transfer matrix as a function of k

        """
        d_list = self.d_list if d_list is None else d_list
        n_list = self.n_list if n_list is None else n_list

        def wrapper(k):
            """Return t22 element of transfer matrix as a function of k

            Args:
                k (float): Parallel wavevector

            Returns:
                array: array of real and imaginary part of t22

            """
            T = self.generalT(k, om, d_list, n_list, pol)
            return T[1, 1].real

        return wrapper

    def find_neff(self, om, pol="TE", mode=0, d_list=None, n_list=None):
        """Find effective index via root searching

        Returns 0.0 if the selected mode does not exsists

        Args:
            om (float): Angular velocity.
            pol (str, optional): Polarization string: 'TE' or 'TM'. Defaults to "TE".
            mode (int, optional): Mode index. Defaults to 0.
            d_list (list, optional): List of floats. Thicknesses of the layers. Defaults to None (self.d_list).
            n_list (TYPE, optional): List of floats. Indexes of the layers. Defaults to None (self.n_list).

        Returns:
            float: effective index

        """
        d_list = self.d_list if d_list is None else d_list
        n_list = self.n_list if n_list is None else n_list
        if len(n_list) == 1:
            return n_list[0]
        if len(n_list) == 2:
            raise ValueError(
                "Multilayer solver has only 2 layers, effective index cannot be calcuated."
            )
        hash = hash_call(om, pol, d_list, n_list)
        to_opt = self._func_to_optimize(om, d_list=d_list, n_list=n_list, pol=pol)
        if hash == self.last_call_hash:
            try:
                return self.res[mode]
            except KeyError:
                om_guess = self.__om_guess
        else:
            self.res = {}
            self.last_call_hash = hash
            nmin, nmax = np.min(n_list), np.max(n_list)
            N_max = self.max_guided_modes(om, d_list=d_list, n_list=n_list)
            om_array = np.linspace(
                om * nmin + 1e-14, om * nmax - 1e-14, max(int(N_max * N_max), 101)
            )
            om_guess = self.approximate_zeros(to_opt, om_array)
            om_guess = om_guess[::-1]
            self.__om_guess = om_guess
        if mode is None:
            self.res = {
                i: opt.root(
                    to_opt, k0, method=self.root_method, options=self.root_options
                ).x[0]
                / om
                for i, k0 in enumerate(om_guess)
            }
            return list(self.res.values())
        try:
            self.res[mode] = (
                opt.root(
                    to_opt,
                    om_guess[mode],
                    method=self.root_method,
                    options=self.root_options,
                ).x[0]
                / om
            )
            return self.res[mode]
        except IndexError:
            return 0.0

    def _plot_slice_field(self, u, d, om, k, n, x):
        """Return main components of the files in a single layer

        Args:
            u (float): Forward propagating wave amplitude.
            d (float): Backward propagating wave amplitude.
            om (float): Angular velocity.
            k (float): Parallel wavevector.
            n (float): Index of the layer.
            x (array): Coordinates for field calculation

        Returns:
            array: Main field component

        """
        b = np.sqrt((1.0 + 0.0j) * n**2.0 * om**2.0 - (1.0 + 0.0j) * k**2.0)
        field = u * np.exp(1.0j * b * x) + d * np.exp(-1.0j * b * x)
        return field

    def _get_x_grid(self, d_list=None):
        """Create x-vectors for filed calculation and plotting

        Args:
           d_list (list, optional): List of floats. Thicknesses of the layers. Defaults to None (self.d_list).

        Returns:
            x_out (list): list of arrays with plotting coordinates for each layer
            x_calc (list): list of arrays with coordinates for field calculation for each layer

        """
        d_list = self.d_list if d_list is None else d_list
        x_cum = np.concatenate([[0.0], np.cumsum(d_list)]) - d_list[0]
        x_tot = np.linspace(x_cum[0], x_cum[-1], self.plot_points)
        x_out, x_calc = [], []
        x_out.append(x_tot[x_tot <= x_cum[1]])
        x_calc.append(x_tot[x_tot <= x_cum[1]])
        for x1, x2 in zip(x_cum[1:], x_cum[2:]):
            x_out.append(x_tot[np.logical_and(x_tot > x1, x_tot <= x2)])
            x_calc.append(x_out[-1] - x1)
        return x_out, x_calc

    def get_index(self, d_list=None, n_list=None):
        d_list = self.d_list if d_list is None else d_list
        n_list = self.n_list if n_list is None else n_list
        x_out, x_calc = self._get_x_grid(d_list)
        nl = [np.zeros_like(x) + n for x, n in zip(x_out, n_list)]
        data = {
            "x": np.concatenate(x_out),
            "index": np.concatenate(nl),
        }
        return data

    @staticmethod
    def _normalize_fields(data):
        """Normilize fields

        Normalization is set the integrated Poynting vector along z direction to 1.0

        Args:
            data (dict): Dictionary with the following fields:
                - 'x': x coordinates
                - 'Ex', 'Ey', and 'Ex': electric field
                - 'Hx', 'Hy', and 'Hx': magnetic field
        Returns:
            dict: same dictionary as input, but with normalized fields

        """
        x = data["x"]
        Pv = data["Ex"] * np.conj(data["Hy"]) - data["Ey"] * np.conj(data["Hx"])
        P = simpson(Pv, x=x)
        for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            data[comp] = data[comp] / np.sqrt(P)
        return data

    def get_field(self, om, pol="TE", mode=0, d_list=None, n_list=None, d=1.0):
        """


        Args:
            om (float): Angular velocity.
            pol (str, optional): Polarization string: 'TE' or 'TM'. Defaults to "TE".
            mode (int, optional): Mode index. Defaults to 0.
            d_list (list, optional): List of floats. Thicknesses of the layers. Defaults to None (self.d_list).
            n_list (TYPE, optional): List of floats. Indexes of the layers. Defaults to None (self.n_list).
            d (float, optional): Backward amplitude in the first layer. Defaults to 1.0.

        Returns:
            dict: Dictionary with field info: x coordinte and field components

        """
        d_list = self.d_list if d_list is None else d_list
        n_list = self.n_list if n_list is None else n_list
        INT_func = self.T_matrix_TE if pol == "TE" else self.T_matrix_TM
        xo, xp = self._get_x_grid(d_list)
        b = self.find_neff(om, pol=pol, mode=mode, d_list=d_list, n_list=n_list) * om
        if b == 0.0:
            return 0.0
        Em, Em2, Eb, Ek, Em3 = [], [], [], [], []
        # xm.append(np.linspace(-d_list[0], 0.0, self.plot_points))
        Em.append(self._plot_slice_field(0.0, d, om, b, n_list[0], xp[0]))
        k = np.sqrt((1.0 + 0.0j) * ((n_list[0] * om) ** 2.0 - b**2.0))
        coeff = 1.0j * (
            np.sqrt((1.0 + 0.0j) * b**2.0 / ((n_list[0] * om) ** 2.0 - b**2.0))
        )
        Em2.append(self._plot_slice_field(0.0, d * b / k, om, b, n_list[0], xp[0]))
        Eb.append(self._plot_slice_field(0.0, -d * b / om, om, b, n_list[0], xp[0]))
        Ek.append(self._plot_slice_field(0.0, -d * k / om, om, b, n_list[0], xp[0]))
        Em3.append(
            self._plot_slice_field(
                0.0, d * om * n_list[0] ** 2.0 / k, om, b, n_list[0], xp[0]
            )
        )
        p_list = []
        p = np.array([0.0, d])
        p_list.append(p)
        for n1, d2, n2, x in zip(n_list[:-1], d_list[1:], n_list[1:], xp[1:]):
            p = np.dot(INT_func(n1, n2, om, b), p)
            p_list.append(p)
            p = np.dot(self.T_prop(n2, om, b, d2), p)
        p_list[-1][1] = 0.0

        for p, n1, d2, n2, x in zip(
            p_list[1:], n_list[:-1], d_list[1:], n_list[1:], xp[1:]
        ):
            print("p", p)
            Em.append(self._plot_slice_field(p[0], p[1], om, b, n2, x))
            k = np.sqrt((1.0 + 0.0j) * ((n2 * om) ** 2.0 - b**2.0))
            Em2.append(
                self._plot_slice_field(-p[0] * b / k, p[1] * b / k, om, b, n2, x)
            )
            Eb.append(
                self._plot_slice_field(-p[0] * b / om, -p[1] * b / om, om, b, n2, x)
            )
            Ek.append(
                self._plot_slice_field(p[0] * k / om, -p[1] * k / om, om, b, n2, x)
            )
            Em3.append(
                self._plot_slice_field(
                    -p[0] * om * n2**2.0 / k, p[1] * om * n2**2.0 / k, om, b, n2, x
                )
            )

            # p = np.dot(self.T_prop(n2, om, b, d2), p)
        xo, Em, Em2, Eb, Ek, Em3 = (
            np.concatenate(xo),
            np.concatenate(Em),
            np.concatenate(Em2),
            np.concatenate(Eb),
            np.concatenate(Ek),
            np.concatenate(Em3),
        )
        zeros = np.zeros_like(Em2)
        if pol == "TE":
            data = {
                "x": xo,
                "Ex": zeros,
                "Ey": Em,
                "Ez": zeros,
                "Hx": Eb / self.Z0,
                "Hy": zeros,
                "Hz": Ek / self.Z0,
            }
        else:
            data = {
                "x": xo,
                "Ex": Em2,
                "Ey": zeros,
                "Ez": Em,
                "Hx": zeros,
                "Hy": Em3 / self.Z0,
                "Hz": zeros,
            }
        return Multilayer._normalize_fields(data)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    layers = [
        (1.0, 1.0),
        (2.0, 0.6),
        (1.0, 1.0),
    ]

    multi = Multilayer(*zip(*layers))
    print(multi.find_neff(2.0 * np.pi / 1.55, pol="TE", mode=0))
    field = multi.get_field(2.0 * np.pi / 1.55, pol="TE", mode=0)
    for component in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        comp = field[component]
        plt.plot(field["x"], np.real(comp) + np.imag(comp), label=component)
    plt.legend()
    plt.show()
