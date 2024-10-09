from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# import warnings

# warnings.filterwarnings('error')


class Timer:
    def __init__(self):
        self.start = None
        self.end = None

    @property
    def elapsed(self):
        return self.end - self.start

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time()


def compose(S, M):
    C = np.zeros((2, 2), dtype=complex)
    T1 = M[0, 0] / (1.0 - S[0, 1] * M[1, 0])
    T2 = S[1, 1] / (1.0 - M[1, 0] * S[0, 1])

    C[0, 0] = T1 * S[0, 0]
    C[1, 1] = T2 * M[1, 1]
    C[0, 1] = M[0, 1] + T1 * S[0, 1] * M[1, 1]
    C[1, 0] = S[1, 0] + T2 * M[1, 0] * S[0, 0]

    return C


def mirror(r, t):
    C = np.zeros((2, 2), dtype=complex)
    C[0, 0] = t
    C[1, 1] = t
    C[0, 1] = r
    C[1, 0] = r
    return C


def prop(n, d, om, k):
    a = np.sqrt((1 + 0j) * ((om * n) ** 2 - k**2))
    #    print a
    C = np.zeros((2, 2), dtype=complex)
    C[1, 1] = np.exp((0 + 1j) * a * d)
    C[0, 0] = np.exp((0 + 1j) * a * d)
    return C


def interfaceTE(n1, n2, om, k):
    k1 = np.sqrt((1 + 0j) * ((om * n1) ** 2 - k**2))
    k2 = np.sqrt((1 + 0j) * ((om * n2) ** 2 - k**2))
    C = np.zeros((2, 2), dtype=complex)
    C[1, 0] = k1 - k2
    C[0, 1] = k2 - k1
    C[1, 1] = 2.0 * k1
    C[0, 0] = 2.0 * k2
    return C / (k1 + k2)


def interfaceTM(n1, n2, om, k):
    k1 = n2**2 * np.sqrt((1 + 0j) * ((om * n1) ** 2 - k**2))
    k2 = n1**2 * np.sqrt((1 + 0j) * ((om * n2) ** 2 - k**2))
    C = np.zeros((2, 2), dtype=complex)
    C[1, 0] = k1 - k2
    C[0, 1] = k2 - k1
    C[1, 1] = 2.0 * k1
    C[0, 0] = 2.0 * k2
    return C / (k1 + k2)


def T_matrix_TE(n1, n2, om, k):
    k1 = np.sqrt((1 + 0j) * ((om * n1) ** 2 - k**2))
    k2 = np.sqrt((1 + 0j) * ((om * n2) ** 2 - k**2))
    C = np.zeros((2, 2), dtype=complex)
    C[0, 1] = k2 - k1
    C[1, 0] = k2 - k1
    C[0, 0] = k2 + k1
    C[1, 1] = k2 + k1
    return C / (2.0 * k2)


def T_matrix_TM(n1, n2, om, k):
    k1 = np.sqrt((1 + 0j) * ((om * n1) ** 2 - k**2))
    k2 = np.sqrt((1 + 0j) * ((om * n2) ** 2 - k**2))
    C = np.zeros((2, 2), dtype=complex)
    C[0, 1] = -(n1**2) * k2 + n2**2 * k1
    C[1, 0] = -(n1**2) * k2 + n2**2 * k1
    C[0, 0] = n1**2 * k2 + n2**2 * k1
    C[1, 1] = n1**2 * k2 + n2**2 * k1
    return C / (2.0 * k1 * n2 * n2)


def T_prop(n, om, k, d):
    w = np.sqrt((1 + 0j) * ((om * n) ** 2 - k**2)) * d
    C = np.zeros((2, 2), dtype=complex)
    C[0, 0] = np.exp((0 + 1j) * w)
    C[1, 1] = np.exp(-(0 + 1j) * w)
    return C


def generalT(k, om, d_list, n_list, pol="TE"):
    INT_func = T_matrix_TE if pol == "TE" else T_matrix_TM
    T = INT_func(n_list[0], n_list[1], om, k)
    for n1, d1, n2 in zip(n_list[1:-1], d_list[1:-1], n_list[2:]):
        T = np.dot(T, T_prop(n1, om, k, d1))
        T = np.dot(T, INT_func(n1, n2, om, k))
    return T


# def band(om,k):
#    s1=0.4
#    s2=0.4
#    n1=1.0
#    n2=3.5
#    I1=interfaceTE(n1,n2,om,k)
#    I2=interfaceTE(n2,n1,om,k)
#    P1=prop(n1,s1,om,k)
#    P2=prop(n2,s2,om,k)
#    T=compose(compose(compose(I1,P2),I2),P1)
#    A=np.zeros((2,2),dtype=complex)
#    B=np.zeros((2,2),dtype=complex)
#    A[0,0],A[0,1],A[1,0],A[1,1]=T[0,0],1.0,T[1,0],0.0
#    B[0,0],B[0,1],B[1,0],B[1,1]=0.0,T[0,1],-1,T[1,1]
#    try:
#        [E,EV]=la.eig(A,b=B)
#    except RuntimeWarning:
#        print 'Error with om= ',om
#        quit()
#
#    return E


def dec(func):
    def ff():
        R = func
        print(R)
        return R

    return ff


def slab(k, om, ncore, nclad, s):
    I1 = interfaceTM(nclad, ncore, om, k)
    P = prop(ncore, s, om, k)
    I2 = interfaceTM(ncore, nclad, om, k)
    TOT = compose(compose(I1, P), I2)
    return 1.0 / np.linalg.det(TOT)
    # return 1.0/np.abs(TOT[0,1])
    # return TOT
    # print(len(np.array([RR.real,RR.imag]))
    # return np.array([RR.real,RR.imag])


def plasm(om, k):
    n = np.sqrt((1 + 0j) * (1.0 - 1.0 / om**2))
    I1 = interfaceTM(1.0, n, om, k)
    #    I2=interfaceTM(n,2.0,om,k)
    #    P=prop(n,5.0,om,k)
    #    TOT=compose(compose(I1,P),I2)
    try:
        return np.abs(1.0 / np.abs(I1[0, 1]))
    except RuntimeWarning:
        return 10.0


def general(k, om, d_list, n_list, pol="TE"):
    if pol == "TM":
        int_func = interfaceTM
    elif pol == "TE":
        int_func = interfaceTE
    else:
        raise ValueError("pol has to be TE or TM")
    if len(d_list) < 3 or len(n_list) < 3:
        raise ValueError("at least 3 layer needed")
    if len(d_list) != len(n_list):
        raise ValueError("d_list and n_list should have the same length")
    T = int_func(n_list[0], n_list[1], om, k)
    for d, n1, n2 in zip(d_list[1:-1], n_list[1:-1], n_list[2:]):
        Tn = prop(n1, d, om, k)
        T = compose(T, Tn)
        Tn = int_func(n1, n2, om, k)
        T = compose(T, Tn)
    return T


def func_to_optimize(om, d_list, n_list, pol="TE"):
    def wrapper(k):
        T = generalT(k, om, d_list, n_list, pol)

        return np.array([T[1, 1].real, T[1, 1].imag])

    return wrapper


def func_determinat(om, d_list, n_list, pol="TE"):
    def inner(k):
        T = generalT(k, om, d_list, n_list, pol)
        return T
        # return np.log10(np.abs(T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]))

    return inner


class MultiLayer:
    step_settings = [
        (1e-2, 0.0),
        (1e-3, 0.0),
        (1e-4, 0.0),
        (1e-5, 0.0),
        (1e-6, 3.0),
        (1e-7, 3.0),
        (1e-8, 3.0),
        (1e-9, 6.0),
    ]

    def __init__(self, n_list, d_list) -> None:
        self.n_list = n_list
        self.d_list = d_list

    def _transfer_char(
        self, k: float, om: float, d_list: list[float], n_list: list[float], pol: str
    ):
        return 1.0 / generalT(k, om, d_list, n_list, pol)[1, 1]

    def _solve_step(
        self,
        kmin: float,
        kmax: float,
        step: float,
        treshold: float,
        om: float,
        pol: str,
    ) -> list[tuple[float, float]]:
        kv = np.arange(kmin, kmax, step)
        caracteristic = [
            np.log10(np.abs(self._transfer_char(_, om, self.d_list, self.n_list, pol)))
            for _ in kv
        ]
        peaks, extra = find_peaks(caracteristic, treshold)
        ksol = [kv[_] for _ in peaks]
        ktuple = [(_ - step, _ + step) for _ in ksol]
        return ktuple

    def solve(self, om, pol):
        brackets = [
            (min(self.n_list) * om + 1e-9, max(self.n_list) * om - 1e-9),
        ]

        for step, treshold in self.step_settings:
            new_brackets = []
            for kmin, kmax in brackets:
                new_brackets += self._solve_step(kmin, kmax, step, treshold, om, pol)
            brackets = new_brackets

        ksol = [0.5 * (_[0] + _[1]) for _ in brackets]
        nsol = [_ / om for _ in ksol]
        return nsol


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    n_solutions = [
        1.807131772097367,
        1.804297363810912,
        1.801441846494003,
        1.220438775992702,
        1.191174978673228,
        1.143946222685255,
    ]

    # d_list = [1.0, 0.6, 1.0]
    # n_list = [1.0, 2.0, 1.0]
    d_list = [1.0, 0.6, 2.0, 0.6, 1.0]
    n_list = [1.0, 2.0, 1.0, 2.0, 1.0]

    om = 2.0 * np.pi / 1.55

    multi = MultiLayer(n_list, d_list)

    with Timer() as t:
        print(multi.solve(om, "TE"))
    print(t.elapsed)

    quit()

    # k_solutions = [_ * om for _ in n_solutions]

    det = func_determinat(om, d_list, n_list, "TM")
    kl = np.arange(om, 2.0 * om, 1e-2)
    Ss = [det(k) for k in kl]

    # plt.plot(kl, [np.log10(np.abs(ln.det(_))) for _ in Ss], label="det")
    plt.plot(kl, [np.log10(np.abs(1.0 / _[0, 0])) for _ in Ss], label="0,0")
    plt.plot(kl, [np.log10(np.abs(1.0 / _[0, 1])) for _ in Ss], label="0,1")
    plt.plot(kl, [np.log10(np.abs(1.0 / _[1, 0])) for _ in Ss], label="1,0")
    plt.plot(kl, [np.log10(np.abs(1.0 / _[1, 1])) for _ in Ss], label="1,1")
    plt.legend()
    plt.show()

    # S = general(k_solutions[2], om, d_list, n_list, "TM")
    # print("S matrix")
    # print(S)
    # print(np.abs(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]))

    # print("Inverse S matrix")
    # I = ln.inv(S)
    # print(I)
    # eig, ev = ln.eig(I)

    # print("eigenvalues and eigenvectors")
    # print(eig)
    # print(ev)

    # print("Singular value decomposition")

    # U, s, V = ln.svd(I)
    # print(U)
    # print(s)
    # print(V)
