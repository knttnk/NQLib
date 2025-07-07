import math
import time
import warnings
from enum import Enum as _Enum
from enum import auto as _auto
from typing import Callable, Tuple, Union, List

import control as _ctrl
from packaging.version import Version

if Version(_ctrl.__version__) >= Version("0.9.2"):
    ctrl_poles = _ctrl.poles
    ctrl_zeros = _ctrl.zeros
else:
    ctrl_poles = _ctrl.pole  # type: ignore
    ctrl_zeros = _ctrl.zero  # type: ignore
    _ctrl.use_numpy_matrix(False)  # type: ignore


import cvxpy
import numpy as _np
from scipy.optimize import differential_evolution as _differential_evolution
from scipy.optimize import minimize as _minimize
from scipy.special import comb as _comb
from .linalg import (block, eig_max, eye, kron, matrix, mpow, norm, ones, pinv,
                     zeros)
from .types import (
    NDArrayNum, Real, infint, InfInt,
    validate_int, validate_int_or_inf, validate_float
)

__all__ = [
    'StaticQuantizer',
    'DynamicQuantizer',
    'order_reduced',
]


class ConnectionType(_Enum):
    """
    Enum for system connection types.

    Attributes
    ----------
    FF : enum
        Feedforward connection.
    FB_WITH_INPUT_QUANTIZER : enum
        Feedback with input quantizer.
    FB_WITH_OUTPUT_QUANTIZER : enum
        Feedback with output quantizer.
    ELSE : enum
        Other connection types.
    """
    FF = _auto()
    FB_WITH_INPUT_QUANTIZER = _auto()
    FB_WITH_OUTPUT_QUANTIZER = _auto()
    ELSE = _auto()


class StaticQuantizer():
    """
    Static quantizer.
    """

    def __init__(self,
                 function: Callable[[NDArrayNum], NDArrayNum],
                 delta: float,
                 *,
                 error_on_excess: bool = True):
        """
        Initialize a StaticQuantizer `q`.

        Parameters
        ----------
        function : Callable[[NDArrayNum], NDArrayNum]
            Quantization function. Must be callable.
        delta : float
            The maximum allowed quantization error. Declares that for any real vector `u`, max(|q(u)-u|) <= `delta`. `delta` > 0.
        error_on_excess : bool, optional
            If True, raise error when error exceeds `delta` (default: True).
            i.e. whether to raise an error when max(|q(u)-u|) > `delta` becomes True.

        Raises
        ------
        TypeError
            If `function` is not callable.
        ValueError
            If quantization error exceeds `delta` and `error_on_excess` is True.
        """
        self.delta = validate_float(
            delta,
            infimum=0.0,  # delta must be greater than 0
        )

        # check and assign `function`
        if not callable(function):
            raise TypeError('`function` must be a callable object.')
        else:
            if error_on_excess:
                def safe_function(u: NDArrayNum) -> NDArrayNum:
                    # returns function(u)
                    v = function(u)
                    if _np.max(_np.abs(v - u)) > delta:
                        raise ValueError(
                            "During the simulation, `numpy.max(abs(q(u) - u))` "
                            "exceeded `delta`. Check the definition of "
                            "your static quantizer.\n"
                            "If this excess is intentional, set the "
                            "argument `error_on_excess` of "
                            "`StaticQuantizer()` to `False`.\n"
                            f"`q(u)` = \n{v}\nwhen `u` = \n{u}\n"
                            f"and `delta` = {delta}"
                        )
                    else:
                        return v
                self._function = safe_function
            else:
                self._function = function

    def __str__(self):
        return f"q(*; delta={self.delta})"

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"  {self._function},\n"
            f"  {self.delta},\n"
            f")"
        )

    def _repr_latex_(self):
        return r"$$q(*;\ " + f"\\Delta={self.delta}" + ")$$"

    def __call__(self, u: NDArrayNum) -> NDArrayNum:
        """
        Call quantize method.

        Parameters
        ----------
        u : NDArrayNum
            Input signal.

        Returns
        -------
        NDArrayNum
            Quantized signal.
        """
        return self.quantize(u)

    def quantize(self, u: NDArrayNum) -> NDArrayNum:
        """
        Quantize the input signal.

        Parameters
        ----------
        u : NDArrayNum
            Input signal.

        Returns
        -------
        NDArrayNum
            Quantized signal.
        """
        return self._function(u)

    @staticmethod
    def mid_tread(d: float,
                  bit: Union[InfInt, int] = infint,
                  *,
                  error_on_excess: bool = True) -> "StaticQuantizer":
        """
        Create a mid-tread uniform StaticQuantizer.

        Parameters
        ----------
        d : float
            Quantization step size. For a real vector u, max(|q(u)-u|) <= d/2. `delta` > 0.
        bit : int or InfInt, optional
            Number of bits. Must satisfy `bit` >= 1 (default: `infint`).
            That is, the returned function can take `2**n` values.
        error_on_excess : bool, optional
            If True, raise error when error exceeds `delta` (=d/2) (default: True).
            i.e. whether to raise an error when max(|q(u)-u|) > `delta` becomes `True`.
            Basically, this error should not occur, but for numerical safety, set this to True.

        Returns
        -------
        q : StaticQuantizer
            Mid-tread quantizer instance.
        """
        try:
            if d <= 0:
                raise ValueError('`d` must be greater than `0`.')
        except TypeError:
            raise TypeError('`d` must be a real number.')
        d = validate_float(
            d,
            infimum=0.0,  # d must be greater than 0
            name="d",
        )
        # function to quantize

        def q(u: NDArrayNum) -> NDArrayNum:
            return ((u + d / 2) // d) * d

        # limit the values
        bit = validate_int_or_inf(
            bit,
            minimum=1,
            name="bit",
        )
        if bit is infint:
            function = q
        else:
            def function(u: NDArrayNum) -> NDArrayNum:
                return _np.clip(
                    q(u),
                    a_min=-(2**(bit - 1) - 1) * d,
                    a_max=2**(bit - 1) * d,
                )

        return StaticQuantizer(
            function=function,
            delta=d / 2,
            error_on_excess=error_on_excess,
        )

    @staticmethod
    def mid_riser(d: float,
                  bit: int | InfInt = infint,
                  *,
                  error_on_excess: bool = True) -> "StaticQuantizer":
        """
        Create a mid-riser uniform StaticQuantizer.

        Parameters
        ----------
        d : float
            Quantization step size. For a real vector u, max(|q(u)-u|) <= d/2. `delta` > 0.
        bit : int or InfInt, optional
            Number of bits. Must satisfy `bit` >= 1 (default: `infint`).
            That is, the returned function can take `2**n` values.
        error_on_excess : bool, optional
            If True, raise error when error exceeds `delta` (=d/2) (default: True).
            i.e. whether to raise an error when max(|q(u)-u|) > `delta` becomes `True`.
            Basically, this error should not occur, but for numerical safety, set this to True.

        Returns
        -------
        StaticQuantizer
            Mid-riser quantizer instance.
        """
        d = validate_float(
            d,
            infimum=0.0,  # d must be greater than 0
            name="d",
        )

        # function to quantize
        def q(u: NDArrayNum) -> NDArrayNum:
            return (u // d + 1 / 2) * d

        # limit the values
        bit = validate_int_or_inf(
            bit,
            minimum=1,
            name="bit",
        )
        if bit is infint:
            function = q
        else:
            def function(u: NDArrayNum) -> NDArrayNum:
                a_max = (2**(bit - 1) - 1 / 2) * d
                return _np.clip(
                    q(u),
                    a_min=-a_max,
                    a_max=a_max,
                )

        return StaticQuantizer(
            function=function,
            delta=d / 2,
            error_on_excess=error_on_excess,
        )


def _find_tau(A_tilde: NDArrayNum, C_1: NDArrayNum, B_2: NDArrayNum, l: int, tau_max: int) -> int:
    """
    Find the smallest integer tau such that
    C_1 @ mpow(A_tilde, tau) @ B_2 is nonzero, tau>=0.

    Parameters
    ----------
    A_tilde : NDArrayNum
        Closed-loop system matrix.
    C_1 : NDArrayNum
        Output matrix.
    B_2 : NDArrayNum
        Input matrix.
    l : int
        Output dimension.
    tau_max : int
        Maximum tau to check.

    Returns
    -------
    int
        tau if found, else -1.
    """
    def is_not_zero(M: NDArrayNum) -> bool:
        return (M * 0 != M).all()

    for t in range(tau_max):
        M = C_1 @ mpow(A_tilde, t) @ B_2
        if is_not_zero(M):
            if _np.linalg.matrix_rank(M) == l:
                return t
            else:
                return -1
    return -1


def _nq_serial_decomposition(system: "System",  # type: ignore
                             q: StaticQuantizer,
                             verbose: bool) -> Tuple["DynamicQuantizer | None", float]:
    """
    Finds the stable and optimal dynamic quantizer for `system`
    using serial decomposition[1]_.

    Parameters
    ----------
    system : System
    q : StaticQuantizer
    T : int
    gain_wv : float
    verbose : bool
    dim : int

    Returns
    -------
    (Q, E) : Tuple[DynamicQuantizer, float]
    """
    if verbose:
        print("Trying to calculate quantizer using serial system decomposition...")
    P = system.P
    if P is None:
        if verbose:
            print("P is None. Couldn't calculate by using serial system decomposition.")
        return None, _np.inf
    tf = P.tf1
    zeros_ = ctrl_zeros(tf)
    poles = ctrl_poles(tf)

    unstable_zeros = [zero for zero in zeros_ if abs(zero) > 1]

    z = _ctrl.TransferFunction([1, 0], [1], tf.dt)

    if len(unstable_zeros) == 0:
        n_time_delay = len([p for p in poles if p == 0])  # count pole-zero
        G = 1 / z**n_time_delay  # type: ignore
        F = tf / G
        F_ss: _ctrl.StateSpace = _ctrl.tf2ss(F)  # type: ignore
        B_F = matrix(F_ss.B)
        C_F = matrix(F_ss.C)

        E = abs(C_F @ B_F)[0, 0] * q.delta
    elif len(unstable_zeros) == 1:
        i = len(poles) - len(zeros_)  # relative order
        if i < 1:
            return None, _np.inf
        a = unstable_zeros[0]
        a = _np.real_if_close(a)
        assert a.imag == 0, "Unstable zero must be real."
        G = (z - a) / z**i
        F = _ctrl.minreal(tf / G, verbose=False)
        F = _ctrl.tf(_np.real(F.num), _np.real(F.den), F.dt)
        F_ss: _ctrl.StateSpace = _ctrl.tf2ss(F)  # type: ignore
        B_F = matrix(F_ss.B)
        C_F = matrix(F_ss.C)

        E = (1 + abs(a)) * abs(C_F @ B_F)[0, 0] * q.delta
    else:
        return None, _np.inf

    A_F = matrix(F_ss.A)
    D_F = matrix(F_ss.D)

    # check
    if (C_F @ B_F)[0, 0] == 0:
        if verbose:
            print("CF @ BF == 0 became true. Couldn't calculate by using serial system decomposition.")
        return None, _np.inf
    if D_F[0, 0] != 0:
        if verbose:
            print("DF == 0 became true. Couldn't calculate by using serial system decomposition.")
        return None, _np.inf
    Q = DynamicQuantizer(
        A=A_F,
        B=B_F,
        C=- 1 / (C_F @ B_F)[0, 0] * C_F @ A_F,
        q=q
    )
    if verbose:
        print("Success!")
    return Q, E


def _SVD_from(H2: List[NDArrayNum],
              T: int) -> Tuple[NDArrayNum, NDArrayNum, NDArrayNum, NDArrayNum]:
    T_dash = math.floor(T / 2) + 1

    if T % 2 == 0:
        H2 = H2 + [_np.zeros_like(H2[0])]

    # H⚫
    # [H2_0     , H2_1     , H2_2     , ...      , H2_T_dash;
    #  H2_1     , H2_2     , ...      ,          ,     :
    #  H2_2     , ...      ,          ,                :
    #   :
    #   :
    #  H2_T_dash, ...      ,          , ...      , H2_2T_dash(or 0)]
    H_list = [
        [H2[r + c] for c in range(T_dash)]
        for r in range(T_dash)
    ]
    H = block(H_list)

    # singular value decomposition
    # find Wo, S, Wc, with which it becomes H = Wo S Wc
    Wo, S_vector, Wc = _np.linalg.svd(H)

    Wo = matrix(Wo)
    S = matrix(_np.diag(S_vector))
    Wc = matrix(Wc)

    # if norm(H - Wo @ S @ Wc) / norm(H) > 0.01:
    if norm(H - Wo @ S @ Wc) / norm(H) > 0.01:
        raise ValueError('SVD failed. Try another method.')

    return H, Wo, S, Wc


def _compose_Q_from_SVD(
    system: "System",  # type: ignore
    q: StaticQuantizer,
    T: int,
    H: _np.ndarray,
    Wo: _np.ndarray,
    S: _np.ndarray,
    Wc: _np.ndarray,
    dim: int | InfInt,
) -> Tuple["DynamicQuantizer", bool]:
    """
    Parameters
    ----------
    system : System
    q : StaticQuantizer
    T : int
        Must be an odd number.
    H : np.matrix
    Wo : np.matrix
    S : np.matrix
    Wc : np.matrix
    dim : int or np.inf

    Returns
    -------
    Tuple[DynamicQuantizer, bool]
        Quantizer is absolutely stable.
        Returned bool means if order is reduced.
    """
    m = system.m
    T_dash = int(H.shape[0] / m)

    # set order
    reduced = False
    if dim < m * T_dash:  # needs reduction
        nQ: int = dim  # type: ignore  # This will not occur if dim is infint.
        reduced = True
    else:
        nQ = m * T_dash

    # reduce order
    Wo_reduced = Wo[0:, :nQ]
    S_reduced = S[:nQ, :nQ]
    Wc_reduced = Wc[:nQ, 0:]
    # H_reduced = Wo_reduced @ S_reduced @ Wc_reduced

    # compose Q
    S_r_sqrt = mpow(S_reduced, 1 / 2)
    B2 = S_r_sqrt @ Wc_reduced @ eye(m * T_dash, m)
    C = eye(m, m * T_dash) @ Wo_reduced @ S_r_sqrt
    _P = pinv(
        eye(m * (T_dash - 1), m * T_dash) @ Wo_reduced @ S_r_sqrt
    )
    A = _P @ eye(m * (T_dash - 1), m * T_dash, k=m) @ Wo_reduced @ S_r_sqrt - B2 @ C

    # check stability
    Q = DynamicQuantizer(A, B2, C, q)
    if Q.is_stable:
        return Q, reduced
    else:
        # unstable case
        nQ_bar = nQ * T

        A_bar = block([
            [zeros((nQ_bar - nQ, nQ)), kron(eye(T - 1), A + B2 @ C)],
            [kron(ones((1, T)), -B2 @ C)],
        ])
        B2_bar = block([
            [zeros((nQ_bar - nQ, m))],
            [B2],
        ])
        C_bar = kron(ones((1, T)), C)
        Q = DynamicQuantizer(A_bar, B2_bar, C_bar, q)
        return Q, reduced


class DynamicQuantizer():
    """
    Dynamic quantizer.

    Parameters
    ----------
    A : NDArrayNum
        State matrix (N x N, real or complex). N >= 1.
    B : NDArrayNum
        Input matrix (N x m, real or complex). m >= 1.
    C : NDArrayNum
        Output matrix (m x N, real or complex).
    q : StaticQuantizer
        Static quantizer instance.
    """

    def __init__(self, A: NDArrayNum, B: NDArrayNum, C: NDArrayNum, q: StaticQuantizer):
        """
        Initialize a DynamicQuantizer instance.

        The dynamic quantizer is defined by the following equations:

            Q : { xi(t+1) =    A xi(t) + B u(t)
                {   v(t)  = q( C xi(t) +   u(t) )

        Parameters
        ----------
        A : NDArrayNum
            State matrix (N x N, real). N >= 1.
        B : NDArrayNum
            Input matrix (N x m, real). m >= 1.
        C : NDArrayNum
            Output matrix (m x N, real).
        q : StaticQuantizer
            Static quantizer instance.

        Raises
        ------
        ValueError
            If matrix dimensions are inconsistent.
        """
        A_mat = matrix(A)
        B_mat = matrix(B)
        C_mat = matrix(C)

        self.N = A_mat.shape[0]
        self.m = B_mat.shape[1]

        if A_mat.shape != (self.N, self.N):
            raise ValueError("A must be a square matrix.")
        if B_mat.shape != (self.N, self.m):
            raise ValueError(
                "The number of rows in matrices `A` and `B` must be the same."
            )
        if C_mat.shape[0] != self.m:
            raise ValueError(
                "The number of columns in matrix `B` and "
                "the number of rows in matrix `C` must be the same."
            )
        if C_mat.shape[1] != self.N:
            raise ValueError(
                "The number of columns in matrices `A` and `C` must be the same."
            )
        if C_mat.shape != (self.m, self.N):
            raise ValueError("`C must be a 2D array.`")

        self.A = A_mat
        self.B = B_mat
        self.C = C_mat
        self.q = q
        self.delta = q.delta
        self.gain_uv = 1.0

    def _matrix_str(self,
                    index: int,
                    formatter: Callable[[_np.number], str] = lambda n: f"{n}",
                    sep: str = ", ",
                    linesep: str = "\n",
                    indent: str = "") -> str:
        """
        Format a matrix as a string for display.

        Parameters
        ----------
        index : int
            0 for A, 1 for B, 2 for C.
        formatter : Callable, optional
            Function to format each element (default: str).
        sep : str, optional
            Separator for elements (default: ', ').
        linesep : str, optional
            Line separator (default: '\n').
        indent : str, optional
            String to prepend to each row (default: '').

        Returns
        -------
        str
            Formatted matrix string.
        """
        mat = [self.A, self.B, self.C][index]
        ret = ""
        rows = mat.shape[0]
        cols = mat.shape[1]
        for r in range(rows):
            ret += indent
            for c in range(cols):
                ret += formatter(mat[r, c])
                if c < cols - 1:
                    ret += sep
            if r < rows - 1:
                ret += linesep
        return ret

    def _str(self, q_str: str) -> str:
        n_indents = 2
        linesep = "],\n"
        indent = " " * (n_indents + 1) + "["
        matrix_strs = [f"{self._matrix_str(i, linesep=linesep, indent=indent)}]],\n" for i in range(3)]
        matrix_strs = [s[:n_indents] + "[" + s[n_indents + 1:] for s in matrix_strs]
        return (
            f"{type(self).__name__}(\n" +
            matrix_strs[0] + matrix_strs[1] + matrix_strs[2] +
            " " * n_indents + f"{q_str},\n" +
            f")"
        )

    def __str__(self):
        n_indents = 2
        return self._str(str(self.q).replace("\n", "\n" + " " * n_indents))

    def __repr__(self):
        n_indents = 2
        return self._str(repr(self.q).replace("\n", "\n" + " " * n_indents))

    def _repr_latex_(self):
        n_indents = 2
        matrix_strs = [
            self._matrix_str(i,
                             sep=" & ",
                             linesep=" \\\\ \n",
                             indent=" " * n_indents * 2) + "\n"
            for i in range(3)
        ]
        return (
            r"$$\begin{cases}\begin{aligned}" + "\n" +
            r"  \xi(k+1) & =\begin{bmatrix}" + "\n" +
            matrix_strs[0] + "\n" +
            r"  \end{bmatrix} \xi(k) + \begin{bmatrix}" + "\n" +
            matrix_strs[1] + "\n" +
            r"  \end{bmatrix} u(k), \\" + "\n" +
            r"  v(k) & =q\left(\begin{bmatrix}" + "\n" +
            matrix_strs[2] + "\n" +
            r"  \end{bmatrix} \xi(k) + u(k);\ \ \Delta=" + f"{self.q.delta}" + r"\right)." + "\n" +
            r"\end{aligned}\end{cases}$$"
        )

    def gain_wv(self, steptime: int | InfInt = infint, verbose: bool = False) -> Real:
        """
        Compute the gain from w to v for this quantizer.

        Parameters
        ----------
        steptime : int or InfInt, optional
            The number of time steps to use for the gain calculation.
            If `infint`, calculation continues until convergence.
            `steptime` >= 1 (default: `infint`, which means until convergence).
        verbose : bool, optional
            If True, print progress information during calculation (default: False).

        Returns
        -------
        float
            Estimated gain w->v.

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        if verbose:
            print("Calculating gain w->v...")
        steptime = validate_int_or_inf(
            steptime,
            minimum=1,  # steptime must be greater than 0
            name="steptime",
        )

        if not self.is_stable:
            return _np.inf  # type: ignore
        else:  # stable case
            sum_Q_wv = eye(self.m)

            i = 0
            A_i = eye(self.N)  # (self.A + self.B@self.C)**i
            ret = float(0.0)
            while i < steptime:
                if verbose:
                    print(f"i = {i}, ret = {ret}")
                sum_Q_wv = sum_Q_wv + abs(self.C @ A_i @ self.B)
                ret_past = ret
                ret = norm(sum_Q_wv)
                if i > 0:
                    if abs(ret - ret_past) < 1e-8:
                        break
                i = i + 1
                A_i = A_i @ (self.A + self.B @ self.C)

            return ret  # type: ignore

    def _objective_function(
        self,
            system: "System",  # type: ignore
            *,
            T: int | InfInt = infint,  # TODO: これより下を反映
            gain_wv: Real = _np.inf,  # type: ignore
            obj_type: str = ["exp", "atan", "1.1", "100*1.1"][0],
    ) -> Real:
        """
        Objective function for numerical optimization.

        Parameters
        ----------
        system : System
            The system for which the quantizer is being optimized. Must be stable and SISO.
        T : int or InfInt, optional
            The number of time steps for estimation.
            `T` >= 1 (default: `infint`, which means until convergence).
        gain_wv : float, optional
            Upper limit for the w->v gain. `gain_wv` >= 0 (default: `np.inf`).
        obj_type : str, optional
            Objective function type. Must be one of ['exp', 'atan', '1.1', '100*1.1'] (default: 'exp').

        Returns
        -------
        float
            Objective value.

        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
        """
        T = validate_int_or_inf(
            T,
            minimum=1,  # T must be greater than 0
            name="T",
        )
        if system.m != 1:
            raise ValueError("`design_GD` and `design_DE` is currently supported for SISO systems only.")

        # Values representing the stability or gain_wv.y
        # if max(constraint_values) < 0, this quantizer satisfies the
        # constraints.
        constraint_values: List[Real] = [
            eig_max(self.A + self.B @ self.C) - 1,
        ]
        if not _np.isposinf(gain_wv):
            constraint_values.append(self.gain_wv(T) - gain_wv)

        max_v = _np.max(constraint_values)
        if max_v < 0:
            types = ["exp", "atan", "1.1", "100*1.1"]
            if obj_type == types[0]:
                return - _np.exp(- system.E(self))
            if obj_type == types[1]:
                return _np.arctan(system.E(self)) - _np.pi / 2
            if obj_type == types[2]:
                return - 1.1 ** (- system.E(self))
            if obj_type == types[3]:
                return - 10000 * _np.exp(- 0.01 * system.E(self))
            raise ValueError(
                f"`obj_type` must be one of {types}, but got {obj_type}."
            )
        else:
            return max_v

    def order_reduced(self, dim: int) -> "DynamicQuantizer":
        """
        Return a reduced-order quantizer.

        Parameters
        ----------
        dim : int
            Desired order (1 <= `dim` < `self.N`).

        Returns
        -------
        DynamicQuantizer
            Reduced-order quantizer.

        Raises
        ------
        ImportError
            If slycot is not installed.
        ValueError
            If `dim` is not in the valid range.

        Notes
        -----
        Note that the quantizer with the reduced order
        will generally have larger `E(Q)` and a larger
        `gain_wv` than those of the original quantizer. 
        You should check the performance and gain yourself.

        This function requires slycot. Please install it.
        """
        dim = validate_int(
            dim,
            minimum=1,
            maximum=self.N - 1,  # order must be less than N
            name="dim",
        )
        try:
            from slycot import ab09ad
        except ImportError:
            raise ImportError((
                "Reducing order of a quantizer requires slycot."
                " Please install it."
            ))
        # マルコフパラメータから特異値分解する
        _Nr, Ar, Br, Cr, _hsv = ab09ad(
            "D",  # means "Discrete time"
            "B",  # balanced (B) or not (N)
            "S",  # scale (S) or not (N)
            self.N,  # np.size(A,0)
            self.m,  # np.size(B,1)
            self.m,  # np.size(C,0)
            self.A, self.B, self.C,
            nr=dim,
            tol=0.0,  # type: ignore
        )
        return DynamicQuantizer(Ar, Br, Cr, self.q)

    @property
    def is_stable(self) -> bool:
        """
        Check if the quantizer is stable.

        Returns
        -------
        bool
            True if stable, False otherwise.

        References
        ----------
        .. [2]  S. Azuma, Y. Minami and T. Sugie: Optimal dynamic quantizers
           for feedback control with discrete-level actuators; Journal of 
           Dynamic Systems, Measurement, and Control, Vol. 133, No. 2, 021005
           (2011)
        """
        if eig_max(self.A + self.B @ self.C) > 1 - 1e-8:
            return False
        else:
            return True

    @property
    def minreal(self) -> "DynamicQuantizer":
        """
        Minimal realization of this quantizer.

        Returns
        -------
        DynamicQuantizer
            Minimal realization of this quantizer.
        """
        minreal_ss: _ctrl.StateSpace = _ctrl.ss(
            self.A,
            self.B,
            self.C,
            self.C @ self.B * 0,
            True,
        ).minreal()  # type: ignore
        return DynamicQuantizer(
            minreal_ss.A,  # type: ignore
            minreal_ss.B,  # type: ignore
            minreal_ss.C,  # type: ignore
            self.q,
        )

    @staticmethod
    def design(
        system: "System",  # type: ignore
        *,
        q: StaticQuantizer,
        T: int | InfInt = infint,
        gain_wv: float = _np.inf,
        dim: int | InfInt = infint,
        verbose: bool = False,
        use_analytical_method: bool = True,
        use_LP_method: bool = True,
        use_design_GB_method: bool = True,
        use_DE_method: bool = False,
        solver: str | None = None
    ) -> Tuple["DynamicQuantizer | None", float]:
        """
        Design a stable and optimal dynamic quantizer for a system.

        Parameters
        ----------
        system : System
            Stable system instance.
        q : StaticQuantizer
            Static quantizer instance.
        T : int or InfInt, optional
            Estimation time (default: infint). `T` >= 1.
        gain_wv : float, optional
            Upper limit of gain w->v (default: inf). `gain_wv` >= 0.
        dim : int or InfInt, optional
            Upper limit of quantizer order (default: infint). `dim` >= 1.
        verbose : bool, optional
            If True, print progress (default: False).
        use_analytical_method : bool, optional
            Use analytical method (default: True).
        use_LP_method : bool, optional
            Use LP method (default: True).
        use_design_GB_method : bool, optional
            Use gradient-based method (default: True).
        use_DE_method : bool, optional
            Use differential evolution method (default: False).
        solver : str or None, optional
            CVXPY solver name (default: None).

        Returns
        -------
        Q : DynamicQuantizer or None
            Designed quantizer or None if not found.
        E : float
            Estimated E(Q). If Q is None, E is inf.

        Raises
        ------
        ValueError
            If system is unstable.
        """
        def _print_report(Q: "DynamicQuantizer | None", method: str):
            if verbose:
                if Q is None:
                    print(
                        f"Using {method}, ",
                        "NQLib couldn't find the quantizer.\n",
                    )
                else:
                    print(
                        f"Using {method}, NQLib found the following quantizer.\n",
                        "Q:\n",
                        "A =\n",
                        f"{Q.A}\n",
                        "B =\n",
                        f"{Q.B}\n",
                        "C =\n",
                        f"{Q.C}\n",
                        f"E = {system.E(Q)}\n",
                        f"gain_wv = {Q.gain_wv()}\n",
                        "\n",
                    )
        # TODO: 引数とドキュメントを見直す
        # TODO: 最小実現する
        # check system
        if system.__class__.__name__ != "System":
            raise TypeError(
                '`system` must be an instance of `nqlib.System`.'
            )
        elif not system.is_stable:
            raise ValueError(
                "`system` you input is unstable. "
                "Please input stable system."
            )

        # check q
        if type(q) is not StaticQuantizer:
            raise TypeError(
                '`q` must be an instance of `nqlib.StaticQuantizer`.'
            )

        # check gain_wv
        gain_wv = validate_float(
            gain_wv,
            infimum=0.0,  # gain_wv must be greater than 0
            name="gain_wv",
        )

        # check dim
        dim = validate_int_or_inf(
            dim,
            minimum=1,  # order must be greater than 0
            name="dim",
        )

        # algebraically optimize
        if use_analytical_method:
            Q, E = DynamicQuantizer.design_AG(
                system,
                q=q,
                dim=dim,
                gain_wv=gain_wv,
                verbose=verbose,
            )
            _print_report(Q, "the analytical method")
            if Q is not None:
                return Q, E

        candidates: List[Tuple[DynamicQuantizer, float]] = []
        # numerically optimize
        T = validate_int_or_inf(
            T,
            minimum=1,  # T must be greater than 0
            name="T",
        )
        if use_LP_method:
            Q, E = DynamicQuantizer.design_LP(
                system,
                q=q,
                dim=dim,
                T=T,
                gain_wv=gain_wv,
                solver=solver,
                verbose=verbose,
            )
            _print_report(Q, "the LP method")
            if Q is not None:
                candidates.append((Q, E))
        if use_design_GB_method and isinstance(dim, int):
            Q, E = DynamicQuantizer.design_GD(
                system,
                q=q,
                dim=dim,
                T=T,
                gain_wv=gain_wv,
                verbose=verbose,
            )
            _print_report(Q, "the gradient based method")
            if Q is not None:
                candidates.append((Q, E))
        if use_DE_method and isinstance(dim, int):
            Q, E = DynamicQuantizer.design_DE(
                system,
                q=q,
                dim=dim,
                T=T,
                gain_wv=gain_wv,
                verbose=verbose,
            )
            _print_report(Q, "the gradient based method")
            if Q is not None:
                candidates.append((Q, E))

        # compare all candidates and return the best
        if len(candidates) > 0:
            Q, E = min(
                candidates,
                key=lambda c: c[1],
            )
            return Q, E
        else:
            if verbose:
                print(
                    "NQLib could not design a quantizer under these conditions. ",
                    "Please try different conditions.",
                )
            return None, _np.inf

    @staticmethod
    def design_AG(system: "System",  # type: ignore
                  *,
                  q: StaticQuantizer,
                  # TODO: 考慮できないパラメータを渡せないようにする
                  dim: int | InfInt = infint,
                  gain_wv: float = _np.inf,
                  allow_unstable: bool = False,
                  verbose: bool = False) -> Tuple["DynamicQuantizer | None", float]:
        """
        Algebraically design a stable and optimal dynamic quantizer for a system.

        Parameters
        ----------
        system : System
            Stable and SISO system instance.
        q : StaticQuantizer
            Static quantizer instance.
        dim : int or InfInt, optional
            Upper limit of quantizer order (default: infint). `dim` >= 1.
        gain_wv : float, optional
            Upper limit of gain w->v (default: inf). `gain_wv` >= 0.
        allow_unstable : bool, optional
            Allow unstable quantizer (default: False).
            It's recommended to set `verbose` to `True`, so that 
            you can remember the result is unstable.
            If this is `True`, design method in reference [3] will not be used.  # TODO: これを反映させる
        verbose : bool, optional
            If True, print progress (default: False).

        Returns
        -------
        Q : DynamicQuantizer or None
            Designed quantizer or None if not found.
        E : float
            Estimated E(Q). If Q is None, E is inf.

        Raises
        ------
        ValueError
            If system is unstable.
        """
        if verbose:
            print("Trying to calculate optimal dynamic quantizer...")

        A_tilde = system.A + system.B2 @ system.C2  # convert to closed loop

        def _Q(A_tilde: NDArrayNum) -> "DynamicQuantizer":
            return DynamicQuantizer(
                A=A_tilde,
                B=system.B2,
                C=-pinv(system.C1 @ mpow(A_tilde, tau) @ system.B2) @ system.C1 @ mpow(A_tilde, tau + 1),
                q=q,
            )
        if (
            not allow_unstable and  # If unstable Q is allowed, this method is not needed.
            (_P := system.P) is not None and
            (system.type == ConnectionType.FF and _P.tf1.issiso())
        ):
            # FF and SISO
            Q, E = _nq_serial_decomposition(system, q, verbose)
            if Q is not None:
                if Q.is_stable:
                    return Q, E
        if system.m >= system.l:
            # S2
            tau = _find_tau(A_tilde, system.C1, system.B2, system.l, 10000)

            if tau == -1:
                if verbose:
                    print("Couldn't calculate optimal dynamic quantizer algebraically. Trying another method...")
                return None, _np.inf

            Q = _Q(A_tilde)

            E: float = norm(abs(system.C1 @ mpow(A_tilde, tau) @ system.B2)) * q.delta  # type: ignore
            Q_gain_wv = Q.gain_wv()

            if not Q.is_stable:
                if allow_unstable:
                    if verbose:
                        print("The quantizer is unstable.")
                    E = _np.inf
                else:
                    if verbose:
                        print("The quantizer is unstable. Try other methods.")
                    return None, _np.inf
            if Q.N > dim:
                if verbose:
                    print(
                        f"The order of the quantizer {Q.N} is greater than {dim}, the value you specified. ",
                        "Try other methods.",
                    )
                return None, _np.inf
            elif Q_gain_wv > gain_wv:
                if verbose:
                    print(
                        f"The `gain_wv` of the quantizer {Q_gain_wv} is greater than {gain_wv}, the value you specified. ",
                        "Try other methods.",
                    )
                return None, _np.inf
            else:
                if verbose:
                    print("Success!")
                return Q, E
        else:
            if verbose:
                print(
                    "`system.m >= system.l` must be `True`. Try other methods.",
                )
            return None, _np.inf

    @staticmethod
    def design_LP(system: "System",  # type: ignore
                  *,
                  q: StaticQuantizer,
                  dim: int | InfInt = infint,
                  T: int | InfInt = infint,
                  gain_wv: float = _np.inf,
                  solver: str | None = None,
                  verbose: bool = False) -> Tuple["DynamicQuantizer | None", float]:
        """
        Design a stable and optimal dynamic quantizer using linear programming method.

        Note that this method doesn't confirm that
        `Q.gain_wv() < gain_wv` becomes `True`.

        Parameters
        ----------
        system : System
            Stable and SISO system instance.
        q : StaticQuantizer
            Static quantizer instance.
        dim : int or InfInt, optional
            Upper limit of quantizer order (default: infint). `dim` >= 1.
        T : int or InfInt, optional
            Estimation time (default: infint) (`T` > 0).
        gain_wv : float, optional
            Upper limit of gain w->v (default: inf) (`gain_wv` > 0).
        solver : str or None, optional
            CVXPY solver name (default: None). You can check the available solvers by
            `nqlib.installed_solvers()`.
            (If `None`, this function doesn't
            specify the solver).
        verbose : bool, optional
            If True, print progress (default: False).

        Returns
        -------
        Q : DynamicQuantizer or None
            Designed quantizer or None if not found.
        E : float
            Estimated E(Q). If `Q` is None, `E` is inf.

        Raises
        ------
        ValueError
            If system is unstable.

        References
        ----------
        .. [4] R. Morita, S. Azuma, Y. Minami and T. Sugie: Graphical design
           software for dynamic quantizers in control systems; SICE Journal 
           of Control, Measurement, and System Integration, Vol. 4, No. 5, 
           pp. 372-379 (2011)
        """
        if verbose:
            print("Trying to design a dynamic quantizer using LP...")
        # Check parameters
        T = validate_int_or_inf(
            T,
            minimum=1,  # T must be greater than 0
            name="T",
        )
        if T is infint:
            if verbose:
                print(
                    "`design_LP` currently supports only finite `T`.",
                    "Specify `T` or try other methods.",
                )
            return None, _np.inf
        else:
            if T % 2 == 0:
                T_solve = T + 1
            else:
                T_solve = T
        dim = validate_int_or_inf(
            dim,
            minimum=1,  # order must be greater than 0
            name="dim",
        )
        gain_wv = validate_float(
            gain_wv,
            infimum=0.0,  # gain_wv must be greater than 0
            name="gain_wv",
        )
        if T * float(dim) > 1000:
            if verbose:
                print("This may take very long time. Please wait or interrupt.")

        def _lp() -> Tuple[float, List[NDArrayNum], List[NDArrayNum], List[NDArrayNum]]:
            """
            Composes and solves linear problem to find good quantizer.

            Returns
            -------
            _np.ndarray

            Solution of LP. Form of
            ```
            _np.array([
                [H_20],
                [H_21],
                :
                [H_2(T-2)],
            ])
            ```
            So, the shape is `(m*(T-1), m)`.

            Raises
            ------
            cvxpy.SolverError
                if no solver named `solver` found.
            """
            m = system.m
            p = system.l

            # Reference:
            # Synthesis of Optimal Dynamic Quantizers for Discrete-Valued Input Control

            C = system.C1
            A = system.A + system.B2 @ system.C2
            B = system.B2

            ############################################################################
            # OP4
            ############################################################################
            # variables
            G = cvxpy.Variable((1, 1), name="G")  # The variable to minimize
            H2 = [
                cvxpy.Variable(
                    (m, m),
                    name=f"H2_{k}"
                ) for k in range(T)  # H2_0, ..., H2_(T-1)
            ]
            H2_bar = [
                cvxpy.Variable(
                    (m, m),
                    name=f"H2_bar_{k}"
                ) for k in range(T)  # H2_bar_0, ..., H2_bar_(T-1)
            ]
            Epsilon_bar = [
                cvxpy.Variable(
                    (p, m),
                    name=f"ε_bar_{k}"
                ) for k in range(1, T)  # ε_bar_1, ..., ε_bar_(T-1)
            ]

            # # Phi
            # [
            #     [CB        ],  # Phi_1
            #     [CAB       CB        ],
            #        :
            #     [CA^(k-1)B ...       CA^(i)B ...        CB        ],
            #        :
            #     [CA^(T-2)B ...       CA^(i)B ...        CB        ],
            # ]
            Phi = [
                _np.zeros((p, m * k))
                for k in range(1, T)  # Phi_1, ..., Phi_(T-1)
            ]
            CA_i = C  # C @ A^i
            for i in range(0, T - 1):
                for k in range(i + 1, T):
                    Phi[k - 1][0:p, (k - i - 1) * m:(k - i) * m] = CA_i @ B
                CA_i = CA_i @ A

            one_m = _np.ones((m, 1))
            one_p = _np.ones((p, 1))

            # compose a problem
            constraints: List[cvxpy.Constraint] = [
                (_np.abs(C @ B) + sum(Epsilon_bar)) @ one_m <= one_p @ G,
                *[
                    -Epsilon_bar[k - 1] <= C @ mpow(A, k) @ B + Phi[k - 1] @ cvxpy.vstack(H2[0:k])
                    for k in range(1, T)
                ],
                *[
                    Epsilon_bar[k - 1] >= C @ mpow(A, k) @ B + Phi[k - 1] @ cvxpy.vstack(H2[0:k])
                    for k in range(1, T)
                ],
                *[
                    -H2_bar[k] <= H2[k]
                    for k in range(T)
                ],
                *[
                    H2[k] <= H2_bar[k]
                    for k in range(T)
                ],
            ]
            if gain_wv < _np.inf:
                _c: cvxpy.Constraint = (_np.eye(m) + sum(H2_bar)) @ one_m <= gain_wv * one_m  # type: ignore
                constraints.append(_c)

            problem = cvxpy.Problem(
                cvxpy.Minimize(G),
                constraints,
            )

            problem.solve(solver=solver, verbose=verbose)
            return (
                G.value[0, 0],  # type: ignore
                [H2[k].value for k in range(T)],
                [H2_bar[k].value for k in range(T)],
                [Epsilon_bar[k - 1].value for k in range(1, T)],
            )

        # STEP1. LP
        start_time_lp = time.time()  # To measure time.
        G, H2, _H2_bar, _Epsilon_bar = _lp()  # Markov Parameter
        end_time_lp = time.time()  # To measure time.
        E = G * q.delta
        if verbose:
            print(f"Solved linear programming problem in {end_time_lp - start_time_lp:.3f}[s].")

        # STEP2. SVD
        start_time_SVD = time.time()  # To measure time.
        H, Wo, S, Wc = _SVD_from(H2, T)
        end_time_SVD = time.time()  # To measure time.
        if verbose:
            print(f"Calculated SVD in {end_time_SVD - start_time_SVD:.3f}[s].")

        # STEP3. compose Q
        start_time_Q = time.time()  # To measure time.
        Q, reduced = _compose_Q_from_SVD(
            system,
            q,
            T_solve,
            H,
            Wo,
            S,
            Wc,
            dim,
        )  # Q is definitely stable.
        end_time_Q = time.time()  # To measure time.
        if verbose:
            print(f"Composed Q in {end_time_Q - start_time_Q:.3f}[s].")

        if reduced:
            if verbose:
                print("Calculating E(Q).")
            E = system.E(Q, steptime=T, _check_stability=False, verbose=verbose)
        # Q_gain_wv = Q.gain_wv(steptime=T, verbose=verbose)
        if Q.N > dim:
            if verbose:
                warnings.warn(
                    f"The order of the quantizer {Q.N} is greater than {dim}, the value you specified. "
                    "Please reduce the order manually using `order_reduced()`, or try other methods.",
                )
            return Q, E
        # elif Q_gain_wv > gain_wv:
        #     warnings.warn(
        #         f"The `gain_wv` of the quantizer {Q_gain_wv} is greater than {gain_wv}, the value you specified. ",
        #     )
        if verbose:
            print("Success!")
        return Q, E

    @staticmethod
    def design_GD(
        system: "System",  # type: ignore
        *,
        q: StaticQuantizer,
        dim: int,
        T: int | InfInt = infint,
        gain_wv: float = _np.inf,
        verbose: bool = False,
        method: str = "SLSQP",
        obj_type: str = ["exp", "atan", "1.1", "100*1.1"][0],
    ) -> Tuple["DynamicQuantizer | None", float]:
        """
        Design a stable and optimal dynamic quantizer using gradient-based optimization.

        Parameters
        ----------
        system : System
            Stable and SISO system instance.
        q : StaticQuantizer
            Static quantizer instance.
        dim : int
            Quantizer order. `dim` >= 1.
        T : int or InfInt, optional
            Estimation time (default: infint). `T` >= 1.
        gain_wv : float, optional
            Upper limit of gain w->v (default: inf). `gain_wv` >= 0.
        verbose : bool, optional
            If True, print progress (default: False).
        method : str, optional
            Optimization method for scipy.optimize.minimize (default: 'SLSQP').
            (If `None`, this function doesn't specify the method).
        obj_type : str, optional
            Objective function type (default: 'exp').

        Returns
        -------
        Q : DynamicQuantizer or None
            Designed quantizer or None if not found.

        E : float
            Estimated E(Q). If Q is None, E is inf.

        Raises
        ------
        ValueError
            If system is unstable.

        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
        """
        T = validate_int_or_inf(
            T,
            minimum=1,  # T must be greater than 0
            name="T",
        )
        dim = validate_int(
            dim,
            minimum=1,  # order must be greater than 0
            name="dim",
        )
        # TODO: check if system is SISO

        # functions to calculate E from
        # x = [an, ..., a1, cn, ..., c1]  (n = dim)
        def a(x: NDArrayNum) -> NDArrayNum:
            """
            a = [an, ..., a1]
            """
            return x[:dim]

        def c(x: NDArrayNum) -> NDArrayNum:
            """
            c = [cn, ..., c1]
            """
            return x[dim:]

        def _Q(x: NDArrayNum) -> DynamicQuantizer:
            # controllable canonical form
            _A = block([
                [zeros((dim - 1, 1)), eye(dim - 1)],
                [-a(x)],
            ])
            _B = block([  # type: ignore
                [zeros((dim - 1, 1))],
                [1],
            ])
            _C = c(x)
            return DynamicQuantizer(
                A=_A,
                B=_B,
                C=_C,
                q=q,
            )

        def obj(x: NDArrayNum) -> Real:
            return _Q(x)._objective_function(system,
                                             T=T,
                                             gain_wv=gain_wv,  # type: ignore
                                             obj_type=obj_type)

        # optimize
        if verbose:
            print(
                "Designing a quantizer with gradient-based optimization.\n"
                f"The optimization method is '{method}'.\n"
                "### Message from `scipy.optimize.minimize()`. ###"
            )
        result = _minimize(
            obj,
            x0=_np.random.randn(2 * dim) * 0.001,  # TODO: Better init
            tol=0,
            options={
                "disp": verbose,
                'maxiter': 10000,
                'ftol': 1e-10,
                'iprint': 15,
            },
            method=method,
        )
        if verbose:
            print(result.message)
            print("### End of message from `scipy.optimize.minimize()`. ###")

        if result.success and obj(result.x) <= 0:
            Q = _Q(result.x)
            E = system.E(Q)
            if verbose:
                print("Optimization succeeded.")
                print(f"E = {E}")
        else:
            Q = None
            E = _np.inf
            if verbose:
                print("Optimization failed.")

        return Q, E

    @staticmethod
    def design_DE(
        system: "System",  # type: ignore
        *,
        q: StaticQuantizer,
        dim: int,  # This must be finite
        T: int | InfInt = infint,  # TODO: これより下を反映
        gain_wv: float = _np.inf,
        verbose: bool = False,
    ) -> Tuple["DynamicQuantizer | None", float]:  # TODO: method のデフォルトを決める
        """
        Design a stable and optimal dynamic quantizer using differential evolution.

        Parameters
        ----------
        system : System
            Stable and SISO system instance.
        q : StaticQuantizer
            Static quantizer instance.
        dim : int
            Order of the quantizer. `dim` >= 1.
        T : int or InfInt, optional
            Estimation time (default: infint). `T` >= 1.
        gain_wv : float, optional
            Upper limit of gain w->v (default: inf). `gain_wv` >= 0.
        verbose : bool, optional
            If True, print progress (default: False).

        Returns
        -------
        Q : DynamicQuantizer or None
            Designed quantizer or None if not found.
        E : float
            Estimated E(Q). If Q is None, E is inf.

        Raises
        ------
        ValueError
            If system is unstable.

        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
        """
        T = validate_int(
            T,
            minimum=1,  # T must be greater than 0
            name="T",
        )
        dim = validate_int(
            dim,
            minimum=1,  # order must be greater than 0
            name="dim",
        )
        # TODO: check if system is SISO

        # functions to calculate E from
        # x = [an, ..., a1, cn, ..., c1]  (n = dim)
        def a(x: NDArrayNum) -> NDArrayNum:
            """
            a = [an, ..., a1]
            """
            return x[:dim]

        def c(x: NDArrayNum) -> NDArrayNum:
            """
            c = [cn, ..., c1]
            """
            return x[dim:]

        def _Q(x: NDArrayNum) -> DynamicQuantizer:
            # controllable canonical form
            _A = block([
                [zeros((dim - 1, 1)), eye(dim - 1)],
                [-a(x)],
            ])
            _B = block([  # type: ignore
                [zeros((dim - 1, 1))],
                [1],
            ])
            _C = c(x)
            return DynamicQuantizer(
                A=_A,
                B=_B,
                C=_C,
                q=q,
            )

        def obj(x: NDArrayNum) -> Real:
            return _Q(x)._objective_function(
                system,
                T=T,
                gain_wv=gain_wv,  # type: ignore
            )

        def comb(k: int) -> int:
            return _comb(dim, k, exact=True, repetition=False)  # type: ignore

        # optimize
        bounds = [matrix([-1, 1])[0] * comb(i) for i in range(dim)] + [matrix([-2, 2])[0] * comb(dim // 2) for _ in range(dim)]
        if verbose:
            print("Designing a quantizer with differential evolution.")
            print("### Message from `scipy.optimize.differential_evolution()`. ###")
        result = _differential_evolution(
            obj,
            bounds=bounds,  # TODO: よりせまく
            atol=1e-6,  # type: ignore
            tol=0,  # relative
            maxiter=1000,
            strategy='rand2exp',
            disp=verbose,
        )

        if verbose:
            print(result.message)
            print("### End of message from `scipy.optimize.differential_evolution()`. ###")

        if result.success:
            Q = _Q(result.x)
            E = system.E(Q)
            if verbose:
                print("Optimization succeeded.")
                print(f"E = {E}")
        else:
            Q = None
            E = _np.inf
            if verbose:
                print("Optimization failed.")

        return Q, E

    def quantize(self, u: NDArrayNum) -> NDArrayNum:
        """
        Quantize the input signal using this dynamic quantizer.

        Parameters
        ----------
        u : NDArrayNum
            Input signal. Shape: (m, length).

        Returns
        -------
        NDArrayNum
            Quantized signal. Shape: (1, length).
        """
        u = matrix(u)
        length = u.shape[1]

        v = zeros((1, length))  # TODO: vと同じ次数になるはずでは？
        xi = zeros((len(self.A), length))

        for i in range(length):
            v[:, i:i + 1] = matrix(self.q(self.C @ xi[:, i:i + 1] + u[:, i:i + 1]))
            if i < length - 1:
                xi[:, i:i + 1 + 1] = matrix(self.A @ xi[:, i:i + 1] + self.B @ (v[:, i:i + 1] - u[:, i:i + 1]))
        return v

    def cost(self,
             system: "System",  # type: ignore
             steptime: int | InfInt = infint,
             _check_stability: bool = True) -> float:
        """
        Compute the cost (E(Q)) for the quantizer and system.

        Parameters
        ----------
        system : System
            System instance.
        steptime : int or InfInt, optional
            Number of steps (default: infint, which implies that this function
            calculates until convergence). `steptime` >= 1.
        _check_stability : bool, optional
            If True, check stability (default: True).

        Returns
        -------
        float
            Estimation of E(Q) in given steptime.

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        return system.E(self, steptime, _check_stability)

    def spec(self,
             steptime: int | InfInt = infint,
             show: bool = True) -> str:
        """
        Return a string summary of the quantizer's specification.

        Parameters
        ----------
        steptime : int or InfInt, optional
            Number of steps (default: infint, which implies that this function
            calculates until convergence). `steptime` >= 1.
        show : bool, optional
            If True, print the summary (default: True).

        Returns
        -------
        str
            Specification summary.
        """
        s = (
            "The specs of \n"
            f"{str(self)}\n"
            f"order    : {self.N}\n"
            f"stability: {'stable' if self.is_stable else 'unstable'}\n"
            f"gain_wv  : {self.gain_wv(steptime)}\n"
        )
        if show:
            print(s)
        return s


def order_reduced(Q: DynamicQuantizer, dim: int) -> DynamicQuantizer:
    """
    Returns the quantizer with its order reduced.

    Note that the quantizer with the reduced order
    will generally have larger `E(Q)` and a larger
    `gain_wv` than those of the original quantizer. 
    You should check the performance and gain yourself.

    This function requires slycot. Please install it.

    Parameters
    ----------
    Q : DynamicQuantizer
        The quantizer to be reduced. Must be an instance of `DynamicQuantizer`.
    dim : int
        Desired order (1 <= `dim` < `Q.N`).

    Returns
    -------
    DynamicQuantizer
        Quantizer with reduced order.
    """
    # check Q
    if type(Q) is not DynamicQuantizer:
        raise TypeError(
            '`Q` must be an instance of `nqlib.DynamicQuantizer`.'
        )
    dim = validate_int(
        dim,
        minimum=1,  # order must be greater than 0
        name="dim",
    )
    return Q.order_reduced(dim)
