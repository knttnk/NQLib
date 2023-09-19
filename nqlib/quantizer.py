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
    ctrl_poles = _ctrl.pole
    ctrl_zeros = _ctrl.zero


import cvxpy
import numpy as _np
from numpy import inf
from scipy.optimize import differential_evolution as _differential_evolution
from scipy.optimize import minimize as _minimize
from scipy.special import comb as _comb

from .linalg import (block, eig_max, eye, kron, matrix, mpow, norm, ones, pinv,
                     zeros)

_ctrl.use_numpy_matrix(False)
__all__ = [
    'StaticQuantizer',
    'DynamicQuantizer',
    'order_reduced',
]


class _ConnectionType(_Enum):
    """
    Inherits from `enum.Enum`.
    This class represents how system is connected.
    Each values correspond to following methods.

    See Also
    --------
    nqlib.System.from_FF()
        Corresponds to `_ConnectionType.FF`.
    nqlib.System.from_FB_connection_with_input_quantizer()
        Corresponds to `_ConnectionType.FB_WITH_INPUT_QUANTIZER`.
    nqlib.System.from_FB_connection_with_output_quantizer()
        Corresponds to `_ConnectionType.FB_WITH_OUTPUT_QUANTIZER`.
    nqlib.System()
        Corresponds to `_ConnectionType.ELSE`.
    """
    FF = _auto()
    FB_WITH_INPUT_QUANTIZER = _auto()
    FB_WITH_OUTPUT_QUANTIZER = _auto()
    ELSE = _auto()


class StaticQuantizer():
    """
    Represents static quantizer.
    """

    def __init__(self,
                 function: Callable[[_np.ndarray], _np.ndarray],
                 delta: float,
                 *,
                 error_on_excess: bool = True):
        """
        Initializes an instance of `StaticQuantizer`.

        Parameters
        ----------
        function : Callable[[_np.ndarray], _np.ndarray]
            For simulation. This function returns quantized value
            derived from input.
        delta : float
            Max of `abs(function(u) - u)`.
        error_on_excess : bool, optional
            Whether to raise an error when
            `numpy.max(abs(function(u) - u)) > delta` becomes `True`.
            (The default is `True`).
        """
        try:
            self.delta = float(delta)
        except:
            raise TypeError('`delta` must be a real number.')
        if self.delta <= 0:
            raise ValueError("`delta` must be greater than `0`.")

        # check and assign `function`
        if not callable(function):
            raise TypeError('`function` must be a callable object.')
        else:
            if error_on_excess:
                def safe_function(u):
                    # returns function(u)
                    v = function(u)
                    if _np.max(abs(v - u)) > delta:
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

    def __call__(self, u):
        """
        Calls `self.quantize`.
        """
        return self.quantize(u)

    def quantize(self, u):
        """
        Quantizes the input signal `u`.
        Returns v in the following figure.

        ```text
               +-----+       
         u --->|  q  |---> v 
               +-----+       
        ```

        "q" in this figure means this quantizer.

        Parameters
        ----------
        u : array_like
            Input signal.

        Returns
        -------
        v : np.ndarray
            Quantized signal.
        """
        return self._function(u)

    @staticmethod
    def mid_tread(d: float,
                  bit: Union[None, int] = None,
                  *,
                  error_on_excess: bool = True) -> "StaticQuantizer":
        """
        Creates an mid-tread uniform `StaticQuantizer` `q`.

        Parameters
        ----------
        d : float
            Quantization step size. Must be greater than `0`.
        bit : None or int, optional
            Number of bits.
            (The default is `None`, which means infinity.)
        error_on_excess : bool, optional
            Whether to raise an error when
            `numpy.max(abs(function(u) - u)) > delta` becomes `True`.
            (The default is `True`).

        Returns
        -------
        q : StaticQuantizer
            Mid-tread uniform `StaticQuantizer`.
        """
        if not isinstance(d, (float, int)):
            raise TypeError('`d` must be an real number.')
        if d <= 0:
            raise ValueError('`d` must be greater than `0`.')

        # function to quantize
        def q(u):
            return ((u + d / 2) // d) * d

        # limit the values
        if bit is None:
            function = q
        elif isinstance(bit, int) and bit > 0:
            def function(u):
                return _np.clip(
                    q(u),
                    a_min=-(2**(bit - 1) - 1) * d,
                    a_max=2**(bit - 1) * d,
                )
        else:
            raise TypeError('`bit` must be an natural number or `None`.')

        return StaticQuantizer(
            function=function,
            delta=d / 2,
            error_on_excess=error_on_excess,
        )

    @staticmethod
    def mid_riser(d: float,
                  bit: Union[None, int] = None,
                  *,
                  error_on_excess: bool = True) -> "StaticQuantizer":
        """
        Creates an mid-riser uniform `StaticQuantizer` `q`.

        Parameters
        ----------
        d : float
            Quantization step size. Must be greater than `0`.
        bit : None or int, optional
            Number of bits.
            (The default is `None`, which means infinity.)
        error_on_excess : bool, optional
            Whether to raise an error when
            `numpy.max(abs(function(u) - u)) > delta` becomes `True`.
            (The default is `True`).

        Returns
        -------
        q : StaticQuantizer
            Mid-riser uniform `StaticQuantizer`.
        """
        if not isinstance(d, (float, int)):
            raise TypeError('`d` must be an real number.')
        if d <= 0:
            raise ValueError('`d` must be greater than `0`.')

        # function to quantize
        def q(u):
            return (u // d + 1 / 2) * d

        # limit the values
        if bit is None:
            function = q
        elif isinstance(bit, int) and bit > 0:
            def function(u):
                a_max = (2**(bit - 1) - 1 / 2) * d
                return _np.clip(
                    q(u),
                    a_min=-a_max,
                    a_max=a_max,
                )
        else:
            raise TypeError('`bit` must be an natural number or `None`.')

        return StaticQuantizer(
            function=function,
            delta=d / 2,
            error_on_excess=error_on_excess,
        )


def _find_tau(A_tilde, C_1, B_2, l: int, tau_max: int) -> int:
    """
    Finds the smallest integer tau satisfying
    `tau >= 0 and C_1 @ mpow(A + B_2@C_2, tau) @ B_2 != 0`

    Parameters
    ----------
    A_tilde
    C_1
    B_2
    l : int
    tau_max : int
        This function looks for tau in the interval [`0`, `tau_max`].

    Returns
    -------
    int
        `tau`. If `tau` doesn't exist, returns `-1`.
    """
    def is_not_zero(M):
        return (M * 0 != M).all()

    for t in range(tau_max):
        M = C_1 @ mpow(A_tilde, t) @ B_2
        if is_not_zero(M):
            if _np.linalg.matrix_rank(M) == l:
                return t
            else:
                return -1
    return -1


def _nq_serial_decomposition(system: "System",
                             q: StaticQuantizer,
                             verbose: bool) -> Tuple["DynamicQuantizer", float]:
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
    tf = system.P.tf1
    zeros_ = ctrl_zeros(tf)
    poles = ctrl_poles(tf)

    unstable_zeros = [zero for zero in zeros_ if abs(zero) > 1]

    z = _ctrl.TransferFunction.z
    z.dt = tf.dt

    if len(unstable_zeros) == 0:
        n_time_delay = len([p for p in poles if p == 0])  # count pole-zero
        G = 1 / z**n_time_delay
        F = tf / G
        F_ss = _ctrl.tf2ss(F)
        B_F = matrix(F_ss.B)
        C_F = matrix(F_ss.C)

        E = abs(C_F @ B_F)[0, 0] * q.delta
    elif len(unstable_zeros) == 1:
        i = len(poles) - len(zeros_)  # relative order
        if i < 1:
            return None, inf
        a = unstable_zeros[0]
        G = (z - a) / z**i
        F = _ctrl.minreal(tf / G, verbose=False)
        F = _ctrl.tf(_np.real(F.num), _np.real(F.den), F.dt)
        F_ss = _ctrl.tf2ss(F)
        B_F = matrix(F_ss.B)
        C_F = matrix(F_ss.C)

        E = (1 + abs(a)) * abs(C_F @ B_F)[0, 0] * q.delta
    else:
        return None, inf

    A_F = matrix(F_ss.A)
    D_F = matrix(F_ss.D)

    # check
    if (C_F @ B_F)[0, 0] == 0:
        if verbose:
            print("CF @ BF == 0 became true. Couldn't calculate by using serial system decomposition.")
        return None, inf
    if D_F[0, 0] != 0:
        if verbose:
            print("DF == 0 became true. Couldn't calculate by using serial system decomposition.")
        return None, inf
    Q = DynamicQuantizer(
        A=A_F,
        B=B_F,
        C=- 1 / (C_F @ B_F)[0, 0] * C_F @ A_F,
        q=q
    )
    if verbose:
        print("Success!")
    return Q, E


def _SVD_from(H2: List[_np.ndarray],
              T: int) -> _np.ndarray:
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


def _compose_Q_from_SVD(system: "System",
                        q: StaticQuantizer,
                        T: int,
                        H: _np.ndarray,
                        Wo: _np.ndarray,
                        S: _np.ndarray,
                        Wc: _np.ndarray,
                        dim: int) -> Tuple["DynamicQuantizer", bool]:
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
        nQ = dim
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
    Represents dynamic quantizer.
    """

    def __init__(self, A, B, C, q: StaticQuantizer):
        """
        Initializes an instance of `DynamicQuantizer`.

        Parameters
        ----------
        A : array_like
            Interpreted as a square matrix.
            `N` is defined as the number of rows in matrix `A`.
        B : array_like
            Interpreted as a (`N` x `m`) matrix.
            `m` is defined as the number of columns in matrix `B`.
        C : array_like
            Interpreted as a (`m` x `N`) matrix.
        q : StaticQuantizer

        Notes
        -----
        An instance of `DynamicQuantizer` represents a dynamic quantizer
        Q given by

            Q : { xi(t+1) =    A xi(t) + B u(t)
                {   v(t)  = q( C xi(t) +   u(t) )
        """
        try:
            A_mat = matrix(A)
            B_mat = matrix(B)
            C_mat = matrix(C)
        except:
            raise TypeError("`A`, `B` and `C` must be interpreted as matrices.")

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
                    formatter=lambda n: f"{n}",
                    sep: str = ", ",
                    linesep="\n",
                    indent=""):
        """
        Returns the formatted string of DynamicQuantizer.

        Args:
            index (int): 0, 1 and 2 represents A, B and C, respectively.
            sep (str, optional): Defaults to ",".
            linesep (str, optional): Defaults to "\n,".
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

    def _str(self, q_str):
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

    def gain_wv(self, steptime: Union[None, int, float] = None) -> float:
        """
        Computes the gain u->v and w->v of this `DynamicQuantizer` in
        `steptime`[1]_.

        Parameters
        ----------
        steptime : int, None or numpy.inf, optional
            (The default is None, which means infinity).

        Returns
        -------
        float
            Estimation of gain w->v .

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        if steptime is None:  # None means infinity.
            steptime = inf
        elif type(steptime) is not int:
            raise TypeError("steptime must be an integer greater than 0 .")
        elif steptime < 1:
            raise ValueError(
                "steptime must be an integer greater than 0 ."
            )

        if not self.is_stable:
            ret = inf
        else:  # stable case
            sum_Q_wv = eye(self.m)

            i = 0
            A_i = eye(self.N)  # (self.A + self.B@self.C)**i
            ret = 0
            while i < steptime:
                sum_Q_wv = sum_Q_wv + abs(self.C @ A_i @ self.B)
                ret_past = ret
                ret = norm(sum_Q_wv)
                if i > 0:
                    if abs(ret - ret_past) < 1e-8:
                        break
                i = i + 1
                A_i = A_i @ (self.A + self.B @ self.C)

        return ret

    def _objective_function(self,
                            system: "System",
                            *,
                            T: int = None,  # TODO: これより下を反映
                            gain_wv: float = inf,
                            obj_type=["exp", "atan", "1.1", "100*1.1"][0]) -> float:
        """
        Used in numerical optimization.

        Parameters
        ----------
        system : System
            Must be stable and SISO.
        T : int, None or numpy.inf, optional
            Estimation time. Must be greater than `0`.
            (The default is `None`, which means infinity).
        gain_wv : float, optional
            Upper limit of gain w->v . Must be greater than `0`.
            (The default is `numpy.inf`).

        Returns
        -------
        value : float

        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
        """
        # if T is None:
        #     # TODO: support infinity evaluation time
        #     return None, inf
        if system.m != 1:
            raise ValueError("`design_GD` and `design_DE` is currently supported for SISO systems only.")

        # Values representing the stability or gain_wv.y
        # if max(constraint_values) < 0, this quantizer satisfies the
        # constraints.
        constraint_values = [
            eig_max(self.A + self.B @ self.C) - 1,
        ]
        if not _np.isposinf(gain_wv):
            constraint_values.append(self.gain_wv(T) - gain_wv)

        max_v = max(constraint_values)
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
        else:
            return max_v

    def order_reduced(self, dim) -> "DynamicQuantizer":
        """
        Returns the quantizer with its order reduced.

        Note that the quantizer with the reduced order
        will generally have larger `E(Q)` and a larger
        `gain_wv` than those of the original quantizer. 
        You should check the performance and gain yourself.

        This function requires slycot. Please install it.

        Parameters
        ----------
        dim : int
            Order of the quantizer to be returned.
            Must be greater than `0` and less than `self.N`.

        Returns
        -------
        Q : DynamicQuantizer

        Raises
        ------
        ImportError
            if NQLib couldn't import slycot.
        """
        try:
            from slycot import ab09ad
        except ImportError as e:
            raise ImportError((
                "Reducing order of a quantizer requires slycot."
                " Please install it."
            ))
        # マルコフパラメータから特異値分解する
        Nr, Ar, Br, Cr, hsv = ab09ad(
            "D",  # means "Discrete time"
            "B",  # balanced (B) or not (N)
            "S",  # scale (S) or not (N)
            self.N,  # np.size(A,0)
            self.m,  # np.size(B,1)
            self.m,  # np.size(C,0)
            self.A, self.B, self.C,
            nr=dim,
            tol=0.0,
        )
        return DynamicQuantizer(Ar, Br, Cr, self.q)

    @property
    def is_stable(self) -> bool:
        """
        Returns stability of this quantizer[1]_.

        Returns
        -------
        bool
            `True` if stable, `False` if not.

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
        Q : DynamicQuantizer
        """
        minreal_ss = _ctrl.ss(
            self.A,
            self.B,
            self.C,
            self.C @ self.B * 0,
            True,
        ).minreal()
        return DynamicQuantizer(minreal_ss.A,
                                minreal_ss.B,
                                minreal_ss.C,
                                self.q)

    @staticmethod
    def design(system: "System",
                       *,
                       q: StaticQuantizer,
                       T: int = None,
                       gain_wv: float = inf,
                       dim: int = inf,
                       verbose: bool = False,
                       use_analytical_method=True,
                       use_LP_method=True,
                       use_design_GB_method=True,
                       use_DE_method=False,
                       solver: str = None) -> Tuple["DynamicQuantizer", float]:
        """
        Calculates the stable and optimal dynamic quantizer `Q` for `system`.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        Parameters
        ----------
        system : System
            Must be stable.
        q : StaticQuantizer
            Returned dynamic quantizer contains this static quantizer.
            `q.delta` is important to estimate E(Q).
        T : int, None or numpy.inf, optional
            Estimation time. Must be greater than `0`.
            (The default is `None`, which means infinity).
        gain_wv : float, optional
            Upper limit of gain w->v . Must be greater than `0`.
            (The default is `numpy.inf`).
        dim : int, optional
            Upper limit of order of `Q`. Must be greater than `0`.
            (The default is `inf`).
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).
        solver : str, optional
            Name of CVXPY solver. You can check the available solvers by
            `nqlib.installed_solvers()`.
            (The default is `None`, which implies that this function doesn't
            specify the solver).

        Returns
        -------
        (Q, E) : Tuple[DynamicQuantizer, float]
            `Q` is the stable and optimal dynamic quantizer for `system`.
            `E` is estimation of E(Q).

        Raises
        ------
        ValueError
            If `system` is unstable.

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        .. [2]  S. Azuma, Y. Minami and T. Sugie: Optimal dynamic quantizers
           for feedback control with discrete-level actuators; Journal of 
           Dynamic Systems, Measurement, and Control, Vol. 133, No. 2, 021005
           (2011)
        .. [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会
           論文集，Vol. 52, pp. 46–51(2016)
        .. [4] R. Morita, S. Azuma, Y. Minami and T. Sugie: Graphical design
           software for dynamic quantizers in control systems; SICE Journal 
           of Control, Measurement, and System Integration, Vol. 4, No. 5, 
           pp. 372-379 (2011)
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
        """
        def _print_report(Q, method: str):
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

        # check gain
        if gain_wv < 1:
            raise ValueError(
                '`gain_wv` must be greater than or equal to `1`.'
            )

        # check dim
        if dim != inf and not isinstance(dim, int):
            raise TypeError("`dim` must be `numpy.inf` or an instance of `int`.")
        elif dim < 1:
            raise ValueError('`dim` must be greater than `0`.')

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

        candidates = []
        # numerically optimize
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
                candidates.append(
                    dict(Q=Q, E=E)
                )
        if dim is None or dim == inf:
            dim = system.n
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
                candidates.append(
                    dict(Q=Q, E=E)
                )
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
                candidates.append(
                    dict(Q=Q, E=E)
                )

        # compare all candidates and return the best
        if len(candidates) > 0:
            Q, E = min(
                candidates,
                key=lambda c: c.E,
            )
            return Q, E
        else:
            if verbose:
                print(
                    "NQLib could not design a quantizer under these conditions. ",
                    "Please try different conditions.",
                )

            return None, inf

    @staticmethod
    def design_AG(system: "System",
                  *,
                  q: StaticQuantizer,
                  dim: int = inf,
                  gain_wv: float = inf,
                  allow_unstable: bool = False,
                  verbose: bool = False) -> Tuple["DynamicQuantizer", float]:
        """
        Finds the stable and optimal dynamic quantizer for `system`
        algebraically[2]_,[3]_.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        If NQLib couldn't find `Q` such that
        ```
        all([
            Q.N <= dim,
            Q.gain_wv() < gain_wv,
            Q.is_stable,
        ])
        ```
        becomes `True`, this method returns `(None, inf)`.

        Parameters
        ----------
        system : System
            Must be stable and SISO.
        q : StaticQuantizer
            Returned dynamic quantizer contains this static quantizer.
            `q.delta` is important to estimate E(Q).
        dim : int
            Upper limit of order of `Q`. Must be greater than `0`.
            (The default is `inf`).
        gain_wv : float, optional
            Upper limit of gain w->v . Must be greater than `0`.
            (The default is `numpy.inf`).
        allow_unstable: bool, optional
            Whether to return valid `Q` even if it is unstable.
            It's recommended to set `verbose` to `True`, so that 
            you can remember the result is unstable.
            If this is `True`, design method in reference [3] will not be used.
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).

        Returns
        -------
        (Q, E) : Tuple[DynamicQuantizer, float]
            `Q` is the stable and optimal dynamic quantizer for `system`.
            `E` is estimation of E(Q).

        Raises
        ------
        ValueError
            If `system` is unstable.

        References
        ----------
        .. [2]  S. Azuma, Y. Minami and T. Sugie: Optimal dynamic quantizers
           for feedback control with discrete-level actuators; Journal of 
           Dynamic Systems, Measurement, and Control, Vol. 133, No. 2, 021005
           (2011)
        .. [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会
           論文集，Vol. 52, pp. 46–51(2016)
        """
        if verbose:
            print("Trying to calculate optimal dynamic quantizer...")

        A_tilde = system.A + system.B2 @ system.C2  # convert to closed loop
        def _Q():
            return DynamicQuantizer(
                A=A_tilde,
                B=system.B2,
                C=-pinv(system.C1 @ mpow(A_tilde, tau) @ system.B2) @ system.C1 @ mpow(A_tilde, tau + 1),
                q=q,
            )
        if (
            (not allow_unstable) and
            (system.type == _ConnectionType.FF and system.P.tf1.issiso())
        ):
            # FF and SISO
            Q, E = _nq_serial_decomposition(system, q, verbose)
            if Q is not None:
                return Q, E
        if system.m >= system.l:
            # S2
            tau = _find_tau(A_tilde, system.C1, system.B2, system.l, 10000)

            if tau == -1:
                if verbose:
                    print("Couldn't calculate optimal dynamic quantizer algebraically. Trying another method...")
                return None, inf

            Q = _Q()

            E = norm(abs(system.C1 @ mpow(A_tilde, tau) @ system.B2)) * q.delta
            Q_gain_wv = Q.gain_wv()

            if not Q.is_stable:
                if allow_unstable:
                    if verbose:
                        print("The quantizer is unstable.")
                    E = inf
                else:
                    if verbose:
                        print("The quantizer is unstable. Try other methods.")
                    return None, inf
            if Q.N > dim:
                if verbose:
                    print(
                        f"The order of the quantizer {Q.N} is greater than {dim}, the value you specified. ",
                        "Try other methods.",
                    )
                return None, inf
            elif Q_gain_wv > gain_wv:
                if verbose:
                    print(
                        f"The `gain_wv` of the quantizer {Q_gain_wv} is greater than {gain_wv}, the value you specified. ",
                        "Try other methods.",
                    )
                return None, inf
            else:
                if verbose:
                    print("Success!")
                return Q, E
        else:
            if verbose:
                print(
                    "`system.m >= system.l` must be `True`. Try other methods.",
                )
            return None, inf

    @staticmethod
    def design_LP(system: "System",
                  *,
                  q: StaticQuantizer,
                  dim: int = inf,
                  T: int = None,
                  gain_wv: float = inf,
                  solver: str = None,
                  verbose: bool = False) -> Tuple["DynamicQuantizer", float]:
        """
        Finds the stable and optimal dynamic quantizer for `system`
        with method using linear programming method[1]_.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        If NQLib couldn't find `Q` such that
        ```
        all([
            Q.N <= dim,
            Q.is_stable,
        ])
        ```
        becomes `True`, this method returns `(None, inf)`.

        Note that this method doesn't confirm that
        `Q.gain_wv() < gain_wv` becomes `True`.

        Parameters
        ----------
        system : System
            Must be stable and SISO.
        q : StaticQuantizer
            Returned dynamic quantizer contains this static quantizer.
            `q.delta` is important to estimate E(Q).
        dim : int
            Upper limit of order of `Q`. Must be greater than `0`.
            (The default is `inf`).
        T : int, None or numpy.inf, optional
            Estimation time. Must be greater than `0`.
            (The default is `None`, which means infinity).
        gain_wv : float, optional
            Upper limit of gain w->v . Must be greater than `0`.
            (The default is `numpy.inf`).
        solver : str, optional
            Name of CVXPY solver. You can check the available solvers by
            `nqlib.installed_solvers()`.
            (The default is `None`, which implies that this function doesn't
            specify the solver).
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).

        Returns
        -------
        (Q, E) : Tuple[DynamicQuantizer, float]
            `Q` is the stable and optimal dynamic quantizer for `system`.
            `E` is estimation of E(Q).

        Raises
        ------
        ValueError
            If `system` is unstable.

        References
        ----------
        .. [4] R. Morita, S. Azuma, Y. Minami and T. Sugie: Graphical design
           software for dynamic quantizers in control systems; SICE Journal 
           of Control, Measurement, and System Integration, Vol. 4, No. 5, 
           pp. 372-379 (2011)
        """
        if verbose:
            print("Trying to design a dynamic quantizer using LP...")
        if T is None or T == inf:
            T = inf
            T_solve = T
            if verbose:
                print(
                    "`design_LP` currently supports only finite `T`.",
                    "Specify `T` or try other methods.",
                )
            return None, inf
        else:
            if T % 2 == 0:
                T_solve = T + 1
            else:
                T_solve = T
        if T * dim > 1000:
            if verbose:
                print("This may take very long time. Please wait or interrupt.")

        def _lp() -> _np.ndarray:
            """
            Composes and solves linear problem to find good quantizer.

            Returns
            -------
            _np.ndarray
                Solution of LP. Form of
                ```text
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
            constraints = [
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
            if gain_wv < inf:
                constraints.append(
                    (_np.eye(m) + sum(H2_bar)) @ one_m <= gain_wv * one_m
                )

            problem = cvxpy.Problem(
                cvxpy.Minimize(G),
                constraints,
            )

            problem.solve(solver=solver, verbose=verbose)
            ret = (
                G.value[0, 0],
                [H2[k].value for k in range(T)],
                [H2_bar[k].value for k in range(T)],
                [Epsilon_bar[k - 1].value for k in range(1, T)],
            )
            return ret

        # STEP1. LP
        start_time_lp = time.time()  # To measure time.
        G, H2, H2_bar, Epsilon_bar = _lp()  # Markov Parameter
        E = G * q.delta
        end_time_lp = time.time()  # To measure time.
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
            E = system.E(Q, steptime=T, _check_stability=False)
        Q_gain_wv = Q.gain_wv(steptime=T)
        if Q.N > dim:
            if verbose:
                warnings.warn(
                    f"The order of the quantizer {Q.N} is greater than {dim}, the value you specified. ",
                    "Please reduce the order manually using `order_reduced()`, or try other methods.",
                )
            return Q, E
        elif Q_gain_wv > gain_wv:
            warnings.warn(
                f"The `gain_wv` of the quantizer {Q_gain_wv} is greater than {gain_wv}, the value you specified. ",
            )
        if verbose:
            print("Success!")
        return Q, E

    @staticmethod
    def design_GD(system: "System",
                  *,
                  q: StaticQuantizer,
                  dim: int,
                  T: int = None,
                  gain_wv: float = inf,
                  verbose: bool = False,
                  method: str = "SLSQP",
                  obj_type=["exp", "atan", "1.1", "100*1.1"][0]) -> Tuple["DynamicQuantizer", float]:
        """
        Finds the stable and optimal dynamic quantizer `Q` for `system`.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        If NQLib couldn't find `Q` such that
        ```
        all([
            Q.N == dim,
            Q.gain_wv() < gain_wv,
            Q.is_stable,
        ])
        ```
        becomes `True`, this method returns `(None, inf)`.

        Parameters
        ----------
        system : System
            Must be stable and SISO.
        q : StaticQuantizer
            Returned dynamic quantizer contains this static quantizer.
            `q.delta` is important to estimate E(Q).
        dim : int
            Upper limit of order of `Q`. Must be greater than `0`.
        T : int, None or numpy.inf, optional
            Estimation time. Must be greater than `0`.
            (The default is `None`, which means infinity).
        gain_wv : float, optional
            Upper limit of gain w->v . Must be greater than `0`.
            (The default is `numpy.inf`).
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).
        method : str, optional
            Specifies which method should be used in
            `scipy.optimize.minimize()`.
            (The default is `None`, which implies that this function doesn't
            specify the method).

        Returns
        -------
        (Q, E) : Tuple[DynamicQuantizer, float]
            `Q` is the stable and optimal dynamic quantizer for `system`.
            `E` is estimation of E(Q).

        Raises
        ------
        ValueError
            If `system` is unstable.

        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
        """  # TODO: ドキュメント更新
        # if T is None:
        #     # TODO: support infinity evaluation time
        #     return None, inf
        if not isinstance(dim, int):
            raise TypeError("`dim` must be `numpy.inf` or an instance of `int`.")
        elif dim < 1:
            raise ValueError("`dim` must be greater than `0`.")
        # TODO: siso のチェック

        # functions to calculate E from
        # x = [an, ..., a1, cn, ..., c1]  (n = dim)
        def a(x):
            """
            a = [an, ..., a1]
            """
            return x[:dim]

        def c(x):
            """
            c = [cn, ..., c1]
            """
            return x[dim:]

        def _Q(x):
            # controllable canonical form
            _A = block([
                [zeros((dim - 1, 1)), eye(dim - 1)],
                [-a(x)],
            ])
            _B = block([
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

        def obj(x):
            return _Q(x)._objective_function(system,
                                             T=T,
                                             gain_wv=gain_wv,
                                             obj_type=obj_type)

        # optimize
        if verbose:
            print("Designing a quantizer with gradient-based optimization.")
            print(f"The optimization method is '{method}'.")
            print("### Message from `scipy.optimize.minimize()`. ###")
        result = _minimize(obj,
                           x0=zeros(2 * dim)[0],
                           tol=0,
                           options={
                               "disp": verbose,
                               'maxiter': 10000,
                               'ftol': 1e-10,
                           },
                           method=method)
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
            E = inf
            if verbose:
                print("Optimization failed.")

        return Q, E

    @staticmethod
    def design_DE(system: "System",
                  *,
                  q: StaticQuantizer,
                  dim: int,
                  T: int = None,  # TODO: これより下を反映
                  gain_wv: float = inf,
                  verbose: bool = False) -> Tuple["DynamicQuantizer", float]:  # TODO: method のデフォルトを決める
        """
        Calculates the stable and optimal dynamic quantizer `Q` for `system`.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        Parameters
        ----------
        system : System
            Must be stable and SISO.
        q : StaticQuantizer
            Returned dynamic quantizer contains this static quantizer.
            `q.delta` is important to estimate E(Q).
        dim : int
            Order of `Q`. Must be greater than `0`.
        T : int, None or numpy.inf, optional
            Estimation time. Must be greater than `0`.
            (The default is `None`, which means infinity).
        gain_wv : float, optional
            Upper limit of gain w->v . Must be greater than `0`.
            (The default is `numpy.inf`).
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).

        Returns
        -------
        (Q, E) : Tuple[DynamicQuantizer, float]
            `Q` is the stable and optimal dynamic quantizer for `system`.
            `E` is estimation of E(Q).

        Raises
        ------
        ValueError
            If `system` is unstable.

        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
        """
        # if T is None:
        #     # TODO: support infinity evaluation time
        #     return None, inf
        if not isinstance(dim, int):
            raise TypeError("`dim` must be an instance of `int`.")
        elif dim < 1:
            raise ValueError("`dim` must be greater than `0`.")
        # TODO: siso のチェック

        # functions to calculate E from
        # x = [an, ..., a1, cn, ..., c1]  (n = dim)
        def a(x):
            """
            a = [an, ..., a1]
            """
            return x[:dim]

        def c(x):
            """
            c = [cn, ..., c1]
            """
            return x[dim:]

        def _Q(x):
            # controllable canonical form
            _A = block([
                [zeros((dim - 1, 1)), eye(dim - 1)],
                [-a(x)],
            ])
            _B = block([
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

        def obj(x):
            return _Q(x)._objective_function(system,
                                             T=T,
                                             gain_wv=gain_wv)

        def comb(k):
            return _comb(dim, k, exact=True, repetition=False)

        # optimize
        bounds = [matrix([-1, 1])[0] * comb(i) for i in range(dim)] + [matrix([-2, 2])[0] * comb(dim // 2) for i in range(dim)]
        if verbose:
            print("Designing a quantizer with differential evolution.")
            print("### Message from `scipy.optimize.differential_evolution()`. ###")
        result = _differential_evolution(
            obj,
            bounds=bounds,  # TODO: よりせまく
            atol=1e-6,  # absolute
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
            E = inf
            if verbose:
                print("Optimization failed.")

        return Q, E

    def quantize(self, u) -> _np.ndarray:
        """
        Quantizes the input signal `u`.
        Returns v in the following figure.

        ```text
               +-----+       
         u --->|  Q  |---> v 
               +-----+       
        ```

        "Q" in this figure means this quantizer.

        Parameters
        ----------
        u : array_like
            Input signal.

        Returns
        -------
        v : np.ndarray
            Quantized signal.
        """
        u = matrix(u)
        length = u.shape[1]

        v = zeros((1, length))
        xi = zeros((len(self.A), length))

        for i in range(length):
            v[:, i:i + 1] = matrix(self.q(self.C @ xi[:, i:i + 1] + u[:, i:i + 1]))
            if i < length - 1:
                xi[:, i:i + 1 + 1] = matrix(self.A @ xi[:, i:i + 1] + self.B @ (v[:, i:i + 1] - u[:, i:i + 1]))
        return v

    def cost(self,
             system: "System",
             steptime: Union[int, None] = None,
             _check_stability: bool = True) -> float:
        """
        Returns estimation of E(Q), where Q is this dynamic quantizer.

        Parameters
        ----------
        system : System
            The system to insert this dynamic quantizer.
        steptime : int or None, optional
            Evaluation time. Must be a natural number.
            (The default is `None`, which implies that this function
            calculates until convergence.)
        _check_stability : bool, optional
            This shouldn't be changed.
            `(steptime is not None or _check_stability)` must be `True`.
            (The default is `True`.)

        Returns
        -------
        float
            Estimation of E(Q) in `steptime`.

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        return system.E(self, steptime, _check_stability)

    def spec(self,
             steptime: Union[int, None] = None,
             show: bool = True) -> float:
        """
        Prints the specs of this DynamicQuantizer.

        Parameters
        ----------
        steptime : int or None, optional
            Evaluation time to compute the gain. Must be a natural number.
            (The default is `None`, which implies that this function
            calculates until convergence.)
        print : bool, optional
            Whether to print or only return the string.

        Returns
        -------
        str
            Printed string.
        """
        s = "The specs of \n"
        s += str(self) + "\n"
        s += f"order    : {self.N}\n"
        s += f"stability: {'stable' if self.is_stable else 'unstable'}\n"
        s += f"gain_wv  : {self.gain_wv(steptime)}\n"
        show and print(s)
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
        The quantizer you want to reduce the order of.
    dim : int
        Order of the quantizer to be returned.
        Must be greater than `0` and less than `self.N`.

    Returns
    -------
    Q : DynamicQuantizer

    Raises
    ------
    ImportError
        if NQLib couldn't import slycot.
    """
    # check Q
    if type(Q) is not DynamicQuantizer:
        raise TypeError(
            '`Q` must be an instance of `nqlib.DynamicQuantizer`.'
        )
    # check dim
    if not isinstance(dim, int):
        raise TypeError("`dim` must be an instance of `int`.")
    elif dim < 1:
        raise ValueError('`dim` must be greater than `0`.')
    return Q.order_reduced(dim)
