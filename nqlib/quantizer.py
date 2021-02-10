import math
import time
from enum import Enum as _Enum
from enum import auto as _auto
from typing import Callable, Tuple, Union

import control as _ctrl
import cvxpy
import numpy as _np
from numpy import inf
from scipy.special import comb as _comb
from scipy.optimize import minimize as _minimize
from scipy.optimize import differential_evolution as _differential_evolution

from .linalg import block, eye, kron, matrix, norm, ones, pinv, zeros, eig_max, mpow  # TODO: mpow はいらん

_ctrl.use_numpy_matrix(False)
__all__ = [
    'StaticQuantizer',
    'DynamicQuantizer',
]


class _ConnectionType(_Enum):
    """
    Inherits from `enum.Enum`.
    This class represents how system is connected.
    Each values correspond to following methods.

    See Also
    --------
    nqlib.IdealSystem.from_FF()
        Corresponds to `_ConnectionType.FF`.
    nqlib.IdealSystem.from_FB_connection_with_input_quantizer()
        Corresponds to `_ConnectionType.FB_WITH_INPUT_QUANTIZER`.
    nqlib.IdealSystem.from_FB_connection_with_output_quantizer()
        Corresponds to `_ConnectionType.FB_WITH_OUTPUT_QUANTIZER`.
    nqlib.IdealSystem()
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
                def new_function(u):
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
                self._function = new_function
            else:
                self._function = function

    def __call__(self, u):
        """
        Calls `self._function`.
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
            return ((u + d/2) // d) * d

        # limit the values
        if bit is None:
            function = q
        elif isinstance(bit, int) and bit > 0:
            def function(u):
                return _np.clip(
                    q(u),
                    a_min=-(2**(bit-1) - 1) * d,
                    a_max=2**(bit-1) * d,
                )
        else:
            raise TypeError('`bit` must be an natural number or `None`.')

        return StaticQuantizer(
            function=function,
            delta=d/2,
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
            return (u // d + 1/2) * d

        # limit the values
        if bit is None:
            function = q
        elif isinstance(bit, int) and bit > 0:
            def function(u):
                a_max = (2**(bit-1) - 1/2) * d
                return _np.clip(
                    q(u),
                    a_min=-a_max,
                    a_max=a_max,
                )
        else:
            raise TypeError('`bit` must be an natural number or `None`.')

        return StaticQuantizer(
            function=function,
            delta=d/2,
            error_on_excess=error_on_excess,
        )


class IsDone(_Enum):
    """
    Inherits from `enum.Enum`.
    This class represents current phase of
    `DynamicQuantizer.find_the_optimal_for()`.
    """
    NOT_YET = False
    ANALYTICALLY = 1
    NUMERICALLY = 2

    def __bool__(self) -> bool:
        if self == IsDone.NOT_YET:
            return False
        else:
            return True


def _lp(system: "IdealSystem",
        T: int,  # odd number
        gain_wv: float,
        solver: str) -> _np.ndarray:
    """
    Composes and solves linear problem to find good quantizer.

    Parameters
    ----------
    system : IdealSystem
    T : int
        Must be an odd number. This function doesn't check.
    solver : str
        Solver name to solve LP.

    Returns
    -------
    _np.ndarray
        Solution of LP.

    Raises
    ------
    cvxpy.SolverError
        if no solver named `solver` found.
    """
    if solver == '':
        solver = None

    m = system.m
    p = system.l

    A_tilde = system.A + system.B2 @ system.C2

    if gain_wv == inf:
        f = zeros((1 + m**2*T + p*m*(T-1), 1))
    else:
        f = zeros((1 + 2*m**2*T + p*m*(T-1), 1))
    f[0, 0] = 1

    # compose Phi
    Phi = zeros((m*p*(T-1), m*m*T))
    for i in range(1, T):
        Phi_dash = kron(
            system.C1 @ mpow(A_tilde, i-1) @ system.B2, eye(m)
        )
        for j in range(i, T):   # TODO: Any easier way?
            Phi[(j-1)*m*p: j*m*p,  (j-i)*m*m: (j-i+1)*m*m] = Phi_dash

    # making 'matrix -> vector' transposer
    eye_sumE = kron(
        ones((1, m*(T-1))), eye(p)
    )
    eye_sumH = kron(
        ones((1, m*T)), eye(m)
    )

    # finalize
    if gain_wv == inf:
        A = block([
            [-ones((p, 1)),          zeros((p, m*m*T)),  eye_sumE],
            [zeros((p*m*(T-1), 1)),  Phi,                   -eye(m*p*(T-1))],
            [zeros((p*m*(T-1), 1)), -Phi,                   -eye(m*p*(T-1))],
        ])
    else:
        A = block([
            [-ones((p, 1)),          zeros((p, m*m*T)),  zeros((p, m*m*T)),          eye_sumE],
            [zeros((p*m*(T-1), 1)),  Phi,                    zeros((p*m*(T-1), m*m*T)), -eye(m*p*(T-1))],
            [zeros((p*m*(T-1), 1)), -Phi,                    zeros((p*m*(T-1), m*m*T)), -eye(m*p*(T-1))],
            [zeros((m, 1)),          zeros((m, m*m*T)),  eye_sumH,                       zeros((m, m*p*(T-1)))],
            [zeros((m*m*T, 1)),      eye(m*m*T),        -eye(m*m*T),                 zeros((m*m*T, m*p*(T-1)))],
            [zeros((m*m*T, 1)),     -eye(m*m*T),        -eye(m*m*T),                 zeros((m*m*T, m*p*(T-1)))],
        ])

    # making C A^k B ,changing matrix to vector
    CAB = zeros(((T-1)*p, m))
    el_CAB = zeros((m*p*(T-1), 1))

    for i in range(1, T):
        CAkB = system.C1 @ mpow(A_tilde, i) @ system.B2
        CAB[(i-1)*p:i*p, 0:m] = CAkB

    for j in range(1, p*(T-1)+1):
        for i in range(1, m+1):
            el_CAB[i+(j-1)*m-1, 0] = CAB[j-1, i-1]

    if gain_wv == inf:
        b = block([
            [-abs(system.C1@system.B2) @ ones((m, 1))],
            [-el_CAB],
            [el_CAB]
        ])
    else:
        b = block([
            [-abs((system.C1@system.B2)@ones((m, 1)))],
            [-el_CAB],
            [el_CAB],
            [(gain_wv-1)*ones((m, 1))],
            [zeros((m*m*T*2, 1))],
        ])

    # solve LP
    x = cvxpy.Variable((f.shape[0], 1))
    objective = cvxpy.Minimize(f.transpose() @ x)
    constraints = [A @ x <= b]
    problem = cvxpy.Problem(objective, constraints)
    try:
        problem.solve(solver=solver)
    except cvxpy.SolverError as e:
        raise cvxpy.SolverError(f"Error from CVXPY.\n{str(e)}")

    return matrix(x.value)


def _SVD_from(x: _np.ndarray,
              T: int,
              m: int,
              p: int) -> _np.ndarray:
    T_dash = math.floor(T/2) + 1

    H_2 = []  # make list of H*_2i
    for t in range(T):
        H_2.append(
            block([
                [x[1 + (t*m + row)*m:1 + (t*m + row + 1)*m, :].T for row in range(m)],
            ])
        )

    # make Hankel matrix
    try:
        H = block([
            H_2[row:row + T_dash] for row in range(T_dash)
        ])
    except:
        # TODO: rewrite
        H = [H_2[row:row + T_dash] for row in range(T_dash)]
        H[-1].append(H[-1][0] * 0)
        H = block(H)

    # singular value decomposition
    # find Wo, S, Wc, with which it becomes H = Wo S Wc
    Wo, S, Wc = _np.linalg.svd(H)
    Wo = matrix(Wo)
    S = matrix(_np.diag(S))
    Wc = matrix(Wc)

    if norm(H - Wo@S@Wc) / norm(H) > 0.01:
        raise ValueError('SVD failed. Aborting Program...')

    return H, Wo, S, Wc


def _compose_Q_from_SVD(system: "IdealSystem",
                        q: StaticQuantizer,
                        T: int,
                        H: _np.ndarray,
                        Wo: _np.ndarray,
                        S: _np.ndarray,
                        Wc: _np.ndarray,
                        dim: int,
                        ) -> Tuple["DynamicQuantizer", bool]:
    """
    Parameters
    ----------
    system : IdealSystem
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
        Returned bool means if dimension is reduced.
    """
    m = system.m
    T_dash = int(H.shape[0] / m)

    # set dimension
    reduced = False
    if dim < m*T_dash:  # needs reduction
        nQ = dim
        reduced = True
    else:
        nQ = m*T_dash

    # reduce dimension
    Wo_reduced = Wo[:, :nQ]
    S_reduced = S[:nQ, :nQ]
    Wc_reduced = Wc[:nQ, :]
    # H_reduced = Wo_reduced @ S_reduced @ Wc_reduced

    # compose Q
    S_r_sqrt = mpow(S_reduced, 1/2)
    B2 = S_r_sqrt @ Wc_reduced @ block([
        [eye(m)],
        [zeros((m*(T_dash-1), m))],
    ])
    C = block([
        [eye(m), zeros((m, m*(T_dash - 1)))]
    ]) @ Wo_reduced @ S_r_sqrt
    A = pinv(
        block([
            [eye(m*(T_dash-1)), zeros((m*(T_dash-1), m))]
        ])@Wo_reduced@S_r_sqrt
    )@block([
        [zeros((m*(T_dash-1), m)), eye(m*(T_dash-1))]
    ]) @ Wo_reduced @ S_r_sqrt - B2 @ C

    # check stability
    Q = DynamicQuantizer(A, B2, C, q)
    if Q.is_stable:
        return Q, reduced
    else:
        # unstable case
        nQ_bar = nQ*T

        A_bar = block([
            [zeros((nQ_bar - nQ, nQ)), kron(eye(T-1), A + B2 @ C)],
            [kron(ones((1, T)), -B2@C)],
        ])
        B2_bar = block([
            [zeros((nQ_bar-nQ, m))],
            [B2],
        ])
        C_bar = kron(ones((1, T)), C)
        Q = DynamicQuantizer(A_bar, B2_bar, C_bar, q)
        return Q, reduced


def _nq_numeric(system: "IdealSystem",
                q: StaticQuantizer,
                T: int,
                T_solve: int,
                gain_wv: float,
                dim: int,
                verbose: bool,
                solver: str) -> Tuple[IsDone, "DynamicQuantizer", float]:
    """
    Finds the stable and optimal dynamic quantizer for `system`
    numerically.

    Parameters
    ----------
    system : IdealSystem
    q : StaticQuantizer
    T : int
    T_solve : int
    gain_wv : float
    dim : int
    verbose : bool
    solver : str

    Returns
    -------
    (isDone, Q, E) : Tuple[IsDone, DynamicQuantizer, float]
    """
    if verbose:
        print("\nCalculating numerically...")
    if T is None:
        # TODO: support infinity evaluation time
        return IsDone.NOT_YET, None, inf
    if dim is None:
        if verbose:
            print("This may take very long time. Please wait or interrupt.")
    elif T*dim > 1000:
        if verbose:
            print("This may take very long time. Please wait or interrupt.")
    # STEP1. LP
    start_time_lp = time.time()  # To measure time.
    x = _lp(system, T_solve, gain_wv, solver)
    E = x[0, 0] * q.delta
    end_time_lp = time.time()  # To measure time.
    if verbose:
        print(f"Solved linear programming problem in {end_time_lp - start_time_lp:.3f}[s].")

    # STEP2. SVD
    start_time_SVD = time.time()  # To measure time.
    H, Wo, S, Wc = _SVD_from(x, T, system.m, system.l)
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
    if verbose:
        print("Success!\n")
    return IsDone.NUMERICALLY, Q, E


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
        return M * 0 != M

    for t in range(tau_max):
        M = C_1 @ mpow(A_tilde, t) @ B_2
        if is_not_zero(M):
            if _np.linalg.matrix_rank(M) == l:
                return t
            else:
                return -1
    return -1


def _nq_optimal(system: "IdealSystem",
                q: StaticQuantizer,
                T: int,
                gain_wv: float,
                dim: int,
                verbose: bool) -> Tuple[IsDone, "DynamicQuantizer", float]:
    """
    Finds the stable and optimal dynamic quantizer for `system`
    analytically[1]_.

    Parameters
    ----------
    system : IdealSystem
    q : StaticQuantizer
    T : int
    gain_wv : float
    dim : int
    verbose : bool

    Returns
    -------
    (isDone, Q, E) : Tuple[IsDone, DynamicQuantizer, float]
    """
    if verbose:
        print("Trying to calculate optimal dynamic quantizer...")
    A_tilde = system.A + system.B2 @ system.C2  # convert to closed loop
    # S2
    tau = _find_tau(A_tilde, system.C1, system.B2, system.l, 10000)

    if tau == -1:
        if verbose:
            print("Couldn't calculate optimal dynamic quantizer. Trying other method...")
        return IsDone.NOT_YET, None, inf

    Q = DynamicQuantizer(
        A=A_tilde,
        B=system.B2,
        C=-pinv(system.C1 @ mpow(A_tilde, tau) @ system.B2) @ system.C1 @ mpow(A_tilde, tau+1),
        q=q,
    )

    E = norm(abs(system.C1 @ mpow(A_tilde, tau) @ system.B2)) * q.delta

    if not Q.is_stable:
        if verbose:
            print("Optimal dynamic quantizer is unstable. Trying other method...")
        return IsDone.NOT_YET, None, inf
    else:
        if verbose:
            print("Success!")
        return IsDone.ANALYTICALLY, Q, E


def _nq_serial_decomposition(system: "IdealSystem",
                             q: StaticQuantizer,
                             T: int,
                             gain_wv: float,
                             dim: int,
                             verbose: bool) -> Tuple[IsDone, "DynamicQuantizer", float]:
    """
    Finds the stable and optimal dynamic quantizer for `system`
    using serial decomposition[1]_.

    Parameters
    ----------
    system : IdealSystem
    q : StaticQuantizer
    T : int
    gain_wv : float
    verbose : bool
    dim : int

    Returns
    -------
    (isDone, Q, E) : Tuple[IsDone, DynamicQuantizer, float]
    """
    if verbose:
        print("Trying to calculate quantizer using serial system decomposition...")
    tf = system.P.tf1
    zeros_ = _ctrl.zero(tf)
    poles = _ctrl.pole(tf)

    unstable_zeros = [zero for zero in zeros_ if abs(zero) > 1]

    z = _ctrl.TransferFunction.z
    z.dt = tf.dt

    if len(unstable_zeros) == 0:
        n_time_delay = len([p for p in poles if p == 0])  # count pole-zero
        G = 1 / z**n_time_delay
        F = tf/G
        F_ss = _ctrl.tf2ss(F)
        B_F = matrix(F_ss.B)
        C_F = matrix(F_ss.C)

        E = abs(C_F@B_F)[0, 0] * q.delta
    elif len(unstable_zeros) == 1:
        i = len(poles) - len(zeros_)  # relative order
        if i < 1:
            return IsDone.NOT_YET, None, inf
        a = unstable_zeros[0]
        G = (z - a) / z**i
        F = _ctrl.minreal(tf / G, verbose=False)
        F_ss = _ctrl.tf2ss(F)
        B_F = matrix(F_ss.B)
        C_F = matrix(F_ss.C)

        E = (1 + abs(a)) * abs(C_F@B_F)[0, 0] * q.delta
    else:
        return IsDone.NOT_YET, None, inf

    A_F = matrix(F_ss.A)
    D_F = matrix(F_ss.D)

    # check
    if (C_F@B_F)[0, 0] == 0:
        if verbose:
            print("CF @ BF == 0 became true. Couldn't calculate by using serial system decomposition.")
        return IsDone.NOT_YET, None, inf
    if D_F[0, 0] != 0:
        if verbose:
            print("DF == 0 became true. Couldn't calculate by using serial system decomposition.")
        return IsDone.NOT_YET, None, inf
    Q = DynamicQuantizer(
        A=A_F,
        B=B_F,
        C=- 1/(C_F@B_F)[0, 0]*C_F@A_F,
        q=q
    )
    if verbose:
        print("Success!")
    return IsDone.ANALYTICALLY, Q, E


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
           quantizers for discrete-valued input control; IEEE Transactions
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
                A_i = A_i @ (self.A + self.B@self.C)

        return ret

    def _objective_function(self,
                            system: "IdealSystem",
                            *,
                            T: int = None,  # TODO: これより下を反映
                            gain_wv: float = inf) -> float:
        """
        Used in numerical optimization.

        Parameters
        ----------
        system : IdealSystem
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
        .. [1] Y. Minami: Design of model following control systems with
           discrete-valued signal constraints;International Journal of Control,
           Automationand Systems, Vol. 14, pp. 331–339 (2016)
        """
        # if T is None:
        #     # TODO: support infinity evaluation time
        #     return None, inf
        # TODO: siso のチェック

        # Values representing the stability or gain_wv.y
        # if max(constraint_values) < 0, this quantizer satisfies the
        # constraints.
        constraint_values = [
            eig_max(self.A+self.B@self.C) - 1,
        ]
        if not _np.isposinf(gain_wv):
            constraint_values.append(self.gain_wv(T) - gain_wv)

        max_v = max(constraint_values)
        if max_v < 0:
            return - 1.1 ** (- system.E(self))
        else:
            return max_v

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
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        if eig_max(self.A+self.B@self.C) > 1 - 1e-8:
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
    def find_the_optimal_for(system: "IdealSystem",
                             *,
                             q: StaticQuantizer,
                             T: int = None,
                             gain_wv: float = inf,
                             dim: int = None,
                             verbose: bool = False,
                             solver: str = "") -> Tuple[IsDone, "DynamicQuantizer", float]:
        """
        Calculates the stable and optimal dynamic quantizer `Q` for `system`.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        Parameters
        ----------
        system : IdealSystem
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
            Upper limit of dimension of `Q`. Must be greater than `0`.
            (The default is `None`, which means infinity).
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).
        solver : str, optional
            Name of CVXPY solver. You can check the available solvers by
            `nqlib.installed_solvers()`.
            (The default is `""`, which implies that this function doesn't
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
        .. [2]  Y. Minami, S. Azuma and T. Sugie:  An optimal dynamic quantizer
           for feedback control with discrete-valued signal constraints;2007
           46th IEEEConference on Decision and Control, pp.  2259–2264, IEEE
           (2007)
        .. [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会
           論文集，Vol. 52, pp. 46–51(2016)
        """
        # TODO: 最小実現する
        # check system
        if system.__class__.__name__ != "IdealSystem":
            raise TypeError(
                '`system` must be an instance of `nqlib.IdealSystem`.'
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

        # check T and choose method
        if T is None or T == inf:
            use_analytic_method = True
            T_solve = T
        else:
            if T % 2 == 0:
                T_solve = T + 1
            else:
                T_solve = T
            use_analytic_method = False

        # check gain
        if gain_wv < 1:
            raise ValueError(
                '`gain_wv` must be greater than or equal to `1`.'
            )

        # check dim
        if dim is None:
            dim = inf
        elif type(dim) is not int:
            raise TypeError('`dim` must be an integer.')
        elif dim < 1:
            raise ValueError('`dim` must be greater than `0`.')

        done = IsDone.NOT_YET
        if use_analytic_method:
            # TODO: Even if `T` is specified, use these methods,
            # check `gain` and `dim`, and return `Q`.
            if system.type == _ConnectionType.FF and system.P.tf1.issiso():
                # FF and SISO
                done, Q, E = _nq_serial_decomposition(system, q, T, gain_wv, dim, verbose)
            if not done and system.m >= system.l:
                done, Q, E = _nq_optimal(system, q, T, gain_wv, dim, verbose)
        if not done:  # numerically
            # TODO: case `T` is infinity
            if verbose:
                print("Couldn't calculate analytically.")
            done, Q, E = _nq_numeric(system,
                                     q,
                                     T,
                                     T_solve,
                                     gain_wv,
                                     dim,
                                     verbose,
                                     solver)

        # TODO: check gain, dim?
        return Q, E

    @staticmethod
    def ODQ(system: "IdealSystem",
            *,
            q: StaticQuantizer,
            T: int = None,
            gain_wv: float = inf,
            dim: int = None,
            verbose: bool = False,
            solver: str = "") -> Tuple["DynamicQuantizer", float]:
        """
        A shortened form of `DynamicQuantizer.find_the_optimal_for()`.

        Calculates the stable and optimal dynamic quantizer `Q` for `system`.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        Parameters
        ----------
        system : IdealSystem
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
            Upper limit of dimension of `Q`. Must be greater than `0`.
            (The default is `None`, which means infinity).
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).
        solver : str, optional
            Name of CVXPY solver. You can check the available solvers by
            `nqlib.installed_solvers()`.
            (The default is `""`, which implies that this function doesn't
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
        .. [2]  Y. Minami, S. Azuma and T. Sugie:  An optimal dynamic quantizer
           for feedback control with discrete-valued signal constraints;2007
           46th IEEEConference on Decision and Control, pp.  2259–2264, IEEE
           (2007)
        .. [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会
           論文集，Vol. 52, pp. 46–51(2016)
        """
        return DynamicQuantizer.find_the_optimal_for(system,
                                                     q=q,
                                                     T=T,
                                                     gain_wv=gain_wv,
                                                     dim=dim,
                                                     verbose=verbose,
                                                     solver=solver)

    @staticmethod
    def gradient_based(system: "IdealSystem",
                       *,
                       q: StaticQuantizer,
                       dim: int,
                       T: int = None,  # TODO: これより下を反映
                       gain_wv: float = inf,
                       verbose: bool = False,
                       method: str = "SLSQP") -> Tuple["DynamicQuantizer", float]:
        """
        Finds the stable and optimal dynamic quantizer `Q` for `system`.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        Parameters
        ----------
        system : IdealSystem
            Must be stable and SISO.
        q : StaticQuantizer
            Returned dynamic quantizer contains this static quantizer.
            `q.delta` is important to estimate E(Q).
        dim : int
            Dimension of `Q`. Must be greater than `0`.
        T : int, None or numpy.inf, optional
            Estimation time. Must be greater than `0`.
            (The default is `None`, which means infinity).
        gain_wv : float, optional
            Upper limit of gain w->v . Must be greater than `0`.
            (The default is `numpy.inf`).
        verbose : bool, optional
            Whether to print the details.
            (The default is `False`).
        method : str, optional  TODO: 削除
            Specifies which method should be used in
            `scipy.optimize.minimize()`.
            (The default is `""`, which implies that this function doesn't
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
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        .. [2]  Y. Minami, S. Azuma and T. Sugie:  An optimal dynamic quantizer
           for feedback control with discrete-valued signal constraints;2007
           46th IEEEConference on Decision and Control, pp.  2259–2264, IEEE
           (2007)
        .. [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会
           論文集，Vol. 52, pp. 46–51(2016)
        """  # TODO: ドキュメント更新
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
                [zeros((dim-1, 1)), eye(dim-1)],
                [-a(x)],
            ])
            _B = block([
                [zeros((dim-1, 1))],
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

        # optimize
        if verbose:
            print("Designing a quantizer with gradient-based optimization.")
            print(f"The optimization method is '{method}'.")
            print("### Message from `scipy.optimize.minimize()`. ###")
        result = _minimize(obj,
                           x0=zeros(2*dim)[0],
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
    def DE(system: "IdealSystem",
            *,
            q: StaticQuantizer,
            dim: int,
            T: int = None,  # TODO: これより下を反映
            gain_wv: float = inf,
            verbose: bool = False) -> Tuple["DynamicQuantizer", float]:  # TODO: method のデフォルトを決める
        """
        A shortened form of `DynamicQuantizer.find_the_optimal_for()`.

        Calculates the stable and optimal dynamic quantizer `Q` for `system`.
        Returns `(Q, E)`. `E` is the estimation of E(Q)[1]_,[2]_,[3]_.

        Parameters
        ----------
        system : IdealSystem
            Must be stable and SISO.
        q : StaticQuantizer
            Returned dynamic quantizer contains this static quantizer.
            `q.delta` is important to estimate E(Q).
        dim : int
            Dimension of `Q`. Must be greater than `0`.
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
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        .. [2]  Y. Minami, S. Azuma and T. Sugie:  An optimal dynamic quantizer
           for feedback control with discrete-valued signal constraints;2007
           46th IEEEConference on Decision and Control, pp.  2259–2264, IEEE
           (2007)
        .. [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会
           論文集，Vol. 52, pp. 46–51(2016)
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
                [zeros((dim-1, 1)), eye(dim-1)],
                [-a(x)],
            ])
            _B = block([
                [zeros((dim-1, 1))],
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
        print(bounds)
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
        # - 'currenttobest1exp'  いいかも
        # - 'rand2exp'  わるくはない
        # - 'currenttobest1bin'  いいかも 答えが求まりにくい？
        # - 'best2bin'  わるくない
        # - 'rand2bin'  わるくない
        # - 'rand1bin'  精度がいい？

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

            '       +-----+       '
            ' u --->|  Q  |---> v '
            '       +-----+       '

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
            v[:, i:i+1] = matrix(self.q(self.C @ xi[:, i:i+1] + u[:, i:i+1]))
            if i < length - 1:
                xi[:, i:i+1+1] = matrix(self.A @ xi[:, i:i+1] + self.B @ (v[:, i:i+1] - u[:, i:i+1]))
        return v
