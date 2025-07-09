import copy
from typing import Tuple, Union

import control as _ctrl
import numpy as _np
from numpy.typing import ArrayLike

from .linalg import block, eye, matrix, norm, zeros, eig_max
from .quantizer import DynamicQuantizer
from .quantizer import StaticQuantizer
from .quantizer import ConnectionType
from packaging.version import Version
from .types import (
    NDArrayNum, infint, InfInt,
    validate_int_or_inf,
)

if Version(_ctrl.__version__) >= Version("0.9.2"):
    # ctrl_poles = _ctrl.poles
    # ctrl_zeros = _ctrl.zeros
    pass
else:
    # ctrl_poles = _ctrl.pole
    # ctrl_zeros = _ctrl.zero
    _ctrl.use_numpy_matrix(False)  # type: ignore

__all__ = [
    'Controller',
    'Plant',
    'System',
]
# TODO: support control.input_output_response()
# TODO: define function returns control.InputOutputSystem


def _indent(s: str, num_space: int) -> str:
    ret = ""
    for line in s.splitlines():
        ret += " " * num_space + line + "\n"
    return ret[:-1]


class Controller(object):
    """
    State-space model of a controller K.

    The controller K is given by:

        K : { x(t+1) = A x(t) + B1 r(t) + B2 y(t)
            {  u(t)  = C x(t) + D1 r(t) + D2 y(t)

    References
    ----------
    [5] Y. Minami and T. Muromaki: Differential evolution-based
    synthesis of dynamic quantizers with fixed-structures; International
    Journal of Computational Intelligence and Applications, Vol. 15,
    No. 2, 1650008 (2016)

    Example
    -------
    >>> import nqlib
    >>> K = nqlib.Controller(A=0,
    ...                      B1=0,
    ...                      B2=[0, 0],
    ...                      C=0,
    ...                      D1=1,
    ...                      D2=[-20, -3])
    >>> K.A
    array([[0]])
    """        

    def __init__(self, A: ArrayLike, B1: ArrayLike, B2: ArrayLike, C: ArrayLike, D1: ArrayLike, D2: ArrayLike):
        """
        Initialize a Controller instance.

        Parameters
        ----------
        A : array_like
            State matrix (`n` x `n`), real.
        B1 : array_like
            Input matrix for r (`n` x `l`), real.
        B2 : array_like
            Input matrix for y (`n` x `p`), real.
        C : array_like
            Output matrix (`m` x `n`), real.
        D1 : array_like
            Feedthrough matrix for r (`m` x `l`), real.
        D2 : array_like
            Feedthrough matrix for y (`m` x `p`), real.

        Raises
        ------
        TypeError
            If any argument cannot be interpreted as a matrix.
        ValueError
            If matrix dimensions are inconsistent.

        Example
        -------
        >>> import nqlib
        >>> K = nqlib.Controller(A=0,
        ...                      B1=0,
        ...                      B2=[0, 0],
        ...                      C=0,
        ...                      D1=1,
        ...                      D2=[-20, -3])
        >>> K.B1
        array([[0]])
        """
        try:
            A_mat = matrix(A)
            B1_mat = matrix(B1)
            B2_mat = matrix(B2)
            C_mat = matrix(C)
            D1_mat = matrix(D1)
            D2_mat = matrix(D2)
        except:
            raise TypeError(
                "`A`, `B1`, `B2`, `C`, `D1`, and "
                "`D2` must be interpreted as matrices."
            )

        self.n = A_mat.shape[0]
        self.l = B1_mat.shape[1]
        self.p = B2_mat.shape[1]
        self.m = C_mat.shape[0]

        if A_mat.shape != (self.n, self.n):
            raise ValueError("A must be a square matrix.")
        if B1_mat.shape != (self.n, self.l):
            raise ValueError(
                "The number of rows in matrices "
                "`A` and `B1` must be the same."
            )
        if B2_mat.shape != (self.n, self.p):
            raise ValueError(
                "The number of rows in matrices "
                "`A` and `B2` must be the same."
            )
        if C_mat.shape != (self.m, self.n):
            raise ValueError(
                "The number of columns in matrices "
                "`A` and `C` must be the same."
            )
        if D1_mat.shape[0] != self.m:
            raise ValueError(
                "The number of rows in matrices "
                "`C` and `D1` must be the same."
            )
        if D1_mat.shape[1] != self.l:
            raise ValueError(
                "The number of columns in matrices "
                "`B1` and `D1` must be the same."
            )
        if D2_mat.shape[0] != self.m:
            raise ValueError(
                "The number of rows in matrices "
                "`C` and `D2` must be the same."
            )
        if D2_mat.shape[1] != self.p:
            raise ValueError(
                "The number of columns in matrices "
                "`B2` and `D2` must be the same."
            )

        self.A = A_mat
        self.B1 = B1_mat
        self.B2 = B2_mat
        self.C = C_mat
        self.D1 = D1_mat
        self.D2 = D2_mat

    def __repr__(self) -> str:
        return (
            f"nqlib.Controller(\n"
            f"  A ={_indent(_np.array_repr(self.A), 5)[5:]},\n"
            f"  B1={_indent(_np.array_repr(self.B1), 5)[5:]},\n"
            f"  B2={_indent(_np.array_repr(self.B2), 5)[5:]},\n"
            f"  C ={_indent(_np.array_repr(self.C), 5)[5:]},\n"
            f"  D1={_indent(_np.array_repr(self.D1), 5)[5:]},\n"
            f"  D2={_indent(_np.array_repr(self.D2), 5)[5:]},\n"
            f")"
        )


class Plant(object):
    """
    State-space model of a plant P.

    The plant P is given by:

        P : { x(t+1) =  A x(t) + B u(t)
            {  z(t)  = C1 x(t)
            {  y(t)  = C2 x(t)

    Example
    -------
    >>> import nqlib
    >>> import numpy as np
    >>> P = nqlib.Plant(A=[[1.15, 0.05],
    ...                    [0, 0.99]],
    ...                 B=[[0.004],
    ...                    [0.099]],
    ...                 C1=[1, 0],
    ...                 C2=np.eye(2))
    >>> P.A
    array([[1.15, 0.05],
           [0.  , 0.99]])
    """

    def __init__(self, A: ArrayLike, B: ArrayLike, C1: ArrayLike, C2: ArrayLike):
        """
        Initialize a Plant instance.

        Parameters
        ----------
        A : array_like
            State matrix (`n` x `n`), real.
        B : array_like
            Input matrix (`n` x `m`), real.
        C1 : array_like
            Output matrix for z (`l1` x `n`), real.
        C2 : array_like
            Output matrix for y (`l2` x `n`), real.

        Raises
        ------
        TypeError
            If any argument cannot be interpreted as a matrix.
        ValueError
            If matrix dimensions are inconsistent.

        References
        ----------
        [5] Y. Minami and T. Muromaki: Differential evolution-based
        synthesis of dynamic quantizers with fixed-structures; International
        Journal of Computational Intelligence and Applications, Vol. 15,
        No. 2, 1650008 (2016)

        Example
        -------
        >>> import nqlib
        >>> import numpy as np
        >>> P = nqlib.Plant(A=[[1.15, 0.05],
        ...                    [0, 0.99]],
        ...                 B=[[0.004],
        ...                    [0.099]],
        ...                 C1=[1, 0],
        ...                 C2=np.eye(2))
        >>> P.A
        array([[1.15, 0.05],
               [0.  , 0.99]])
        """
        try:
            A_mat = matrix(A)
            B_mat = matrix(B)
            C1_mat = matrix(C1)
            C2_mat = matrix(C2)
        except:
            raise TypeError(
                "`A`, `B`, `C1` and `C2` must be interpreted as "
                "matrices."
            )

        self.n = A_mat.shape[0]
        self.m = B_mat.shape[1]
        self.l1 = C1_mat.shape[0]
        self.l2 = C2_mat.shape[0]

        if A_mat.shape != (self.n, self.n):
            raise ValueError("A must be a square matrix.")
        if B_mat.shape != (self.n, self.m):
            raise ValueError(
                "The number of rows in matrices "
                "`A` and `B` must be the same."
            )
        if C1_mat.shape != (self.l1, self.n):
            raise ValueError(
                "The number of columns in matrices "
                "`A` and `C1` must be the same."
            )
        if C2_mat.shape != (self.l2, self.n):
            raise ValueError(
                "The number of columns in matrices "
                "`A` and `C2` must be the same."
            )

        self.A = A_mat
        self.B = B_mat
        self.C1 = C1_mat
        self.C2 = C2_mat

        ss1 = _ctrl.ss(
            self.A, self.B, self.C1, zeros((self.C1.shape[0], self.B.shape[1]))
        )
        ss2 = _ctrl.ss(
            self.A, self.B, self.C2, zeros((self.C2.shape[0], self.B.shape[1]))
        )
        self.tf1: _ctrl.TransferFunction = _ctrl.ss2tf(ss1)
        self.tf2: _ctrl.TransferFunction = _ctrl.ss2tf(ss2)

    @staticmethod
    def from_TF(tf: _ctrl.TransferFunction) -> "Plant":
        """
        Create a Plant instance from a transfer function.

        Parameters
        ----------
        tf : control.TransferFunction
            Transfer function from input `u` to output `z`.

        Returns
        -------
        Plant
            Plant instance with `C2` set to zero.

        Example
        -------
        >>> import nqlib
        >>> import control
        >>> tf = control.TransferFunction([1], [1, 2, 1], 1)  # 1 / (s^2 + 2s + 1)
        >>> P = nqlib.Plant.from_TF(tf)
        >>> P.A.shape[0]
        2
        """
        ss: _ctrl.StateSpace = _ctrl.tf2ss(tf)  # type: ignore
        _C = matrix(ss.C)
        ret = Plant(
            A=matrix(ss.A),
            B=matrix(ss.B),
            C1=_C,
            C2=matrix(zeros(_C.shape)),
        )
        ret.tf1 = tf
        ret.tf2 = tf * 0
        return ret

    def __repr__(self) -> str:
        return (
            f"nqlib.Plant(\n"
            f"  A ={_indent(_np.array_repr(self.A), 5)[5:]},\n"
            f"  B ={_indent(_np.array_repr(self.B), 5)[5:]},\n"
            f"  C1={_indent(_np.array_repr(self.C1), 5)[5:]},\n"
            f"  C2={_indent(_np.array_repr(self.C2), 5)[5:]},\n"
            f")"
        )


class System():
    """
    State-space model of an ideal system.
    Ideal here means that the system does not have a quantizer.

    The system is given by:

        G : { x(t+1) =  A x(t) + B1 r(t) + B2 v(t)
            {  z(t)  = C1 x(t) + D1 r(t)
            {  u(t)  = C2 x(t) + D2 r(t)

    Example
    -------
    >>> import nqlib
    >>> G = nqlib.System(
    ...     A=[[1.15, 0.05],
    ...        [0.00, 0.99]],
    ...     B1=[[0.],
    ...         [0.]],
    ...     B2=[[0.004],
    ...         [0.099]],
    ...     C1=[1., 0.],
    ...     C2=[-15., -3.],
    ...     D1=0,
    ...     D2=1,
    ... )
    >>> G.A
    array([[1.15, 0.05],
           [0.  , 0.99]])
    """

    def __str__(self) -> str:
        def equation_with_matrix(left_side: str, matrix: _np.ndarray):
            line_matrix = str(matrix).split("\n")
            ret = left_side + " = "
            indent_size = len(ret)
            ret += line_matrix[0] + "\n"
            for _ in range(1, len(line_matrix)):
                ret += (" " * indent_size) + line_matrix[_] + "\n"
            return ret
        ret = "System given by\n"
        ret += "  { x(k+1) = A  x(k) + B1 r(k) + B2 v(k)\n"
        ret += "G {   z(k) = C1 x(k) + D1 r(k)\n"
        ret += "  {   u(k) = C2 x(k) + D2 r(k)\n"
        ret += "where\n"
        ret += "v = u\n"
        ret += equation_with_matrix(" A", self. A)
        ret += equation_with_matrix("B1", self.B1)
        ret += equation_with_matrix("B2", self.B2)
        ret += equation_with_matrix("C1", self.C1)
        ret += equation_with_matrix("C2", self.C2)
        ret += equation_with_matrix("D1", self.D1)
        ret += equation_with_matrix("D2", self.D2)
        ret += "shown in the following figure\n"
        ret += "       +-----------+       \n"
        ret += "  r -->|           |--> z  \n"
        ret += "       |     G     |       \n"
        ret += "   +-->|           |---+   \n"
        ret += "   |   +-----------+   |   \n"
        ret += "   +-------------------+   \n"
        ret += "           u = v           \n"
        return ret

    def __repr__(self) -> str:
        return (
            f"nqlib.System(\n"
            f"  A ={_indent(_np.array_repr(self.A), 5)[5:]},\n"
            f"  B1={_indent(_np.array_repr(self.B1), 5)[5:]},\n"
            f"  B2={_indent(_np.array_repr(self.B2), 5)[5:]},\n"
            f"  C1={_indent(_np.array_repr(self.C1), 5)[5:]},\n"
            f"  C2={_indent(_np.array_repr(self.C2), 5)[5:]},\n"
            f"  D1={_indent(_np.array_repr(self.D1), 5)[5:]},\n"
            f"  D2={_indent(_np.array_repr(self.D2), 5)[5:]},\n"
            f")"
        )

    def response_with_quantizer(
        self,
        quantizer: Union[DynamicQuantizer, StaticQuantizer],
        input: ArrayLike,
        x_0: ArrayLike,
    ) -> Tuple[NDArrayNum, NDArrayNum, NDArrayNum, NDArrayNum]:
        """
        Simulate the system with a quantizer and return results.

        Parameters
        ----------
        quantizer : DynamicQuantizer or StaticQuantizer
            Quantizer to use in the simulation.
        input : array_like
            Input signal (reference `r`). Shape: (`p`, `length`).
        x_0 : array_like
            Initial state vector. Shape: (`n`, 1).

        Returns
        -------
        t : np.ndarray
            Time steps. Shape: (`1`, `length`).
        u : np.ndarray
            Input to quantizer. Shape: (`m`, `length`).
        v : np.ndarray
            Quantized input. Shape: (`m`, `length`).
        z : np.ndarray
            Output signal. Shape: (`l`, `length`).

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> import numpy as np
        >>> sys = nqlib.System(1, 1, 1, 1, 1, 1, 1)
        >>> q = nqlib.StaticQuantizer.mid_tread(0.1)
        >>> t, u, v, z = sys.response_with_quantizer(q, np.ones((1, 5)), np.zeros((1, 1)))
        >>> t.shape
        (1, 5)
        """
        _quantizer: DynamicQuantizer = DynamicQuantizer(
            zeros((1, 1)), zeros((1, self.m)),
            zeros((self.m, 1)),
            q=quantizer,
        ) if type(quantizer) is StaticQuantizer else quantizer  # type: ignore
        A = _quantizer.A
        B = _quantizer.B
        C = _quantizer.C
        q = _quantizer.q
        # TODO: support xP_0, xK_0
        # TODO: support time
        r = matrix(input)
        length = r.shape[1]
        k = matrix(range(0, length))

        z = zeros((self.l, length))
        u = zeros((self.m, length))
        v = copy.deepcopy(u)
        x = zeros((len(x_0), length))  # type: ignore
        xi = zeros((len(A), length))  # type: ignore
        x[:, 0:1] = matrix(x_0)

        for i in range(length):
            u[:, i:i + 1] = matrix(self.C2 @ x[:, i:i + 1] + self.D2 @ r[:, i:i + 1])
            v[:, i:i + 1] = matrix(q(C @ xi[:, i:i + 1] + u[:, i:i + 1]))
            z[:, i:i + 1] = matrix(self.C1 @ x[:, i:i + 1] + self.D1 @ r[:, i:i + 1])
            if i < length - 1:
                xi[:, i + 1:i + 2] = matrix(A @ xi[:, i:i + 1] + B @ (v[:, i:i + 1] - u[:, i:i + 1]))
                x[:, i + 1:i + 2] = matrix(self.A @ x[:, i:i + 1] + self.B1 @ r[:, i:i + 1] + self.B2 @ v[:, i:i + 1])
        return k, u, v, z

    def response(self, input: ArrayLike, x_0: ArrayLike) -> Tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """
        Simulate the system and return results.

        Parameters
        ----------
        input : array_like
            Input signal (reference `r`). Shape: (`p`, `length`).
        x_0 : array_like
            Initial state vector. Shape: (`n`, 1).

        Returns
        -------
        t : np.ndarray
            Time steps. Shape: (`1`, `length`).
        u : np.ndarray
            Input signal to plant. Shape: (`m`, `length`).
        z : np.ndarray
            Output signal. Shape: (`l`, `length`).

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> import numpy as np
        >>> sys = nqlib.System(1, 1, 1, 1, 1, 1, 1)
        >>> t, u, z = sys.response(np.ones((1, 5)), np.zeros((1, 1)))
        >>> t.shape
        (1, 5)
        """
        # TODO: support xP_0, xK_0
        # TODO: support time
        r = matrix(input)
        length = r.shape[1]
        k = matrix(range(0, length))

        z = zeros((self.l, length))
        u = zeros((self.m, length))
        x = zeros((len(x_0), length))  # type: ignore
        x[:, 0:1] = matrix(x_0)

        for i in range(length):
            u[:, i:i + 1] = matrix(self.C2 @ x[:, i:i + 1] + self.D2 @ r[:, i:i + 1])
            z[:, i:i + 1] = matrix(self.C1 @ x[:, i:i + 1] + self.D1 @ r[:, i:i + 1])
            if i < length - 1:
                x[:, i + 1:i + 2] = matrix(
                    self.A @ x[:, i:i + 1] + self.B1 @ r[:, i:i + 1] + self.B2 @ u[:, i:i + 1]
                )
        return k, u, z

    def __init__(self, A: ArrayLike, B1: ArrayLike, B2: ArrayLike, C1: ArrayLike, C2: ArrayLike, D1: ArrayLike, D2: ArrayLike):
        """
        Initialize a System instance (ideal system without quantizer).

        Parameters
        ----------
        A : array_like
            State matrix (`n` x `n`), real.
        B1 : array_like
            Input matrix for r (`n` x `p`), real.
        B2 : array_like
            Input matrix for v (`n` x `m`), real.
        C1 : array_like
            Output matrix for z (`l` x `n`), real.
        C2 : array_like
            Output matrix for u (`m` x `n`), real.
        D1 : array_like
            Feedthrough matrix for r to z (`l` x `p`), real.
        D2 : array_like
            Feedthrough matrix for r to u (`m` x `p`), real.

        Raises
        ------
        TypeError
            If any argument cannot be interpreted as a matrix.
        ValueError
            If matrix dimensions are inconsistent.

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> G = nqlib.System(
        ...     A=[[1.15, 0.05],
        ...        [0.00, 0.99]],
        ...     B1=[[0.],
        ...         [0.]],
        ...     B2=[[0.004],
        ...         [0.099]],
        ...     C1=[1., 0.],
        ...     C2=[-15., -3.],
        ...     D1=0,
        ...     D2=1,
        ... )
        >>> G.A
        array([[1.15, 0.05],
               [0.  , 0.99]])
        """
        try:
            A_mat = matrix(A)
            B1_mat = matrix(B1)
            B2_mat = matrix(B2)
            C1_mat = matrix(C1)
            C2_mat = matrix(C2)
            D1_mat = matrix(D1)
            D2_mat = matrix(D2)
        except:
            raise TypeError(
                "`A`, `B1`, `B2`, `C1`, `C2`, `D1`, and "
                "`D2` must be interpreted as matrices."
            )

        self.n = A_mat.shape[0]
        self.p = B1_mat.shape[1]
        self.m = B2_mat.shape[1]
        self.l = C1_mat.shape[0]

        if A_mat.shape != (self.n, self.n):
            raise ValueError("A must be a square matrix.")
        if B1_mat.shape != (self.n, self.p):
            raise ValueError(
                "The number of rows in matrices "
                "`A` and `B1` must be the same."
            )
        if B2_mat.shape != (self.n, self.m):
            raise ValueError(
                "The number of rows in matrices "
                "`A` and `B2` must be the same."
            )
        if C1_mat.shape != (self.l, self.n):
            raise ValueError(
                "The number of columns in matrices "
                "`A` and `C1` must be the same."
            )
        if C2_mat.shape != (self.m, self.n):
            raise ValueError(
                "The sizes of `B2` and the transpose of `C2` must be"
                "the same."
            )
        if D1_mat.shape[0] != self.l:
            raise ValueError(
                "The number of rows in matrices "
                "`C1` and `D1` must be the same."
            )
        if D1_mat.shape[1] != self.p:
            raise ValueError(
                "The number of columns in matrices "
                "`B1` and `D1` must be the same."
            )
        if D2_mat.shape[0] != self.m:
            raise ValueError(
                "The number of rows in matrices "
                "`C2` and `D2` must be the same."
            )
        if D2_mat.shape[1] != self.p:
            raise ValueError(
                "The number of columns in matrices "
                "`B1` and `D2` must be the same."
            )

        self.A = A_mat
        self.B1 = B1_mat
        self.B2 = B2_mat
        self.C1 = C1_mat
        self.C2 = C2_mat
        self.D1 = D1_mat
        self.D2 = D2_mat

        # 解析解用に，内部システムとその接続の仕方を保存
        self.type = ConnectionType.ELSE
        self.P: Plant | None = None
        self.K: Controller | None = None

    @staticmethod
    def from_FF(P: Plant) -> "System":
        """
        Create a System instance from a Plant (feedforward connection).

        'from_FF' means that a quantizer is inserted as shown in the
        following figure.

        ..  code-block:: none

            |       ┌─────┐  v  ┌─────┐
            | u ───>│  Q  ├────>│  P  ├───> z
            |       └─────┘     └─────┘

        Parameters
        ----------
        P : Plant
            Plant instance. The plant to which the quantizer is connected in feedforward.

        Returns
        -------
        System
            System with quantizer inserted before plant (feedforward).

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> P = nqlib.Plant(
        ...     A =[[ 1.8, 0.8],
        ...         [-1.0, 0. ]],
        ...     B =[[1.],
        ...         [0.]],
        ...     C1 =[0.01, -0.09],
        ...     C2 =[0, 0],
        ... )
        >>> G = nqlib.System.from_FF(P)
        >>> G.is_stable
        True
        """
        n = P.A.shape[0]
        m = P.B.shape[1]
        l = P.C1.shape[0]

        ret = System(
            A=P.A,
            B1=zeros(shape=(n, m)),
            B2=P.B,
            C1=P.C1,
            C2=zeros(shape=(m, n)),
            D1=zeros(shape=(l, m)),
            D2=eye(m, m),
        )
        ret.type = ConnectionType.FF
        ret.P = P
        return ret

    @staticmethod
    def from_FB_connection_with_input_quantizer(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        Create a System instance from Plant and Controller with input quantizer (feedback connection).

        'from_FB_connection_with_input_quantizer' means that a
        quantizer is inserted as shown in the following figure.
        
        ..  code-block:: none

            |       ┌───────┐     ┌───────┐     ┌───────┐       
            | r ───>│       │  u  │       │  v  │       ├───> z 
            |       │   K   ├────>│   Q   ├────>│   P   │       
            |    ┌─>│       │     │       │     │       ├──┐    
            |    │  └───────┘     └───────┘     └───────┘  │ y  
            |    └─────────────────────────────────────────┘    

        Parameters
        ----------
        P : Plant
            Plant instance. The plant in the feedback loop.
        K : Controller
            Controller instance. The controller in the feedback loop.

        Returns
        -------
        System
            System with quantizer inserted at controller input (feedback).

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> P = nqlib.Plant(0.5, 1, 1, 1)
        >>> K = nqlib.Controller(0, 1, 1, 1, 1, 1)
        >>> sys = nqlib.System.from_FB_connection_with_input_quantizer(P, K)
        >>> sys.is_stable
        False
        """
        if P.l2 != K.p:
            raise ValueError(
                "The number of columns in matrix `P.C2` and the "
                "number of rows in matrix `K.B2` must be the same."
            )

        A = block([
            [P.A, zeros(shape=(P.A.shape[0], K.A.shape[1]))],
            [K.B2 @ P.C2, K.A],
        ])
        B1 = block([
            [zeros(shape=(P.A.shape[0], K.B1.shape[1]))],
            [K.B1],
        ])
        B2 = block([
            [P.B],
            [zeros(shape=(K.A.shape[0], P.B.shape[1]))],
        ])
        C1 = block([
            [P.C1, zeros(shape=(P.C1.shape[0], K.A.shape[1]))],
        ])
        C2 = block([
            [K.D2 @ P.C2, K.C]
        ])
        D1 = zeros(shape=(C1.shape[0], B1.shape[1]))
        D2 = K.D1

        ret = System(A, B1, B2, C1, C2, D1, D2)
        ret.type = ConnectionType.FB_WITH_INPUT_QUANTIZER
        ret.P = P
        ret.K = K
        return ret

    @staticmethod
    def from_FB_connection_with_output_quantizer(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        Create a System instance from Plant and Controller with output quantizer (feedback connection).

        'from_FB_connection_with_output_quantizer' means that a
        quantizer is inserted as shown in the following figure.

        ..  code-block:: none

            |       ┌───────┐           ┌───────┐       
            | r ───>│       │           │       ├───> z 
            |       │   K   ├──────────>│   P   │       
            |    ┌─>│       │           │       ├──┐    
            |  v │  └───────┘  ┌─────┐  └───────┘  │ u  
            |    └─────────────┤  Q  │<────────────┘    
            |                  └─────┘

        Parameters
        ----------
        P : Plant
            Plant instance. The plant in the feedback loop.
        K : Controller
            Controller instance. The controller in the feedback loop.

        Returns
        -------
        System
            System with quantizer inserted at controller output (feedback).

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> P = nqlib.Plant(0.5, 1, 1, 1)
        >>> K = nqlib.Controller(0, 1, 1, 1, 1, 1)
        >>> sys = nqlib.System.from_FB_connection_with_output_quantizer(P, K)
        >>> sys.is_stable
        False
        """
        if P.m != K.m:
            raise ValueError(
                "The number of columns in matrix `K.C` and the "
                "number of rows in matrix `P.B` must be the same."
            )

        A = block([
            [P.A, P.B @ K.C],
            [zeros(shape=(K.A.shape[0], P.A.shape[1])), K.A],
        ])
        B1 = block([
            [P.B @ K.D1],
            [K.B1],
        ])
        B2 = block([
            [P.B @ K.D2],
            [K.B2],
        ])
        C1 = block([
            P.C1, zeros(shape=(P.C1.shape[0], K.A.shape[1]))
        ])
        C2 = block([
            P.C2, zeros(shape=(P.C2.shape[0], K.A.shape[1]))
        ])
        D1 = zeros(shape=(C1.shape[0], B1.shape[1]))
        D2 = zeros(shape=(B2.shape[1], B1.shape[1]))

        ret = System(
            A,
            B1,
            B2,
            C1,
            C2,
            D1,
            D2,
        )
        ret.type = ConnectionType.FB_WITH_OUTPUT_QUANTIZER
        ret.P = P
        ret.K = K
        return ret

    @staticmethod
    def from_FBIQ(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        Alias for from_FB_connection_with_input_quantizer.

        Parameters
        ----------
        P : Plant
            Plant instance. The plant in the feedback loop.
        K : Controller
            Controller instance. The controller in the feedback loop.

        Returns
        -------
        System
            System with quantizer inserted at controller input (feedback).

        Example
        -------
        >>> import nqlib
        >>> P = nqlib.Plant(0.5, 1, 1, 1)
        >>> K = nqlib.Controller(0, 1, 1, 1, 1, 1)
        >>> sys = nqlib.System.from_FBIQ(P, K)
        >>> sys.is_stable
        False
        """
        return System.from_FB_connection_with_input_quantizer(P, K)

    @staticmethod
    def from_FBOQ(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        Alias for from_FB_connection_with_output_quantizer.

        Parameters
        ----------
        P : Plant
            Plant instance. The plant in the feedback loop.
        K : Controller
            Controller instance. The controller in the feedback loop.

        Returns
        -------
        System
            System with quantizer inserted at controller output (feedback).

        Example
        -------
        >>> import nqlib
        >>> P = nqlib.Plant(0.5, 1, 1, 1)
        >>> K = nqlib.Controller(0, 1, 1, 1, 1, 1)
        >>> sys = nqlib.System.from_FBOQ(P, K)
        >>> sys.is_stable
        False
        """
        return System.from_FB_connection_with_output_quantizer(P, K)

    @property
    def is_stable(self) -> bool:
        """
        Check if the closed-loop system is stable.

        Returns
        -------
        bool
            True if stable, False otherwise.

        Example
        -------
        >>> import nqlib
        >>> sys = nqlib.System(0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        >>> sys.is_stable
        True
        """
        A_tilde = self.A + self.B2 @ self.C2  # convert to closed loop
        if eig_max(A_tilde) > 1:
            return False
        else:
            return True

    def E(self,
          Q: DynamicQuantizer,
          steptime: int | InfInt = infint,
          _check_stability: bool = True,
          verbose: bool = False) -> float:
        """
        Estimate E(Q) for the system and quantizer.

        Parameters
        ----------
        Q : DynamicQuantizer
            Dynamic quantizer instance. The quantizer whose performance is evaluated.
        steptime : int or InfInt, optional
            Evaluation time. Must be a natural number (`steptime` >= 1). Default: `infint`.
        _check_stability : bool, optional
            If True, check stability (default: True).
            This shouldn't be changed.
        verbose : bool, optional
            If True, print progress (default: False).

        Returns
        -------
        float
            Estimated value of E(Q) in the given `steptime`.

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> import numpy as np
        >>> G = nqlib.System(
        ...     A=[[1.15, 0.05],
        ...        [0.00, 0.99]],
        ...     B1=[[1],
        ...         [1]],
        ...     B2=[[0.004],
        ...         [0.099]],
        ...     C1=[1., 0.], C2=[-15., -3.],
        ...     D1=0, D2=0,
        ... )
        >>> q = nqlib.StaticQuantizer.mid_tread(d=2)
        >>> Q, E = nqlib.DynamicQuantizer.design_AG(
        ...     system=G,
        ...     q=q,
        ... )
        >>> np.isclose(E, G.E(Q))
        np.True_
        """
        steptime = validate_int_or_inf(
            steptime,
            minimum=1,
            name="steptime",
        )
        if (not _check_stability) and (steptime is infint):
            raise ValueError(
                "`(not _check_stability and steptime is infint)` must be `False`."
            )
        A_tilde = self.A + self.B2 @ self.C2  # convert to closed loop (only G)

        A_bar = block([
            [A_tilde, self.B2 @ Q.C],
            [zeros((Q.A.shape[0], A_tilde.shape[0])), Q.A + Q.B @ Q.C],
        ])
        B_bar = block([
            [self.B2],
            [Q.B],
        ])
        C_bar = block([
            [self.C1, zeros((self.C1.shape[0], Q.A.shape[0]))]
        ])

        # E = infinity
        if _check_stability:
            Qcl: _ctrl.StateSpace = _ctrl.ss(
                A_bar, B_bar, C_bar, C_bar @ B_bar * 0,  # type: ignore
                True,
            )
            Qcl_minreal = Qcl.minreal()
            if eig_max(Qcl_minreal.A) > 1:  # type: ignore
                return _np.inf

        k = 0
        A_bar_k = eye(*(A_bar.shape))
        sum_CAB = zeros((self.C1.shape[0], self.B2.shape[1]))
        E_current = 0.0
        if steptime is infint:
            k_max = infint

            # smallest k that satisfies
            # `eig_max(A_bar)**k / eig_max(A_bar) < 1e-8`
            eig_max_ = eig_max(A_bar)
            if eig_max_ <= 1e-8:
                k_min = 1
            elif 0.9999 <= eig_max_:  # TODO: なおす
                k_min = 1000
            else:
                k_min = 1 - 8 / _np.log10(eig_max_)
        else:
            k_max = steptime
            k_min = 1
        while k <= k_max:
            E_past = E_current

            sum_CAB = sum_CAB + abs(C_bar @ A_bar_k @ B_bar)
            E_current = norm(sum_CAB) * Q.delta

            if k >= k_min:
                if abs(E_current - E_past) / E_current < 1e-8:
                    break
            if verbose:
                print(f"k={k}, E_past={E_past}, E_current={E_current}, E_current - E_past={E_current - E_past}")
            k = k + 1
            A_bar_k = A_bar_k @ A_bar

        return E_current  # type: ignore

    def is_stable_with_quantizer(self, Q: DynamicQuantizer | StaticQuantizer) -> bool:
        """
        Check if the system is stable with the given quantizer.

        Parameters
        ----------
        Q : DynamicQuantizer or StaticQuantizer
            Quantizer instance. The quantizer to check stability with.

        Returns
        -------
        bool
            True if stable, False otherwise.

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> G = nqlib.System(
        ...     A=[[1.15, 0.05],
        ...        [0.00, 0.99]],
        ...     B1=[[1],
        ...         [1]],
        ...     B2=[[0.004],
        ...         [0.099]],
        ...     C1=[1., 0.], C2=[-15., -3.],
        ...     D1=0, D2=0,
        ... )
        >>> q = nqlib.StaticQuantizer.mid_tread(d=2)
        >>> Q, E = nqlib.DynamicQuantizer.design_AG(
        ...     system=G,
        ...     q=q,
        ... )
        >>> G.is_stable_with_quantizer(Q)
        True
        """
        if isinstance(Q, StaticQuantizer):
            return self.is_stable
        return self.is_stable and Q.is_stable

    def is_stable_with(self, Q: DynamicQuantizer | StaticQuantizer) -> bool:
        """
        Alias for is_stable_with_quantizer.

        Parameters
        ----------
        Q : DynamicQuantizer or StaticQuantizer
            Quantizer instance. The quantizer to check stability with.

        Returns
        -------
        bool
            True if stable, False otherwise.

        References
        ----------
        [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
        quantizers for discrete-valued input control;IEEE Transactions
        on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

        Example
        -------
        >>> import nqlib
        >>> G = nqlib.System(
        ...     A=[[1.15, 0.05],
        ...        [0.00, 0.99]],
        ...     B1=[[1],
        ...         [1]],
        ...     B2=[[0.004],
        ...         [0.099]],
        ...     C1=[1., 0.], C2=[-15., -3.],
        ...     D1=0, D2=0,
        ... )
        >>> q = nqlib.StaticQuantizer.mid_tread(d=2)
        >>> Q, E = nqlib.DynamicQuantizer.design_AG(
        ...     system=G,
        ...     q=q,
        ... )
        >>> G.is_stable_with(Q)
        True
        """
        return self.is_stable_with_quantizer(Q)
