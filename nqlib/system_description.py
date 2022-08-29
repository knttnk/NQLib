import copy
from typing import Tuple, Union

import control as _ctrl
import numpy as _np
from numpy import inf

from .linalg import block, eye, matrix, norm, zeros, eig_max
from .quantizer import DynamicQuantizer as _DynamicQuantizer
from .quantizer import StaticQuantizer as _StaticQuantizer
from .quantizer import _ConnectionType

_ctrl.use_numpy_matrix(False)
__all__ = [
    'System',
    'Controller',
    'Plant',
]
# TODO: support control.input_output_response()
# TODO: define function returns control.InputOutputSystem


class Controller(object):
    """
    An instance of `Controller` represents the state-space model
    of controller K given by

        K : { x(t+1) = A x(t) + B1 r(t) + B2 y(t)
            {  u(t)  = C x(t) + D1 r(t) + D2 y(t)
    """

    def __init__(self, A, B1, B2, C, D1, D2):
        """
        Initializes an instance of `Controller`. You can create an
        instance of `System` with `Controller`[1]_.

        Parameters
        ----------
        A : array_like
            Interpreted as a square matrix.
            `n` is defined as the number of rows in matrix `A`.
        B1 : array_like
            Interpreted as a (`n` x `l`) matrix.
            `l` is defined as the number of columns in matrix `B1`.
        B2 : array_like
            Interpreted as a (`n` x `p`) matrix.
            `p` is defined as the number of columns in matrix `B2`.
        C : array_like
            Interpreted as a (`m` x `n`) matrix.
            `m` is defined as the number of rows in matrix `C`.
        D1 : array_like
            Interpreted as a (`m` x `l`) matrix.
        D2 : array_like
            Interpreted as a (`m` x `p`) matrix.

        Notes
        -----
        An instance of `Controller` represents the state-space model
        of controller K given by

            K : { x(t+1) = A x(t) + B1 r(t) + B2 y(t)
                {  u(t)  = C x(t) + D1 r(t) + D2 y(t)

        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
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


class Plant(object):
    """
    An instance of `Plant` represents the state-space model
    of plant P given by

        P : { x(t+1) =  A x(t) + B u(t)
            {  z(t)  = C1 x(t)
            {  y(t)  = C2 x(t)
    """

    def __init__(self, A, B, C1, C2):
        """
        Initializes an instance of `Plant`. You can create an
        instance of `System` with `Plant`.

        Parameters
        ----------
        A : array_like
            Interpreted as a square matrix.
            `n` is defined as the number of rows in matrix `A`.
        B : array_like
            Interpreted as a (`n` x `m`) matrix.
            `m` is defined as the number of columns in matrix `B`.
        C1 : array_like
            Interpreted as a (`l1` x `n`) matrix.
            `l1` is defined as the number of rows in matrix `C1`.
        C2 : array_like
            Interpreted as a (`l2` x `n`) matrix.
            `l2` is defined as the number of rows in matrix `C2`.

        Notes
        -----
        An instance of `Plant` represents the state-space model
        of plant P given by

            P : { x(t+1) =  A x(t) + B u(t)
                {  z(t)  = C1 x(t)
                {  y(t)  = C2 x(t)
                
        References
        ----------
        .. [5] Y. Minami and T. Muromaki: Differential evolution-based
           synthesis of dynamic quantizers with fixed-structures; International
           Journal of Computational Intelligence and Applications, Vol. 15,
           No. 2, 1650008 (2016)
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
        self.tf1 = _ctrl.ss2tf(ss1)
        self.tf2 = _ctrl.ss2tf(ss2)

    @staticmethod
    def from_TF(tf: _ctrl.TransferFunction) -> "Plant":
        """
        Creates an instance of `Plant` from transfer function. Note
        that `C2` becomes 0 .

        Parameters
        ----------
        tf : control.TransferFunction
            Transfer function from input u to output z.

        Returns
        -------
        Plant

        Notes
        -----
        An instance of `Plant` represents a plant
        P given by

            P : { x(t+1) =  A x(t) + B u(t)
                {  z(t)  = C1 x(t)
                {  y(t)  = C2 x(t)

        But if you use this method, `C2` becomes `0`.
        """
        ss = _ctrl.tf2ss(tf)
        ret = Plant(
            A=matrix(ss.A),
            B=matrix(ss.B),
            C1=matrix(ss.C),
            C2=matrix(zeros(ss.C.shape)),
        )
        ret.tf1 = tf
        ret.tf2 = tf*0
        return ret


class System():
    """
    Represents ideal system. 'Ideal' means that this system doesn't
    contain a quantizer.
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

    def response_with_quantizer(self,
                                quantizer: Union[_DynamicQuantizer, _StaticQuantizer],
                                input,
                                x_0) -> Tuple[_np.ndarray]:
        """
        Performs a simulation with `quantizer` and returns results.

        Parameters
        ----------
        quantizer : Union[DynamicQuantizer, StaticQuantizer]
        input : array_like
        x_0 : array_like

        Returns
        -------
        (t, u, v, z): Tuple[np.ndarray]
            Time, input, quantized input, and output.

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        if type(quantizer) is _StaticQuantizer:
            quantizer = _DynamicQuantizer(
                zeros((1, 1)),  zeros((1, self.m)),
                zeros((self.m, 1)),
                q=quantizer,
            )
        # TODO: support xP_0, xK_0
        # TODO: support time
        r = matrix(input)
        length = r.shape[1]
        k = matrix(range(0, length))

        z = zeros((self.l, length))
        u = zeros((self.m, length))
        v = copy.deepcopy(u)
        x = zeros((len(x_0), length))
        xi = zeros((len(quantizer.A), length))
        x[:, 0:1] = matrix(x_0)

        for i in range(length):
            u[:, i:i+1] = matrix(self.C2 @ x[:, i:i+1] + self.D2 @ r[:, i:i+1])
            v[:, i:i+1] = matrix(quantizer.q(quantizer.C @ xi[:, i:i+1] + u[:, i:i+1]))
            z[:, i:i+1] = matrix(self.C1 @ x[:, i:i+1] + self.D1 @ r[:, i:i+1])
            if i < length - 1:
                xi[:, i+1:i+2] = matrix(quantizer.A @ xi[:, i:i+1] + quantizer.B @ (v[:, i:i+1] - u[:, i:i+1]))
                x[:, i+1:i+2] = matrix(self.A @ x[:, i:i+1] + self.B1 @ r[:, i:i+1] + self.B2 @ v[:, i:i+1])
        return k, u, v, z

    def response(self, input, x_0) -> Tuple[_np.ndarray]:
        """
        Performs a simulation and returns results.

        Parameters
        ----------
        input : array_like
        x_0 : array_like

        Returns
        -------
        (t, u, z): Tuple[np.ndarray]
            Time, input, and output.

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        # TODO: support xP_0, xK_0
        # TODO: support time
        r = matrix(input)
        length = r.shape[1]
        k = matrix(range(0, length))

        z = zeros((self.l, length))
        u = zeros((self.m, length))
        x = zeros((len(x_0), length))
        x[:, 0:1] = matrix(x_0)

        for i in range(length):
            u[:, i:i+1] = matrix(self.C2 @ x[:, i:i+1] + self.D2 @ r[:, i:i+1])
            z[:, i:i+1] = matrix(self.C1 @ x[:, i:i+1] + self.D1 @ r[:, i:i+1])
            if i < length - 1:
                x[:, i+1:i+2] = matrix(
                    self.A @ x[:, i:i+1] + self.B1 @ r[:, i:i+1] + self.B2 @ u[:, i:i+1]
                )
        return k, u, z

    def __init__(self, A, B1, B2, C1, C2, D1, D2):
        """
        Initializes an instance of `System`. 'Ideal' means that
        this system doesn't contain a quantizer.

        Parameters
        ----------
        A : array_like
            Interpreted as a square matrix.
            `n` is defined as the number of rows in matrix `A`.
        B1 : array_like
            Interpreted as a (`n` x `p`) matrix.
            `p` is defined as the number of columns in matrix `B1`.
        B2 : array_like
            Interpreted as a (`n` x `m`) matrix.
            `m` is defined as the number of columns in matrix `B2`.
        C1 : array_like
            Interpreted as a (`l` x `n`) matrix.
            `l` is defined as the number of rows in matrix `C1`.
        C2 : array_like
            Interpreted as a (`m` x `n`) matrix.
        D1 : array_like
            Interpreted as a (`l` x `p`) matrix.
        D2 : array_like
            Interpreted as a (`m` x `p`) matrix.

        Notes
        -----
        An instance of `System` represents ideal system
        G given by

            G : { x(t+1) =  A x(t) + B1 r(t) + B2 v(t)
                {  z(t)  = C1 x(t) + D1 r(t)
                {  u(t)  = C2 x(t) + D2 r(t)
                
        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
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
        self.type = _ConnectionType.ELSE
        self.P = None
        self.K = None

    @staticmethod
    def from_FF(P: Plant) -> "System":
        """
        Creates an instance of `System` from plant `P`.
        'from_FF' means that a quantizer is inserted as shown in the
        following figure[1]_.

        ```text
               +-----+  v  +-----+
         u --->|  Q  |---->|  P  |---> z
               +-----+     +-----+
        ```

        Parameters
        ----------
        P : Plant

        Returns
        -------
        System

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
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
        ret.type = _ConnectionType.FF
        ret.P = P
        return ret

    @staticmethod
    def from_FB_connection_with_input_quantizer(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        Creates an instance of `System` from plant `P` and
        controller `K`.
        'from_FB_connection_with_input_quantizer' means that a
        quantizer is inserted as shown in the following figure[1]_.

        ```text
               +-------+     +-------+     +-------+       
         r --->|       |  u  |       |  v  |       |---> z 
               |   K   |---->|   Q   |---->|   P   |       
            +->|       |     |       |     |       |--+    
            |  +-------+     +-------+     +-------+  | y  
            +-----------------------------------------+    
        ```

        Parameters
        ----------
        P : Plant
        K : Controller

        Returns
        -------
        System

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        if P.l2 != K.p:
            raise ValueError(
                "The number of columns in matrix `P.C2` and the "
                "number of rows in matrix `K.B2` must be the same."
            )

        A = block([
            [P.A,         zeros(shape=(P.A.shape[0], K.A.shape[1]))],
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
            [P.C1,  zeros(shape=(P.C1.shape[0], K.A.shape[1]))],
        ])
        C2 = block([
            [K.D2 @ P.C2,  K.C]
        ])
        D1 = zeros(shape=(C1.shape[0], B1.shape[1]))
        D2 = K.D1

        ret = System(A, B1, B2, C1, C2, D1, D2)
        ret.type = _ConnectionType.FB_WITH_INPUT_QUANTIZER
        ret.P = P
        ret.K = K
        return ret

    @staticmethod
    def from_FB_connection_with_output_quantizer(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        Creates an instance of `System` from plant `P` and
        controller `K`.
        'from_FB_connection_with_output_quantizer' means that a
        quantizer is inserted as shown in the following figure[1]_.

        ```text
               +-------+           +-------+       
         r --->|       |           |       |---> z 
               |   K   |---------->|   P   |       
            +->|       |           |       |--+    
          v |  +-------+  +-----+  +-------+  | u  
            +-------------|  Q  |<------------+    
                          +-----+                  
        ```

        Parameters
        ----------
        P : Plant
        K : Controller

        Returns
        -------
        System

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        if P.m != K.m:
            raise ValueError(
                "The number of columns in matrix `K.C` and the "
                "number of rows in matrix `P.B` must be the same."
            )

        A = block([
            [P.A,                                        P.B @ K.C],
            [zeros(shape=(K.A.shape[0], P.A.shape[1])),  K.A],
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
            P.C1,  zeros(shape=(P.C1.shape[0], K.A.shape[1]))
        ])
        C2 = block([
            P.C2,  zeros(shape=(P.C2.shape[0], K.A.shape[1]))
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
        ret.type = _ConnectionType.FB_WITH_OUTPUT_QUANTIZER
        ret.P = P
        ret.K = K
        return ret

    @staticmethod
    def from_FBIQ(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        A shortened form of `from_FB_connection_with_input_quantizer`.

        Creates an instance of `System` from plant `P` and
        controller `K`.
        'from_FB_connection_with_input_quantizer' means that a
        quantizer is inserted as shown in the following figure[1]_.

        ```text
               +-------+     +-------+     +-------+       
         r --->|       |  u  |       |  v  |       |---> z 
               |   K   |---->|   Q   |---->|   P   |       
            +->|       |     |       |     |       |--+    
            |  +-------+     +-------+     +-------+  | y  
            +-----------------------------------------+    
        ```

        Parameters
        ----------
        P : Plant
        K : Controller

        Returns
        -------
        System

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        return System.from_FB_connection_with_input_quantizer(P, K)

    @staticmethod
    def from_FBOQ(
        P: Plant,
        K: Controller
    ) -> "System":
        """
        A shortened form of `from_FB_connection_with_output_quantizer`.

        Creates an instance of `System` from plant `P` and
        controller `K`.
        'from_FB_connection_with_output_quantizer' means that a
        quantizer is inserted as shown in the following figure[1]_.

        ```text
               +-------+           +-------+       
         r --->|       |           |       |---> z 
               |   K   |---------->|   P   |       
            +->|       |           |       |--+    
          v |  +-------+  +-----+  +-------+  | u  
            +-------------|  Q  |<------------+    
                          +-----+                  
        ```

        Parameters
        ----------
        P : Plant
        K : Controller

        Returns
        -------
        System

        References
        ----------
        .. [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic
           quantizers for discrete-valued input control;IEEE Transactions
           on Automatic Control, Vol. 53,pp. 2064–2075 (2008)
        """
        return System.from_FB_connection_with_output_quantizer(P, K)

    @property
    def is_stable(self) -> bool:
        """
        Returns stability of this system.

        Returns
        -------
        bool
            `True` if stable, `False` if not.
        """
        A_tilde = self.A + self.B2 @ self.C2  # convert to closed loop
        if eig_max(A_tilde) > 1:
            return False
        else:
            return True

    def E(self,
          Q: _DynamicQuantizer,
          steptime: Union[int, None] = None,
          _check_stability: bool = True) -> float:
        """
        Returns estimation of E(Q).

        Parameters
        ----------
        Q : DynamicQuantizer
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

        if (steptime is not None) and steptime <= 0:
            raise ValueError("steptime must be a natural number.")
        if not _check_stability and steptime is None:
            raise ValueError(
                "`(steptime is not None or _check_stability)` must be `True`."
            )
        A_tilde = self.A + self.B2@self.C2  # convert to closed loop (only G)

        A_bar = block([
            [A_tilde,             self.B2@Q.C],
            [zeros((Q.A.shape[0], A_tilde.shape[0])), Q.A+Q.B@Q.C],
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
            Qcl = _ctrl.ss(A_bar, B_bar, C_bar, C_bar@B_bar*0, True)
            Qcl_minreal = Qcl.minreal()
            if eig_max(Qcl_minreal.A) > 1:
                return inf

        k = 0
        A_bar_k = eye(*(A_bar.shape))
        sum_CAB = zeros((self.C1.shape[0], self.B2.shape[1]))
        E_current = 0
        if steptime is None:
            k_max = inf

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
                if abs(E_current-E_past)/E_current < 1e-8:
                    break
            k = k + 1
            A_bar_k = A_bar_k @ A_bar

        return E_current

    def is_stable_with_quantizer(self, Q) -> bool:
        """
        Returns stability of this system with quantizer `Q`[1]_.

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
        return self.is_stable and Q.is_stable

    def is_stable_with(self, Q) -> bool:
        """
        A shortened form of `is_stable_with_quantizer`.

        Returns stability of this system with quantizer `Q`[1]_.

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
        return self.is_stable_with_quantizer(Q)
