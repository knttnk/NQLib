import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike

from .types import NDArrayNum, Real

np.set_printoptions(precision=5, suppress=True)


def matrix(M: ArrayLike) -> NDArrayNum:
    ret = np.array(M, ndmin=2)
    if not np.issubdtype(ret.dtype, np.number):
        raise TypeError(
            "This matrix must be a numeric type (e.g., float, int, complex).\n"
            f"Received type: {ret.dtype}.\n"
            f"Received value: {M}."
        )
    return ret


def kron(A: NDArrayNum, B: NDArrayNum) -> NDArrayNum:
    return matrix(np.kron(A, B))


def block(M: ArrayLike) -> NDArrayNum:
    return matrix(np.block(M))


def eye(N: int, M: int | None = None, k: int = 0) -> NDArrayNum:
    return matrix(np.eye(N, M, k=k))


def norm(A: NDArrayNum) -> np.floating | np.integer:
    ret = np.linalg.norm(A, ord=np.inf)
    return ret


def zeros(shape: int | tuple[int, ...]) -> NDArrayNum:
    return matrix(np.zeros(shape))


def ones(shape: int | tuple[int, ...]) -> NDArrayNum:
    return matrix(np.ones(shape))


def pinv(a: NDArrayNum) -> NDArrayNum:
    return matrix(np.linalg.pinv(a))


def eig_max(A: NDArrayNum) -> Real:
    return max(abs(np.linalg.eig(A)[0]))


def mpow(A: NDArrayNum, x: Real | int | float) -> NDArrayNum:
    return matrix(scipy.linalg.fractional_matrix_power(A, x))
