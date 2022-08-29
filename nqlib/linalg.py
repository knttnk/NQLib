import numpy as np
import scipy.sparse.linalg

np.set_printoptions(precision=5, suppress=True)

array = np.array


def matrix(M) -> np.ndarray:
    return np.array(M, ndmin=2)


def kron(A, B) -> np.ndarray:
    return matrix(np.kron(A, B))


def block(M) -> np.ndarray:
    return matrix(np.block(M))


def eye(N, M=None) -> np.ndarray:
    return matrix(np.eye(N, M))


def norm(A: np.ndarray) -> float:
    return np.linalg.norm(A, ord=np.inf)


def zeros(shape) -> np.ndarray:
    return matrix(np.zeros(shape))


def ones(shape) -> np.ndarray:
    return matrix(np.ones(shape))


def pinv(a: np.ndarray) -> np.ndarray:
    return matrix(np.linalg.pinv(a))


def eig_max(A) -> float:
    return max(abs(np.linalg.eig(A)[0]))


def mpow(A: np.ndarray, x) -> np.ndarray:
    return matrix(scipy.linalg.fractional_matrix_power(A, x))
