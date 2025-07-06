from enum import Enum as _Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
Real = np.floating | np.integer
NDArrayNum = NDArray[Real]

__all__ = ["infint"]


class InfInt(_Enum):
    """
    Represents infinite integer.
    This class is used to represent infinity in a type-safe way.
    It is used in the `dim`, `T` and so on.
    """
    Value = np.inf

    def __lt__(self, other: Any) -> bool:
        """
        Compare if InfInt is less than another value.

        Parameters
        ----------
        other : Any
            Value to compare.

        Returns
        -------
        bool
        """
        return np.inf < other

    def __le__(self, other: Any) -> bool:
        """
        Compare if InfInt is less than or equal to another value.

        Parameters
        ----------
        other : Any
            Value to compare.

        Returns
        -------
        bool
        """
        return np.inf <= other

    def __gt__(self, other: Any) -> bool:
        """
        Compare if InfInt is greater than another value.

        Parameters
        ----------
        other : Any
            Value to compare.

        Returns
        -------
        bool
        """
        return np.inf > other

    def __ge__(self, other: Any) -> bool:
        """
        Compare if InfInt is greater than or equal to another value.

        Parameters
        ----------
        other : Any
            Value to compare.

        Returns
        -------
        bool
        """
        return np.inf >= other

    def __float__(self) -> float:
        """
        Convert InfInt to float.

        Returns
        -------
        float
        """
        return float(np.inf)


infint = InfInt.Value


def validate_int_or_inf(
    val: Any,
    *,
    minimum: int | None = None,
    name: str = "",
) -> int | InfInt:
    """
    Validates if the given value is an integer or InfInt.

    Parameters
    ----------
    val : Any
        Value to validate.
    minimum : int or None, optional
        Minimum allowed value (inclusive). Default is None.
    name : str, optional
        Name of the variable for error messages. Default is ''.

    Returns
    -------
    int or InfInt
        Validated integer or InfInt value.

    Notes
    -----
    After validation, `minimum` <= `val` is satisfied. InfInt is allowed.
    """
    # check type
    if isinstance(val, InfInt):
        return val
    return validate_int(val, minimum=minimum, name=name)


def validate_int(
    val: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    name: str = "",
) -> int:
    """
    Validates if the given value is an integer or InfInt.

    Parameters
    ----------
    val : Any
        Value to validate.
    minimum : int or None, optional
        Minimum allowed value (inclusive). Default is None.
    maximum : int or None, optional
        Maximum allowed value (inclusive). Default is None.
    name : str, optional
        Name of the variable for error messages. Default is ''.

    Returns
    -------
    int
        Validated integer value.

    Notes
    -----
    After validation, `minimum` <= `val` <= `maximum` is satisfied.
    """
    if not float(val).is_integer():
        raise TypeError(
            f"{name} must be an integer or InfInt, but got {val}."
        )
    val_int = int(val)
    # check range
    if minimum is not None and val_int < minimum:
        raise ValueError(
            f"{name} must be greater than or equal to {minimum}, but got {val_int}."
        )
    if maximum is not None and val_int > maximum:
        raise ValueError(
            f"{name} must be less than or equal to {maximum}, but got {val_int}."
        )
    return val_int


def validate_float(
    val: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    infimum: float | None = None,
    supremum: float | None = None,
    name: str = "",
) -> float:
    """
    Validates if the given value is a float.

    Parameters
    ----------
    val : Any
        Value to validate.
    minimum : float or None, optional
        Minimum allowed value (inclusive). Default is None.
    maximum : float or None, optional
        Maximum allowed value (inclusive). Default is None.
    infimum : float or None, optional
        Lower bound (exclusive). Default is None.
    supremum : float or None, optional
        Upper bound (exclusive). Default is None.
    name : str, optional
        Name of the variable for error messages. Default is ''.

    Returns
    -------
    float
        Validated float value.

    Raises
    ------
    TypeError
        If the value is not a float.
    ValueError
        If the value is out of the specified range.

    Notes
    -----
    After validation,
    `minimum` <= `val`,
    `infimum` < `val`,
    `val` < `supremum`,
    `val` <= `maximum`
    are satisfied.
    """
    if not isinstance(val, (float, int, Real)):
        raise TypeError(f"{name} must be a float, but got {val} of type {type(val)}.")
    if minimum is not None and infimum is not None:
        raise ValueError(
            f"Cannot specify both `minimum` ({minimum}) and `infimum` ({infimum}) for {name}."
        )
    if maximum is not None and supremum is not None:
        raise ValueError(
            f"Cannot specify both `maximum` ({maximum}) and `supremum` ({supremum}) for {name}."
        )
    # check range
    if minimum is not None and val < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}, but got {val}.")
    if infimum is not None and val <= infimum:
        raise ValueError(f"{name} must be greater than {infimum}, but got {val}.")
    if supremum is not None and val >= supremum:
        raise ValueError(f"{name} must be less than {supremum}, but got {val}.")
    if maximum is not None and val > maximum:
        raise ValueError(f"{name} must be less than or equal to {maximum}, but got {val}.")
    return val
