# Changelog for NQLib

## 1.0.1

- `IdealSystem.E()` now supports `StaticQuantizer`.
- `StaticQuantizer` now has the method `cost()` to compute its cost, similar to `DynamicQuantizer.cost()`.

## 1.0.0

### Breaking changes

- NQLib now requires Python 3.11 or later.
- `design_AG()` now raises an error if `dim` or `gain_wv` are passed.
  - Users have to check specifications of the quantizer and set `dim` and `gain_wv` manually.
- The following arguments have been renamed based on their roles:
  - `gain_wv` → `max_gain_wv`
  - `T` → `steptime`, `steptime_gain_wv`, `steptime_E`
  - `dim` → `N`, `max_N`, `new_N`
  - Regarding `design_LP()`, roles of its arguments are complicated, so the names are not changed.

### New features

- Users can now optimize an SISO dynamic quantizer with their own method.
  - For an SISO dynamic quantizer `Q` with its order `N`, `Q.to_parameters()` now returns an array of parameters of length `2*N`.
  - `DynamicQuantizer.from_SISO_parameters()` now creates an `N`th-order dynamic quantizer from an array of `2*N` parameters.
  - `Q.objective_function()` now returns the objective function of the quantizer. See details in the documentation and [tests (test_user_optimization)](tests/test_nqlib.py).

### Bug fixes

- Fixed a bug that NQLib may fail to import when `control` does not have `use_numpy_matrix`.
- Fixed a bug that `design_AG()` when the system's zero is not real.

### Others

- Modified the directory structure.
- Added `verbose` argument to `gain_wv()` and `E()`.
- `design_GD()` randomly initiates the parameters before optimization.
- Improved type hints for major functions and classes.
- Added the class `InfInt` and its singleton `infint` to represent infinite integers.
- Improved documentation. All major functions and classes now have docstrings including examples.
- Automatically test NQLib with `pytest` and `doctest`.

## 0.5.1

- Fixed a bug that `design_LP()` may fail.
- Added `allow_unstable` to `design_AG()`.
- Added `__repr__` to `Plant`, `Controller`, `System`. So `repr()` now returns a more readable string.

## 0.5.0

- Added MIMO system support for `design_LP()`.

## 0.4.0

- Added `DynamicQuantizer.spec()` which prints the specs of the quantizer.
- Quantizers are now represented in an easy-to-understand way. The latex expressions are also included.
