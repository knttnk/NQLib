
# 0.5.1

- Fixed a bug that `design_LP()` may fail.
- Added `allow_unstable` to `design_AG()`.
- Added `__repr__` to `Plant`, `Controller`, `System`. So `repr()` now returns a more readable string.

# 0.5.0

- Added MIMO system support for `design_LP()`.

# 0.4.0

- Added `DynamicQuantizer.spec()` which prints the specs of the quantizer.
- Quantizers are now represented in an easy-to-understand way. The latex expressions are also included.
