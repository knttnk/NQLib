"""
NQLib
=====

NQLib is a Python library to design noise shaping quantizer for
discrete-valued input control.

## What can I do with NQLib?

In the real world, a dynamic system may have to be controlled by
discrete-valued signals due to the inclusion of actuators that are driven
by ON/OFF or network with capacity limitations. In such a case, a good output
may be obtained by converting continuous-valued input to discrete-valued
input with a quantizer designed by NQLib.

## Documentation

All the documentation is available at https://knttnk.github.io/NQLib/.

Examples of usage are available at https://colab.research.google.com/drive/1Ui-XqaTZCjwqRXC3ZeMeMCbPqGxK9YXO.

## References

NQLib is a Python library version of ODQ Toolbox, 
which were developed in MATLAB.

- https://github.com/rmorita-jp/odqtoolbox
- https://www.mathworks.com/products/matlab.html

The algorithms used in NQLib are based on the following paper.

- [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic quantizers for
  discrete-valued input control;IEEE Transactions on Automatic Control, Vol. 53,
  pp. 2064–2075 (2008)
- [2] S. Azuma, Y. Minami and T. Sugie: Optimal dynamic quantizers for feedback
  control with discrete-level actuators; Journal of Dynamic Systems, Measurement,
  and Control, Vol. 133, No. 2, 021005 (2011)
- [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会論文集，Vol. 52, pp.
  46–51(2016)
- [4] R. Morita, S. Azuma, Y. Minami and T. Sugie: Graphical design software for
  dynamic quantizers in control systems; SICE Journal of Control, Measurement, and
  System Integration, Vol. 4, No. 5, pp. 372-379 (2011)
- [5] Y. Minami and T. Muromaki: Differential evolution-based synthesis of dynamic
  quantizers with fixed-structures; International Journal of Computational
  Intelligence and Applications, Vol. 15, No. 2, 1650008 (2016)

## License

This software is released under the MIT License, see LICENSE.txt.
"""
from __future__ import annotations

from .types import infint
from .system_description import Controller, Plant, System
from .quantizer import StaticQuantizer, DynamicQuantizer, order_reduced
from cvxpy import installed_solvers  # TODO: describe in documentation

__all__ = [
    # Types
    'infint',
    # System Description
    'Controller',
    'Plant',
    'System',
    # Quantizer
    'StaticQuantizer',
    'DynamicQuantizer',
    'order_reduced',
    # solvers
    'installed_solvers',
]

from packaging.version import Version
__version__ = Version("1.0.0")
