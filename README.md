# **Welcome to NQLib**
NQLib is a 
<a href="https://www.python.org/" target="_blank">Python</a>
library to design noise shaping quantizer for discrete-valued input control.

# **What can I do with NQLib?**
In the real world, a dynamic system may have to be controlled by discrete-valued signals due to the inclusion of actuators that are driven by ON/OFF or network with capacity limitations. In such a case, a good output may be obtained by converting continuous-valued input to discrete-valued input with a quantizer designed by NQLib.

## **Example**
See
<a href="https://colab.research.google.com/drive/1Ui-XqaTZCjwqRXC3ZeMeMCbPqGxK9YXO" target="_blank">here</a>.

# **Install**
You can install NQLib by using `pip`.
```
pip install nqlib
```
# **References**
NQLib is a Python library version of
<a href="https://github.com/rmorita-jp/odqtoolbox" target="_blank">ODQ Toolbox</a>,
which were developed in
<a href="https://www.mathworks.com/products/matlab.html" target="_blank">MATLAB</a>.

The algorithms used in NQLib are based on the following paper.

- [1] S. Azuma and T. Sugie: Synthesis of optimal dynamic quantizers for discrete-valued input control;IEEE Transactions on Automatic Control, Vol. 53,pp. 2064–2075 (2008)

- [2] S. Azuma, Y. Minami and T. Sugie: Optimal dynamic quantizers for feedback control with discrete-level actuators; Journal of Dynamic Systems, Measurement, and Control, Vol. 133, No. 2, 021005 (2011)

- [3] 南，加嶋：システムの直列分解に基づく動的量子化器設計；計測自動制御学会論文集，Vol. 52, pp. 46–51(2016)

- [4] R. Morita, S. Azuma, Y. Minami and T. Sugie: Graphical design software for dynamic quantizers in control systems; SICE Journal of Control, Measurement, and System Integration, Vol. 4, No. 5, pp. 372-379 (2011)

- [5] Y. Minami and T. Muromaki: Differential evolution-based synthesis of dynamic quantizers with fixed-structures; International Journal of Computational Intelligence and Applications, Vol. 15, No. 2, 1650008 (2016)


# License
This software is released under the MIT License, see LICENSE.txt.