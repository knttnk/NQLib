
import sys
sys.path.append(r"H:\マイドライブ\IshikawaMinamiLab\研究\NQLib")
import nqlib as nq
import numpy as np

ideal_system = nq.IdealSystem(A=[[ 1.8, 0.8],
                                 [-1.0, 0.0]],
                              B1=[[0],
                                  [0]],
                              B2=[[1],
                                  [0]],
                              C1=[0.10, 0.09],
                              C2=[0, 0],
                              D1=0,
                              D2=1)

q = nq.StaticQuantizer.mid_tread(d=2)
t = np.arange(0, 101)
r = 0.9 * np.cos(0.11*np.pi*t) + np.sin(0.02*np.pi*t)
x_0 = [[0],[0]]


#@title ## **with no constraints using DE** { vertical-output: true }
Q, E = nq.DynamicQuantizer.DE(ideal_system, 
                                          q=q, 
                                          dim=2, 
                                          verbose=True)
print(Q.A)
print(Q.B)
print(Q.C)
_, u, z_i = ideal_system.response(r, x_0)
_, _, v, z_Q = ideal_system.response_with_quantizer(Q,
                                                    r,
                                                    x_0)
