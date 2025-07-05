import sys
sys.path.append(r"H:\マイドライブ\IshikawaMinamiLab\研究\NQLib")
import nqlib_dev

G = nqlib_dev.IdealSystem(
    A=[[1.15, 0.05],
        [0.00, 0.99]],
    B1=[[0.],
        [0.]],
    B2=[[0.004],
        [0.099]],
    C1=[1., 0.],
    C2=[-15., -3.],
    D1=0,
    D2=1,
)

q = nqlib_dev.StaticQuantizer.mid_tread(d=2)

print("-original------------------------------------------------------------------------")
Q, E = nqlib_dev.DynamicQuantizer.LP(G,
                                     q=q,
                                     T=100,
                                     gain_wv=2)
print(f"G.E(Q) = {G.E(Q)}")
print(f"Q.gain_wv() = {Q.gain_wv()}")

print("-odq------------------------------------------------------------------------")
Q_reduced, E = nqlib_dev.DynamicQuantizer.LP(G,
                                             q=q,
                                             T=100,
                                             gain_wv=2,
                                             dim=5)
print(f"G.E(Q) = {G.E(Q_reduced)}")
print(f"Q.gain_wv() = {Q_reduced.gain_wv()}")

print("-nqlib------------------------------------------------------------------------")
Q_reduced = Q.dimension_reduced(5)
print(f"G.E(Q) = {G.E(Q_reduced)}")
print(f"Q.gain_wv() = {Q_reduced.gain_wv()}")
