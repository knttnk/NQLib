import unittest
import control
import numpy
import random
rand = random.random


class NQLibTest(unittest.TestCase):
    def test_cost(self):
        import nqlib
        G = nqlib.System(
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
        q = nqlib.StaticQuantizer.mid_tread(d=2)
        Q, E = nqlib.DynamicQuantizer.design(G,
                                             q=q,
                                             verbose=True)
        E_E = G.E(Q)
        E_cost = Q.cost(G)
        self.assertAlmostEqual(E, E_E)
        self.assertAlmostEqual(E, E_cost)

    def test_response(self):
        import nqlib
        G = nqlib.System(
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
        q = nqlib.StaticQuantizer.mid_tread(d=2)
        Q, E = nqlib.DynamicQuantizer.design(G,
                                             q=q,
                                             verbose=True)
        t = numpy.arange(0, 101, 1)
        r = 0.3 * numpy.sin(0.1 * numpy.pi * t) - 0.5 * numpy.cos(0.03 * numpy.pi * t)
        x_0 = [[0.1],
               [0.0]]

        t, u_i, z_i = G.response(r, x_0)
        t, u_Q, v_Q, z_Q = G.response_with_quantizer(Q, r, x_0)
        self.assertGreaterEqual(E, numpy.max(abs(z_Q - z_i)))  # E >= max

    def test_order_reduction(self):
        import nqlib

        def randn(shape): return 2 * numpy.random.randn(*shape)
        A = numpy.array([[1.1, 0.5],
                         [0.2, 0.9]])
        B2 = numpy.array([[0.04],
                          [0.09]])
        C2 = numpy.array([-14., -3.], ndmin=2)
        D_shape = (1, 1)

        ideal_system = nqlib.System(
            A=A,
            B1=randn(B2.shape),
            B2=B2,
            C1=randn(C2.shape),
            C2=C2,
            D1=randn(D_shape),
            D2=randn(D_shape),
        )

        q = nqlib.StaticQuantizer.mid_tread(d=2)
        Q_5, E_5 = nqlib.DynamicQuantizer.design_GD(
            ideal_system,
            q=q,
            dim=5,
            verbose=True,
        )
        # BUG: GDで勾配が0になるのか最適化がされず係数が全て0になることがある
        Q_1 = Q_5.order_reduced(1)
        Q_1D = nqlib.order_reduced(Q_5, 1)

        self.assertEqual(
            Q_1.A, Q_1D.A,
            msg=(
                f"Q_1 and Q_1D is not equal.\n"
            ),
        )
        self.assertEqual(
            Q_5.N, 5,
            msg=(
                f"original order is {Q_5.N}, not 5\n"
                f"system:\n"
                f"{ideal_system}"
            ),
        )
        self.assertEqual(
            Q_1.N, 1,
            msg=(
                f"reduced order is {Q_1.N}, not 2\n"
                f"system:\n"
                f"{ideal_system}"
            ),
        )

        E_2 = ideal_system.E(Q_1)
        self.assertGreater(
            E_2, E_5,
            msg=(
                f"reduced quantizer has E_2 = {E_2}, better than E_5 = {E_5}\n"
                f"system:\n"
                f"{ideal_system}"
            ),
        )

    def test_mid_tread(self):
        import nqlib
        d = abs(rand() * 10 + 0.1)

        # validate construction
        with self.assertRaises(TypeError):
            q = nqlib.StaticQuantizer.mid_tread(
                d="12"  # type: ignore
            )
        with self.assertRaises(TypeError):
            q = nqlib.StaticQuantizer.mid_tread(
                d=d,
                bit=3.1,  # type: ignore
            )
        with self.assertRaises(ValueError):
            q = nqlib.StaticQuantizer.mid_tread(
                d=-d  # type: ignore
            )
        q = nqlib.StaticQuantizer.mid_tread(
            numpy.float64(d),
            3,
            error_on_excess=True,
        )

        self.assertEqual(q(0), 0)
        self.assertEqual(q(d / 4), 0)
        self.assertEqual(q(d + d / 4), d)

        with self.assertRaises(ValueError):
            q(-4 * d)
        with self.assertRaises(ValueError):
            q(5 * d)

        q = nqlib.StaticQuantizer.mid_tread(d, 3, error_on_excess=False)
        self.assertEqual(q(-4 * d), -3 * d)
        self.assertEqual(q(5 * d), 4 * d)

        q = nqlib.StaticQuantizer.mid_tread(d)
        self.assertEqual(q(d + d / 4), d)

    def test_mid_riser(self):
        import nqlib
        d = rand() * 10 + 0.1
        with self.assertRaises(TypeError):
            q = nqlib.StaticQuantizer.mid_riser(
                d="1"  # type: ignore
            )
        with self.assertRaises(TypeError):
            q = nqlib.StaticQuantizer.mid_riser(
                d=d,
                bit=3.1,  # type: ignore
            )
        with self.assertRaises(ValueError):
            q = nqlib.StaticQuantizer.mid_riser(
                d=-d  # type: ignore
            )
        q = nqlib.StaticQuantizer.mid_riser(d, 3, error_on_excess=True)

        self.assertEqual(q(0.1), d / 2)
        self.assertEqual(q(-3 * d / 4), -d / 2)
        self.assertEqual(q(2.5 * d), 2.5 * d)

        with self.assertRaises(ValueError):
            q(-4.01 * d)
        with self.assertRaises(ValueError):
            q(4.01 * d)

        q = nqlib.StaticQuantizer.mid_riser(d, 3, error_on_excess=False)
        self.assertEqual(q(-6 * d), -3.5 * d)
        self.assertEqual(q(5 * d), 3.5 * d)

        q = nqlib.StaticQuantizer.mid_riser(d)
        self.assertEqual(q(2.5 * d), 2.5 * d)

    def test_static_quantizer(self):
        import nqlib
        d = rand() * 10 + 0.1
        with self.assertRaises(TypeError):
            q = nqlib.StaticQuantizer(
                lambda x: x,
                delta="r",
            )
        with self.assertRaises(ValueError):
            q = nqlib.StaticQuantizer(
                lambda x: x,
                delta=-d,
            )
        with self.assertRaises(TypeError):
            q = nqlib.StaticQuantizer(
                "lambda x: x",
                delta=d,
            )
        q = nqlib.StaticQuantizer(
            lambda x: 0,
            delta=0.1,
            error_on_excess=True,
        )
        with self.assertRaises(ValueError):
            q(-10)
        q = nqlib.StaticQuantizer(
            lambda x: 0,
            delta=0.1,
            error_on_excess=False,
        )
        q(10)

    def test_dynamic_quantizer(self):
        import nqlib
        q = nqlib.StaticQuantizer.mid_tread(1)
        with self.assertRaises(TypeError):
            Q = nqlib.DynamicQuantizer(
                "A", "B", "C", q
            )

    def test_system(self):
        import nqlib
        A = numpy.array([[1.15, 0.05],
                         [0.00, 0.99]])
        B2 = numpy.array([[0.004],
                          [0.099]])
        C2 = numpy.array([-15., -3.], ndmin=2)
        D_shape = (1, 1)

        ideal_system = nqlib.System(
            A=A,
            B1=numpy.random.randn(*B2.shape),
            B2=B2,
            C1=numpy.random.randn(*C2.shape),
            C2=C2,
            D1=numpy.random.randn(*D_shape),
            D2=numpy.random.randn(*D_shape),
        )

        with self.assertRaises(TypeError):
            nqlib.System(
                A="A",
                B1=numpy.random.randn(*B2.shape),
                B2=B2,
                C1=numpy.random.randn(*C2.shape),
                C2=C2,
                D1=numpy.random.randn(*D_shape),
                D2=numpy.random.randn(*D_shape),
            )
        with self.assertRaises(ValueError):
            nqlib.System(
                A=[1],
                B1=numpy.random.randn(*B2.shape),
                B2=B2,
                C1=numpy.random.randn(*C2.shape),
                C2=C2,
                D1=numpy.random.randn(*D_shape),
                D2=numpy.random.randn(*D_shape),
            )

    def test_dynamic_quantizer_parameters(self):
        import nqlib
        for length in range(1, 50):
            parameters = numpy.random.randn(length * 2) * 10
            q = nqlib.StaticQuantizer.mid_tread(1)
            Q = nqlib.DynamicQuantizer.from_SISO_parameters(
                parameters,
                q=q,
            )
            Q_parameters = Q.to_parameters()
            self.assertTrue(
                all(
                    numpy.isclose(
                        Q_parameters,
                        parameters,
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                ),
                msg=(
                    f"Q.to_parameters() = {Q_parameters}, "
                    f"parameters = {parameters}"
                ),
            )

    def test_AG_serial_decomposition(self):
        import nqlib
        z = control.tf('z')
        P = nqlib.Plant.from_TF(
            0.01 * (z + 1.2) / (z - 0.95) / (z - 0.9)
        )
        ideal_system = nqlib.System.from_FF(P)

        q = nqlib.StaticQuantizer.mid_tread(d=2)

        Q, E = nqlib.DynamicQuantizer.design_AG(
            ideal_system,
            q=q,
            verbose=True,
        )

        self.assertAlmostEqual(
            0.022, E,
            msg=f"actual E = \n{E}",
        )
        self.assertAlmostEqual(
            ideal_system.E(Q), E,
            msg=f"actual E = \n{E}",
        )

    def test_AG(self):
        import nqlib
        P = nqlib.Plant(A=[[1.15, 0.05],
                           [0, 0.99]],
                        B=[[0.004],
                           [0.099]],
                        C1=[1, 0],
                        C2=numpy.eye(2))
        K = nqlib.Controller(A=0,
                             B1=0,
                             B2=[0, 0],
                             C=0,
                             D1=1,
                             D2=[-20, -3])
        ideal_system = nqlib.System.from_FB_connection_with_input_quantizer(P, K)

        q = nqlib.StaticQuantizer.mid_tread(d=2)

        Q, E = nqlib.DynamicQuantizer.design_AG(ideal_system,
                                                q=q,
                                                verbose=True)

        self.assertAlmostEqual(
            0.004, E,
            msg=f"actual E = \n{E}",
        )
        self.assertAlmostEqual(
            ideal_system.E(Q), E,
            msg=f"actual E = \n{E}",
        )

    def test_LP(self):
        import nqlib
        G = nqlib.System(
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

        q = nqlib.StaticQuantizer.mid_tread(d=2)

        Q, E = nqlib.DynamicQuantizer.design_LP(G,
                                                q=q,
                                                T=100,
                                                gain_wv=2,
                                                dim=5,
                                                verbose=True)

        self.assertAlmostEqual(
            0.023360532739376112, E,
            msg=f"LP actual E = \n{E}",
        )
        self.assertAlmostEqual(
            G.E(Q, 100), E,
            msg=f"estimation actual E = \n{E}",
        )
        self.assertGreaterEqual(Q.gain_wv(100), 2)  # E >= max

    def test_LP_MIMO(self):
        import nqlib
        G = nqlib.System(
            A=[[1.15, 0.05],
               [0.00, 0.99]],
            B1=[[0.1, 0.],
                [1.0, 0.]],
            B2=[[0.004, 0.01],
                [0.099, 0.1]],
            C1=[1., 0.],
            C2=[[-5., -3.],
                [-1, 1]],
            D1=[0, 1],
            D2=[[1, 0],
                [1, 1]],
        )

        q = nqlib.StaticQuantizer.mid_tread(d=2)

        Q, E = nqlib.DynamicQuantizer.design_LP(G,
                                                q=q,
                                                T=11,
                                                gain_wv=2,
                                                dim=2,
                                                verbose=True)

    def test_LP_MIMO2(self):
        import nqlib
        G = nqlib.System(
            A=[[1.15, 0.05],
               [0.00, 0.99]],
            B1=[[0.1, 0.],
                [1.0, 0.]],
            B2=[[0.004, 0.01],
                [0.099, 0.1]],
            C1=[1., 0.],
            C2=[[-5., -3.],
                [-1, 1]],
            D1=[0, 1],
            D2=[[1, 0],
                [1, 1]],
        )

        q = nqlib.StaticQuantizer.mid_tread(d=2)

        Q, E = nqlib.DynamicQuantizer.design_LP(G,
                                                q=q,
                                                T=11,
                                                gain_wv=2,
                                                dim=1,
                                                verbose=True)

    def test_GD(self):
        import nqlib

        def randn(shape): return 2 * numpy.random.randn(*shape)
        A = numpy.array([[1.15, 0.05],
                         [0.00, 0.99]])
        B2 = numpy.array([[0.004],
                          [0.099]])
        C1 = numpy.array([1., 0.])
        C2 = numpy.array([-15., -3.], ndmin=2)
        D_shape = (1, 1)

        ideal_system = nqlib.System(
            A=A,
            B1=randn(B2.shape),
            B2=B2,
            C1=C1,
            C2=C2,
            D1=randn(D_shape),
            D2=randn(D_shape),
        )

        q = nqlib.StaticQuantizer.mid_tread(d=2)
        Q, E = nqlib.DynamicQuantizer.design_AG(
            ideal_system,
            q=q,
            verbose=True,
        )

        print(ideal_system)
        self.assertIsNotNone(Q, "Quantizer Q is None")

        if E > 1000:
            print("E is inf")

        # 勾配法
        Q_g, E_g = nqlib.DynamicQuantizer.design_GD(
            ideal_system,
            q=q,
            dim=A.shape[0],
            verbose=True,
        )
        print(f"optimal E = {E}, optimized E_g = {E_g}")

        self.assertAlmostEqual(
            E_g, E,
            places=1,
            msg=(
                f"optimal E = {E}, optimized E_g = {E_g}\n"
                f"system:\n"
                f"{ideal_system}"
            ),
        )

    def test_DE(self):
        import nqlib

        def randn(shape): return 2 * numpy.random.randn(*shape)
        A = numpy.array([[1.15, 0.05],
                         [0.00, 0.99]])
        B2 = numpy.array([[0.004],
                          [0.099]])
        C1 = numpy.array([1., 0.])
        C2 = numpy.array([-15., -3.], ndmin=2)
        D_shape = (1, 1)

        ideal_system = nqlib.System(
            A=A,
            B1=randn(B2.shape),
            B2=B2,
            C1=C1,
            C2=C2,
            D1=randn(D_shape),
            D2=randn(D_shape),
        )

        q = nqlib.StaticQuantizer.mid_tread(d=2)
        Q, E = nqlib.DynamicQuantizer.design_AG(ideal_system,
                                                q=q,
                                                verbose=True)

        self.assertIsNotNone(Q, "Quantizer Q is None")

        if E > 1000:
            print("E is inf")

        # 進化型
        Q_d, E_d = nqlib.DynamicQuantizer.design_DE(
            ideal_system,
            q=q,
            dim=Q.minreal.A.shape[0],
            verbose=True,
        )

        self.assertAlmostEqual(
            E_d, E,
            places=1,
            msg=(
                f"optimal E = {E}, optimized E_d = {E_d}\n"
                f"system:\n"
                f"{ideal_system}"
            ),
        )

    def test_user_optimization(self):
        import nqlib
        from scipy.optimize import minimize
        P = nqlib.Plant(A=[[0.95, -0.37],
                        [0.25, 0.95]],
                        B=[[0.37],
                        [0.05]],
                        C1=[0, 1],
                        C2=numpy.eye(2))
        K = nqlib.Controller(A=1,
                             B1=1,
                             B2=[0, -1],
                             C=0.31,
                             D1=0,
                             D2=[-0.94, -0.66])
        G = nqlib.System.from_FBIQ(P, K)
        q = nqlib.StaticQuantizer.mid_tread(d=1)

        Q_odq, _ = nqlib.DynamicQuantizer.design_AG(
            G,
            q=q, allow_unstable=True,
        )
        if Q_odq is None:
            self.fail("Q_odq is None, design failed")

        def obj(x):
            Q = nqlib.DynamicQuantizer.from_SISO_parameters(x, q=q)
            return Q.objective_function(
                G,
                T=100,
                gain_wv=2,
                # obj_type="exp",
            )
        optimization_result = minimize(
            obj,
            Q_odq.to_parameters(),
            method="Nelder-Mead",
            options={
                "xatol": 1e-8,
                "disp": True,
                'maxiter': 10000,
            },
        )
        if not optimization_result.success:
            self.fail(f"Optimization failed: {optimization_result.message}")
        Q = nqlib.DynamicQuantizer.from_SISO_parameters(
            optimization_result.x,
            q=q,
        )
        self.assertTrue(
            Q.is_stable,
            msg="Q is unstable, optimization failed."
        )
        E = G.E(Q)
        self.assertTrue(
            E < numpy.inf,
            msg="E is infinite, optimization failed or system is unstable."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
