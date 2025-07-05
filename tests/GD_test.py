import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.size': 15,
    "mathtext.fontset": "cm",
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.top': True,
    'ytick.right': True,
    'pdf.fonttype': 42,  # type3フォント回避
    'ps.fonttype': 42,  # type3フォント回避
})

import random
import sys
import time
from typing import Callable, Iterable

import control
import numpy
import asyncio
import matplotlib.pyplot as plt

sys.path.append(r"H:\マイドライブ\IshikawaMinamiLab\研究\NQLib")
import nqlib_dev as nqlib

rand = random.random
def randn(shape): return 2 * numpy.random.randn(*shape)


def log(s, file=False):
    s = str(s)
    print(s)
    if file:
        with open("gdtest.txt", "a+", encoding="utf-8") as f:
            f.write(s + "\n")

def round(M):
    return numpy.round(M, decimals=2)


types = ["exp", "atan", "1.1", "100*1.1"]
typenames = [r"$exp(J)$", r"$atan(J)-\pi/2$", r"$1.1^J$", r"$-10000exp(-0.01J)$"]
times = 50
timeout = 10
time_taken = numpy.ones((len(types), times))*timeout

def try_system(ideal_system, type_index, gain_max, N_max, T, t):
    q = nqlib.StaticQuantizer.mid_tread(d=2)
    obj_type = types[type_index%len(types)]

    t_start = time.time()

    # 勾配法
    Q_GD, E_GD = nqlib\
        .DynamicQuantizer.design_GD(ideal_system,
                                    q=q,
                                    gain_wv=gain_max,
                                    dim=N_max,
                                    T=T,
                                    verbose=False,
                                    obj_type=obj_type)
    
    time_taken[type_index][t] = (time.time() - t_start)


def exec_timeout(f: Callable, args, timeout: int = 10):
    async def async_f():
        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, f, *args),  # 動機処理をタイムアウト付きで実行
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            log("Timeout\n")
        except Exception as e:
            log(type(e))
            print(e)

    asyncio.run(async_f())


def try_times(times=100, timeout=10):
    for t in range(times):
        rand_sys = control.drss(random.randint(2, 5)).minreal()
        A = rand_sys.A
        B2 = rand_sys.B
        C2 = rand_sys.C
        D_shape = rand_sys.D.shape

        ideal_system = nqlib.System(
            A=round(A - B2 @ C2),
            B1=round(randn(B2.shape)),
            B2=round(B2),
            C1=round(randn(C2.shape)),
            C2=round(C2),
            D1=round(randn(D_shape)),
            D2=round(randn(D_shape)),
        )

        T = 100
        gain_max = 1 + numpy.abs(randn([1]))
        N_max = random.randint(1, 5)
        
        print(ideal_system)
        for type_index in range(len(types)):
            print(types[type_index])
            exec_timeout(try_system, args=(ideal_system, type_index, gain_max, N_max, T, t), timeout=timeout)


if __name__ == "__main__":
    try_times(times=times, timeout=timeout)

    fig = plt.figure()
    ax = fig.add_subplot(111,ylim=(0,timeout))
    for i in range(len(types)):
        ax.plot(range(times), time_taken[i], label=typenames[i])
    ax.legend()
    plt.show()
