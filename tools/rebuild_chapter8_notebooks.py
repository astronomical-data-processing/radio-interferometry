import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = ROOT / "8_Calibration"

METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}


def _source_lines(text: str):
    content = dedent(text).strip("\n")
    if not content:
        return []
    return [line + "\n" for line in content.splitlines()]


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


IMPORT_BLOCK = """
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = None

STYLE_PATH = Path("../style/course.css")
TOGGLE_PATH = Path("../style/code_toggle.html")

if HTML is not None and display is not None:
    if STYLE_PATH.exists():
        display(HTML(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>"))
    if TOGGLE_PATH.exists():
        display(HTML(TOGGLE_PATH.read_text(encoding="utf-8")))

plt.rcParams["figure.figsize"] = (9, 4.5)
plt.rcParams["axes.grid"] = True
np.set_printoptions(precision=3, suppress=True)

RNG = np.random.default_rng(20260419)
"""


NOTEBOOKS = {
    "8_0_introduction.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [8. 校准](8_0_introduction.ipynb)
                * 下一节： [8.1 作为最小二乘问题的校准](8_1_calibration_least_squares_problem.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            校准是把“被系统响应污染的数据”重新映射回“尽可能接近天空真值的数据”的过程。若把某一时刻、某一基线上的可见度写成

            $$
            V_{pq}^{\rm obs} = G_p\,X_{pq}\,G_q^\ast + N_{pq},
            $$

            那么校准的核心问题就是：在不直接观测到真实天空相干量 $X_{pq}$ 的前提下，如何稳健地估计天线增益 $G_p$、评估模型误差，并判断哪些误差能够被“吸收”为方向无关项，哪些已经必须提升到方向依赖处理。

            这也是第 8 章与前面第 5 至第 7 章最关键的衔接点。前面几章告诉我们误差从哪里来；本章则讨论这些误差如何进入求解器、如何影响残差，以及为什么“求得一个解”并不等于“得到可信的科学结果”。
            """
        ),
        md("***"),
        md(
            r"""
            ## 第 8 章：校准，从误差模型到可操作解链 <a id='calibration:sec:intro'></a>

            本章围绕四个问题展开：

            - 如何把校准写成一个显式的最小二乘问题；
            - 为什么 1GC 能改正大部分方向无关系统误差；
            - 为什么 2GC 必须与成像和天空模型迭代地耦合；
            - 当主波束、指向误差、电离层等方向依赖效应出现时，为什么需要 3GC。

            与旧版讲义不同，这一章会尽量把“方程形式、求解策略、失败模式、实践流程”放在一起。也就是说，我们不只讨论公式，也讨论什么时候它们会失灵。
            """
        ),
        code(
            """
            times = np.linspace(0.0, 6.0, 240)

            amp_1 = 1.00 + 0.05 * np.sin(2.0 * np.pi * times / 3.1)
            phase_1 = 0.35 * np.sin(2.0 * np.pi * times / 2.6 + 0.2)
            gain_1 = amp_1 * np.exp(1j * phase_1)

            amp_2 = 0.95 + 0.08 * np.cos(2.0 * np.pi * times / 4.0 - 0.4)
            phase_2 = -0.28 * np.cos(2.0 * np.pi * times / 3.3 + 0.3)
            gain_2 = amp_2 * np.exp(1j * phase_2)

            true_vis = 1.2 * np.exp(1j * 0.18)
            noise = 0.02 * (RNG.normal(size=times.size) + 1j * RNG.normal(size=times.size))
            observed_vis = gain_1 * np.conj(gain_2) * true_vis + noise
            calibrated_vis = observed_vis / (gain_1 * np.conj(gain_2))

            amp_rms_before = np.sqrt(np.mean((np.abs(observed_vis) - np.abs(true_vis)) ** 2))
            amp_rms_after = np.sqrt(np.mean((np.abs(calibrated_vis) - np.abs(true_vis)) ** 2))
            phase_rms_before = np.sqrt(
                np.mean((np.angle(observed_vis / true_vis)) ** 2)
            )
            phase_rms_after = np.sqrt(
                np.mean((np.angle(calibrated_vis / true_vis)) ** 2)
            )

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

            axes[0].plot(times, np.abs(observed_vis), lw=1.5, color="tab:red", label="受污染振幅")
            axes[0].plot(times, np.abs(calibrated_vis), lw=1.8, color="tab:blue", label="改正后振幅")
            axes[0].axhline(np.abs(true_vis), ls="--", color="black", label="真实振幅")
            axes[0].set_ylabel("amplitude [arb.]")
            axes[0].set_title("Calibration removes time-dependent amplitude and phase corruption")
            axes[0].legend(loc="upper right")

            axes[1].plot(times, np.angle(observed_vis / true_vis), lw=1.5, color="tab:red", label="改正前相位误差")
            axes[1].plot(times, np.angle(calibrated_vis / true_vis), lw=1.8, color="tab:blue", label="改正后相位误差")
            axes[1].axhline(0.0, ls="--", color="black")
            axes[1].set_xlabel("time [hour]")
            axes[1].set_ylabel("phase error [rad]")
            axes[1].legend(loc="upper right")

            plt.tight_layout()
            print(f"振幅 RMS：改正前 {amp_rms_before:.3f}，改正后 {amp_rms_after:.3f}")
            print(f"相位 RMS：改正前 {phase_rms_before:.3f} rad，改正后 {phase_rms_after:.3f} rad")
            """
        ),
        md(
            r"""
            上面的例子只展示了最理想的情形：我们“知道”两面天线的复增益，因此能把基线上的污染完全除掉。真实问题当然没有这么简单。我们需要面对至少三类不完美：

            - **噪声与有限信噪比**：弱校准源会让解本身变得不稳定；
            - **天空模型误差**：若模型漏掉了重要结构，求解器可能把天空误差错误地吸收到增益里；
            - **误差模型不匹配**：若真实污染是方向依赖的、而我们却只拟合方向无关增益，那么“解得很好”的结果也可能是物理上错误的。

            因此，校准从来不是单纯的“求一个复数”，而是一个把观测设计、硬件误差、先验模型和成像结果联系在一起的闭环过程。
            """
        ),
        md(
            r"""
            #### 本章内容

            - [8.1 作为最小二乘问题的校准](8_1_calibration_least_squares_problem.ipynb)：把天线增益求解显式写成非线性最小二乘问题；
            - [8.2 第一代校准（1GC）](8_2_1gc.ipynb)：理解校准源、闭合量、增益转移和实际观测链；
            - [8.3 第二代校准（2GC）](8_3_2gc.ipynb)：讨论自校准与天空模型之间的迭代耦合；
            - [8.4 第三代校准（3GC）](8_4_3gc.ipynb)：讨论方向依赖校准、差分增益和宽场处理的必要性；
            - [8.5 延伸阅读与参考文献](8_5_further_reading_and_references.ipynb)：给出进一步深入学习校准的经典资料。
            - [校准习题](8_problem_set.ipynb)：把最小二乘、参考天线、模型不完整和交替求解器做成可练习任务。

            建议的阅读顺序是先完成 8.1 中的求解器直觉，再进入 8.2 至 8.4。这样在第 9 章构建完整的数据处理工作流时，会更容易理解为什么某些步骤必须按固定顺序执行。
            """
        ),
    ],
    "8_2_1gc.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [8. 校准](8_0_introduction.ipynb)
                * 上一节： [8.1 作为最小二乘问题的校准](8_1_calibration_least_squares_problem.ipynb)
                * 下一节： [8.3 第二代校准（2GC）](8_3_2gc.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            第一代校准（1GC）解决的是最经典、也最普遍的一类问题：我们利用一个或多个外场校准源去追踪系统增益、带通、延迟或绝对通量标尺，然后把这些解转移到目标场。

            它的成功建立在一个关键假设上：在校准源方向上求得的误差项，能够在目标场方向上近似成立。这个假设在窄视场、较稳定的大气和方向依赖效应不太严重时通常是合理的，因此 1GC 仍然是大多数处理链的起点。
            """
        ),
        md("***"),
        md(
            r"""
            ## 8.2 第一代校准（1GC）<a id='cal:sec:1gccal'></a>

            在现代软件中，1GC 往往表现为一条看起来很“工程化”的解链，例如 `setjy -> bandpass -> gaincal -> fluxscale -> applycal`。但在算法上，它仍然是一个把观测数据、校准源模型和天线增益联系起来的求解问题。

            这一节重点关注三件事：

            - 校准源为什么能约束系统增益；
            - 为什么解需要在时间和频率上插值、平滑或转移；
            - 为什么闭合量是诊断 1GC 是否合理的重要工具。
            """
        ),
        code(
            """
            def baseline_pairs(nant):
                return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


            def point_source_model_cube(flux, nt, nant):
                model = np.full((nt, nant, nant), flux + 0.0j, dtype=complex)
                for p in range(nant):
                    model[:, p, p] = 0.0
                return model


            def apply_gains(model, gains, noise_std=0.0, rng=None):
                data = gains[:, :, None] * model * np.conj(gains[:, None, :])
                if noise_std > 0.0:
                    if rng is None:
                        rng = np.random.default_rng(0)
                    noise = noise_std * (
                        rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
                    )
                    data = data + noise
                for p in range(data.shape[1]):
                    data[:, p, p] = 0.0
                return data


            def solve_gains(data, model, n_iter=30, ref_ant=0, phase_only=False):
                nt, nant, _ = data.shape
                gains = np.ones((nt, nant), dtype=complex)
                eps = 1e-12

                for t in range(nt):
                    gt = np.ones(nant, dtype=complex)
                    for _ in range(n_iter):
                        new = gt.copy()
                        for p in range(nant):
                            mask = np.ones(nant, dtype=bool)
                            mask[p] = False
                            num = np.sum(data[t, p, mask] * gt[mask] * np.conj(model[t, p, mask]))
                            den = np.sum(
                                np.abs(gt[mask]) ** 2 * np.abs(model[t, p, mask]) ** 2
                            ) + eps
                            new[p] = num / den

                        if phase_only:
                            amp = np.maximum(np.abs(new), eps)
                            new = new / amp

                        ref = new[ref_ant]
                        new = new / (ref / max(np.abs(ref), eps))
                        gt = new

                    gains[t] = gt

                return gains


            def gain_track(times, nant):
                ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
                amp = 1.0 + 0.04 * np.sin(times[:, None] * (0.8 + 0.05 * np.arange(nant)) + ant_phase)
                phase = 0.25 * np.sin(times[:, None] * (1.2 + 0.08 * np.arange(nant)) + 0.5 * ant_phase)
                return amp * np.exp(1j * phase)


            def interpolate_complex(times_src, gains_src, times_dst):
                amp = np.abs(gains_src)
                phase = np.unwrap(np.angle(gains_src))
                amp_i = np.vstack(
                    [np.interp(times_dst, times_src, amp[:, ant]) for ant in range(amp.shape[1])]
                ).T
                phase_i = np.vstack(
                    [np.interp(times_dst, times_src, phase[:, ant]) for ant in range(phase.shape[1])]
                ).T
                return amp_i * np.exp(1j * phase_i)


            def closure_phase(vis12, vis23, vis31):
                return np.angle(vis12 * vis23 * vis31)


            def closure_amplitude(vis12, vis34, vis13, vis24):
                return np.abs(vis12 * vis34 / (vis13 * vis24))
            """
        ),
        md(
            r"""
            ### 8.2.1 校准源观测与解转移 <a id='cal:sec:calobs'></a>

            典型的 1GC 观测不会只看目标源，而是把几类校准源穿插进时间轴中：

            - **绝对通量校准源**：定义整体振幅标尺；
            - **带通校准源**：追踪频率响应；
            - **相位校准源**：高时间频率地追踪大气和电子链路相位变化；
            - **目标场**：真正的科学观测。

            下面这张示意图把这种“交替观测”的节奏表现出来。真正的阵列调度更复杂，但核心思想相同：用较短、较可靠的校准扫描去支撑更长的目标场积分。
            """
        ),
        code(
            """
            blocks = [
                ("通量校准源", "tab:orange", [(0.0, 0.35), (5.5, 5.85)]),
                ("相位校准源", "tab:green", [(0.8, 1.0), (2.1, 2.3), (3.4, 3.6), (4.7, 4.9)]),
                ("目标场", "tab:blue", [(0.35, 0.8), (1.0, 2.1), (2.3, 3.4), (3.6, 4.7), (4.9, 5.5)]),
            ]

            fig, ax = plt.subplots(figsize=(10, 2.8))

            for idx, (label, color, spans) in enumerate(blocks):
                for start, stop in spans:
                    ax.barh(idx, stop - start, left=start, height=0.45, color=color, alpha=0.85)

            ax.set_yticks(range(len(blocks)))
            ax.set_yticklabels([item[0] for item in blocks])
            ax.set_xlabel("观测时间 [hour]")
            ax.set_title("1GC 中常见的交替观测节奏")
            ax.set_xlim(0.0, 6.0)
            plt.tight_layout()
            """
        ),
        md(
            r"""
            这个时间结构直接决定了解的物理含义。若相位校准源与目标场相距很远，或者扫描间隔太长，那么即便校准源上的解很好，也可能无法准确代表目标场上的传播误差。后面我们会用一个简单模拟来量化这种“转移误差”。
            """
        ),
        md(
            r"""
            ### 8.2.2 在点源校准源上求解方向无关增益

            对位于相位中心、结构已知的点源，模型可见度矩阵最简单。下面的例子用一个亮点源模拟相位校准源，再用交替迭代求解器恢复每面天线的复增益。你可以把它视为 `gaincal` 一类任务背后的最简化原型。
            """
        ),
        code(
            """
            nant = 6
            times_cal = np.linspace(0.0, 6.0, 36)
            model_cal = point_source_model_cube(flux=5.0, nt=times_cal.size, nant=nant)

            true_gains_cal = gain_track(times_cal, nant)
            data_cal = apply_gains(model_cal, true_gains_cal, noise_std=0.05, rng=RNG)
            solved_gains_cal = solve_gains(data_cal, model_cal, n_iter=35, ref_ant=0, phase_only=False)

            baseline = (1, 4)
            corrected_cal = data_cal / (
                solved_gains_cal[:, :, None] * np.conj(solved_gains_cal[:, None, :]) + 1e-12
            )

            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")

            axes[0, 0].plot(times_cal, np.abs(true_gains_cal[:, 3]), color="black", lw=2.0, label="真实")
            axes[0, 0].plot(times_cal, np.abs(solved_gains_cal[:, 3]), color="tab:blue", lw=1.8, label="求解")
            axes[0, 0].set_ylabel("|g_3|")
            axes[0, 0].legend(loc="upper right")
            axes[0, 0].set_title("Antenna 3 amplitude")

            axes[1, 0].plot(times_cal, np.angle(true_gains_cal[:, 3]), color="black", lw=2.0, label="真实")
            axes[1, 0].plot(times_cal, np.angle(solved_gains_cal[:, 3]), color="tab:blue", lw=1.8, label="求解")
            axes[1, 0].set_ylabel("phase [rad]")
            axes[1, 0].set_xlabel("time [hour]")
            axes[1, 0].set_title("Antenna 3 phase")

            axes[0, 1].plot(times_cal, np.abs(data_cal[:, baseline[0], baseline[1]]), color="tab:red", lw=1.5)
            axes[0, 1].plot(
                times_cal,
                np.abs(corrected_cal[:, baseline[0], baseline[1]]),
                color="tab:blue",
                lw=1.8,
            )
            axes[0, 1].axhline(5.0, color="black", ls="--")
            axes[0, 1].set_ylabel("amplitude [Jy]")
            axes[0, 1].set_title(f"Baseline {baseline[0]}-{baseline[1]} amplitude")

            axes[1, 1].plot(
                times_cal,
                np.angle(data_cal[:, baseline[0], baseline[1]] / 5.0),
                color="tab:red",
                lw=1.5,
                label="改正前",
            )
            axes[1, 1].plot(
                times_cal,
                np.angle(corrected_cal[:, baseline[0], baseline[1]] / 5.0),
                color="tab:blue",
                lw=1.8,
                label="改正后",
            )
            axes[1, 1].axhline(0.0, color="black", ls="--")
            axes[1, 1].set_ylabel("phase error [rad]")
            axes[1, 1].set_xlabel("time [hour]")
            axes[1, 1].set_title(f"Baseline {baseline[0]}-{baseline[1]} phase")
            axes[1, 1].legend(loc="upper right")

            plt.tight_layout()
            """
        ),
        md(
            r"""
            可以看到，求解器能够把“基线上的复误差”分解回“天线上的复增益”。这正是天线基校准的核心优势。只要污染主要来自每面天线自身的方向无关误差，基线数目随 $N_{\rm ant}^2$ 增长，而待求参数只随 $N_{\rm ant}$ 增长，因此问题通常是强约束的。
            """
        ),
        md(
            r"""
            ### 8.2.3 解转移、插值与目标场剩余误差

            下面把同一组校准解转移到目标场。为了模拟“校准源和目标场不完全等价”，我们额外加入一个与方向相关的相位残差项，代表校准源与目标场穿过不同大气路径后的差异。你会看到，1GC 虽然显著改善了数据，但不能把这类误差全部消除。
            """
        ),
        code(
            """
            times_target = np.linspace(0.15, 5.95, 96)
            model_target = point_source_model_cube(flux=1.5, nt=times_target.size, nant=nant)

            common_gains = gain_track(times_target, nant)
            ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
            target_extra_phase = 0.12 * np.sin(2.8 * times_target[:, None] + ant_phase[None, :])
            target_direction_term = np.exp(1j * target_extra_phase)
            true_gains_target = common_gains * target_direction_term

            data_target = apply_gains(model_target, true_gains_target, noise_std=0.02, rng=RNG)

            interp_solutions = interpolate_complex(times_cal, solved_gains_cal, times_target)
            corrected_target = data_target / (
                interp_solutions[:, :, None] * np.conj(interp_solutions[:, None, :]) + 1e-12
            )

            test_bl = (0, 5)
            raw_phase = np.angle(data_target[:, test_bl[0], test_bl[1]] / 1.5)
            corr_phase = np.angle(corrected_target[:, test_bl[0], test_bl[1]] / 1.5)

            rms_raw = np.sqrt(np.mean(raw_phase**2))
            rms_corr = np.sqrt(np.mean(corr_phase**2))

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

            axes[0].plot(times_target, raw_phase, color="tab:red", lw=1.5, label="改正前")
            axes[0].plot(times_target, corr_phase, color="tab:blue", lw=1.8, label="1GC 后")
            axes[0].axhline(0.0, color="black", ls="--")
            axes[0].set_xlabel("time [hour]")
            axes[0].set_ylabel("baseline phase error [rad]")
            axes[0].set_title(f"Baseline {test_bl[0]}-{test_bl[1]} target phase")
            axes[0].legend(loc="upper right")

            axes[1].plot(times_target, np.abs(data_target[:, test_bl[0], test_bl[1]]), color="tab:red", lw=1.5)
            axes[1].plot(
                times_target,
                np.abs(corrected_target[:, test_bl[0], test_bl[1]]),
                color="tab:blue",
                lw=1.8,
            )
            axes[1].axhline(1.5, color="black", ls="--")
            axes[1].set_xlabel("time [hour]")
            axes[1].set_ylabel("amplitude [Jy]")
            axes[1].set_title("Transferred solutions improve, but do not fully remove, target errors")

            plt.tight_layout()
            print(f"目标场相位 RMS：改正前 {rms_raw:.3f} rad，1GC 后 {rms_corr:.3f} rad")
            """
        ),
        md(
            r"""
            这正是实践中常见的情形：1GC 通常能去掉最大尺度、最平滑的系统误差，但目标场仍会留下模型不匹配或方向依赖残差。也因此，1GC 之后往往还要进入自校准，甚至进一步进入 3GC。
            """
        ),
        md(
            r"""
            ### 8.2.4 闭合量：为什么它们是诊断工具 <a id='cal:sec:closure'></a>

            闭合相位和闭合振幅是 1GC 中最重要的诊断量之一。对纯天线基的方向无关增益来说，闭合量会把每面天线自身的复增益抵消掉，因此它们反映的是**天空结构与非天线基误差**，而不是简单的增益漂移。
            """
        ),
        code(
            """
            u12, u23, u31 = 42.0, 31.0, -73.0
            u13, u24, u34 = 73.0, 55.0, 24.0

            fluxes = np.array([1.0, 0.35, 0.18])
            ls = np.array([0.0, 0.012, -0.019])

            def structured_vis(u):
                return np.sum(fluxes * np.exp(-2j * np.pi * u * ls))


            v12 = structured_vis(u12)
            v23 = structured_vis(u23)
            v31 = structured_vis(u31)
            v13 = structured_vis(u13)
            v24 = structured_vis(u24)
            v34 = structured_vis(u34)

            g = np.array(
                [
                    1.05 * np.exp(1j * 0.35),
                    0.92 * np.exp(-1j * 0.18),
                    1.08 * np.exp(1j * 0.12),
                    0.97 * np.exp(-1j * 0.25),
                ]
            )

            v12_obs = g[0] * np.conj(g[1]) * v12
            v23_obs = g[1] * np.conj(g[2]) * v23
            v31_obs = g[2] * np.conj(g[0]) * v31
            v13_obs = g[0] * np.conj(g[2]) * v13
            v24_obs = g[1] * np.conj(g[3]) * v24
            v34_obs = g[2] * np.conj(g[3]) * v34

            cp_true = closure_phase(v12, v23, v31)
            cp_obs = closure_phase(v12_obs, v23_obs, v31_obs)
            ca_true = closure_amplitude(v12, v34, v13, v24)
            ca_obs = closure_amplitude(v12_obs, v34_obs, v13_obs, v24_obs)

            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
            axes[0].bar(["真实闭合相位", "受增益污染后"], [cp_true, cp_obs], color=["black", "tab:blue"])
            axes[0].set_ylabel("closure phase [rad]")
            axes[0].set_title("Closure phase is gain-invariant")

            axes[1].bar(["真实闭合振幅", "受增益污染后"], [ca_true, ca_obs], color=["black", "tab:green"])
            axes[1].set_ylabel("closure amplitude")
            axes[1].set_title("Closure amplitude is also gain-invariant")
            plt.tight_layout()

            print(f"闭合相位：真实 {cp_true:.4f} rad，受污染后 {cp_obs:.4f} rad")
            print(f"闭合振幅：真实 {ca_true:.4f}，受污染后 {ca_obs:.4f}")
            """
        ),
        md(
            r"""
            这也是为什么在真实处理时，我们经常先看闭合量，再看求解器是否“收敛”。如果闭合量本身已经表现出强烈的非物理结构，那么问题可能不是一个简单的天线基增益，而是 RFI、相关器异常、错误的天空模型，或更复杂的方向依赖误差。
            """
        ),
        md(
            r"""
            ### 8.2.5 1GC 的实践解链

            对连续谱观测，一个典型的 1GC 处理链通常包含下列步骤：

            - `listobs` / `plotms`：先做数据检查，确认观测结构和异常扫描；
            - `setjy`：给绝对通量校准源设定模型；
            - `bandpass`：求带通响应；
            - `gaincal`：求时间依赖增益与相位；
            - `fluxscale`：把振幅解绑定到绝对通量标尺；
            - `applycal`：把解转移到目标场；
            - `plotms` / quick image：检查改正后数据与图像质量。

            真正的关键不在命令名字本身，而在于理解每一步对应什么物理量、它假设了什么、以及失败时最先会在哪类诊断图上暴露出来。
            """
        ),
    ],
    "8_3_2gc.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [8. 校准](8_0_introduction.ipynb)
                * 上一节： [8.2 第一代校准（1GC）](8_2_1gc.ipynb)
                * 下一节： [8.4 第三代校准（3GC）](8_4_3gc.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            第二代校准（2GC）通常指方向无关自校准。它不再依赖外场校准源，而是直接利用目标场数据本身求解增益，再把改正后的数据送回成像与去卷积步骤，持续更新天空模型。

            自校准的威力来自闭环迭代，但它的风险也来自闭环迭代：如果天空模型有系统性错误，求解器会很乐于把模型误差“解释”为增益。理解这一点，是把 2GC 用对而不是用坏的关键。
            """
        ),
        md("***"),
        md(
            r"""
            ## 8.3 第二代校准（2GC）：方向无关自校准

            这一节通过一个最小化的一维成像实验来观察自校准的三个核心特征：

            - 用不完整模型做自校准时，图像会改善，但仍可能带偏；
            - 当天空模型更完整时，残差和动态范围会显著改善；
            - 求解时间间隔过长或过短都会带来代价。
            """
        ),
        code(
            """
            def baseline_pairs(nant):
                return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


            def baseline_u(ant_x_m, hour_angle_h, wavelength_m=0.214):
                pairs = baseline_pairs(len(ant_x_m))
                hour_angle_rad = np.deg2rad(15.0 * hour_angle_h)
                u = np.zeros((hour_angle_h.size, len(pairs)))
                for ti, ha in enumerate(hour_angle_rad):
                    for bi, (p, q) in enumerate(pairs):
                        u[ti, bi] = (ant_x_m[q] - ant_x_m[p]) * np.sin(ha) / wavelength_m
                return pairs, u


            def sky_vis_1d(u, fluxes, ls):
                vis = np.zeros_like(u, dtype=complex)
                for flux, ll in zip(fluxes, ls):
                    vis += flux * np.exp(-2j * np.pi * u * ll)
                return vis


            def vec_to_cube(vis_vec, pairs, nant):
                nt, nb = vis_vec.shape
                cube = np.zeros((nt, nant, nant), dtype=complex)
                for bi, (p, q) in enumerate(pairs):
                    cube[:, p, q] = vis_vec[:, bi]
                    cube[:, q, p] = np.conj(vis_vec[:, bi])
                return cube


            def cube_to_vec(cube, pairs):
                return np.stack([cube[:, p, q] for p, q in pairs], axis=1)


            def apply_gains(model_cube, gains, noise_std=0.0, rng=None):
                data = gains[:, :, None] * model_cube * np.conj(gains[:, None, :])
                if noise_std > 0.0:
                    if rng is None:
                        rng = np.random.default_rng(0)
                    noise = noise_std * (
                        rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
                    )
                    data = data + noise
                for p in range(data.shape[1]):
                    data[:, p, p] = 0.0
                return data


            def solve_gains(data, model, n_iter=30, ref_ant=0, phase_only=False):
                nt, nant, _ = data.shape
                gains = np.ones((nt, nant), dtype=complex)
                eps = 1e-12

                for t in range(nt):
                    gt = np.ones(nant, dtype=complex)
                    for _ in range(n_iter):
                        new = gt.copy()
                        for p in range(nant):
                            mask = np.ones(nant, dtype=bool)
                            mask[p] = False
                            num = np.sum(data[t, p, mask] * gt[mask] * np.conj(model[t, p, mask]))
                            den = np.sum(
                                np.abs(gt[mask]) ** 2 * np.abs(model[t, p, mask]) ** 2
                            ) + eps
                            new[p] = num / den

                        if phase_only:
                            amp = np.maximum(np.abs(new), eps)
                            new = new / amp

                        ref = new[ref_ant]
                        new = new / (ref / max(np.abs(ref), eps))
                        gt = new

                    gains[t] = gt

                return gains


            def dirty_image_1d(u, vis, l_grid):
                u_flat = np.concatenate([u.ravel(), -u.ravel()])
                vis_flat = np.concatenate([vis.ravel(), np.conj(vis.ravel())])
                phase = np.exp(2j * np.pi * u_flat[:, None] * l_grid[None, :])
                return np.real(vis_flat @ phase / vis_flat.size)


            def off_source_rms(image, l_grid, source_positions, exclusion=0.005):
                mask = np.ones_like(l_grid, dtype=bool)
                for pos in source_positions:
                    mask &= np.abs(l_grid - pos) > exclusion
                return np.sqrt(np.mean(image[mask] ** 2))


            def average_in_time(arr, block):
                n = arr.shape[0] // block
                trimmed = arr[: n * block]
                new_shape = (n, block) + arr.shape[1:]
                return trimmed.reshape(new_shape).mean(axis=1)


            ant_x = np.array([0.0, 38.0, 102.0, 188.0, 310.0, 462.0])
            times_h = np.linspace(-3.0, 3.0, 48)
            pairs, u = baseline_u(ant_x, times_h)
            nant = ant_x.size

            flux_true = np.array([1.0, 0.28, 0.14])
            l_true = np.array([0.0, 0.011, -0.021])
            vis_true = sky_vis_1d(u, flux_true, l_true)
            model_true = vec_to_cube(vis_true, pairs, nant)

            ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
            amp = 1.0 + 0.03 * np.sin(1.1 * times_h[:, None] + ant_phase[None, :])
            phase = 0.55 * np.sin(2.4 * times_h[:, None] + 0.6 * ant_phase[None, :])
            true_gains = amp * np.exp(1j * phase)

            data = apply_gains(model_true, true_gains, noise_std=0.02, rng=RNG)

            flux_initial = np.array([1.0])
            l_initial = np.array([0.0])
            vis_initial = sky_vis_1d(u, flux_initial, l_initial)
            model_initial = vec_to_cube(vis_initial, pairs, nant)
            """
        ),
        md(
            r"""
            ### 8.3.1 自校准的闭环：模型、求解与再成像 <a id='cal:sec:selfcal'></a>

            先从一个故意不完整的天空模型开始。这里我们只保留视场中心的亮源，而忽略两个较弱的离轴源。这样做并不合理，但它非常适合说明“错误模型也能让图像变好，却不一定让结果更真实”这一点。
            """
        ),
        code(
            """
            gains_phase_only = solve_gains(data, model_initial, n_iter=30, ref_ant=0, phase_only=True)
            data_phase_only = data / (
                gains_phase_only[:, :, None] * np.conj(gains_phase_only[:, None, :]) + 1e-12
            )

            gains_full = solve_gains(data, model_true, n_iter=30, ref_ant=0, phase_only=False)
            data_full = data / (
                gains_full[:, :, None] * np.conj(gains_full[:, None, :]) + 1e-12
            )

            l_grid = np.linspace(-0.04, 0.04, 700)
            image_raw = dirty_image_1d(u, cube_to_vec(data, pairs), l_grid)
            image_phase_only = dirty_image_1d(u, cube_to_vec(data_phase_only, pairs), l_grid)
            image_full = dirty_image_1d(u, cube_to_vec(data_full, pairs), l_grid)

            true_sky_image = np.zeros_like(l_grid)
            for flux, ll in zip(flux_true, l_true):
                idx = np.argmin(np.abs(l_grid - ll))
                true_sky_image[idx] = flux

            rms_raw = off_source_rms(image_raw, l_grid, l_true)
            rms_phase = off_source_rms(image_phase_only, l_grid, l_true)
            rms_full = off_source_rms(image_full, l_grid, l_true)
            peak_raw = image_raw.max()
            peak_phase = image_phase_only.max()
            peak_full = image_full.max()

            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

            for ax, img, title, color in [
                (axes[0], image_raw, "未自校准", "tab:red"),
                (axes[1], image_phase_only, "相位自校准（模型不完整）", "tab:orange"),
                (axes[2], image_full, "幅相自校准（模型更完整）", "tab:blue"),
            ]:
                ax.plot(l_grid, img, color=color, lw=1.6)
                for pos in l_true:
                    ax.axvline(pos, color="black", ls=":", alpha=0.6)
                ax.set_ylabel("dirty image")
                ax.set_title(title)

            axes[2].set_xlabel("direction cosine l")
            plt.tight_layout()

            print(f"离源 RMS：原始 {rms_raw:.4f}，相位自校后 {rms_phase:.4f}，完整自校后 {rms_full:.4f}")
            print(f"峰值响应：原始 {peak_raw:.4f}，相位自校后 {peak_phase:.4f}，完整自校后 {peak_full:.4f}")
            """
        ),
        md(
            r"""
            这个例子里，自校准确实让图像逐步变干净了，但两轮迭代的物理含义并不相同：

            - **第一轮相位自校准**主要利用亮核建立相位参考，因此中心结构显著改善；
            - **第二轮更完整模型**把离轴源也纳入可见度预测中，于是求解器不再需要把它们“错误地吸收入增益”，残差才会进一步下降。

            这也是为什么实践中常说“自校准不是魔法，它依赖好的 sky model”。
            """
        ),
        md(
            r"""
            ### 8.3.2 解间隔（solution interval）的权衡

            自校准还有一个核心参数是解间隔。间隔太长，会把快速相位变化平均掉；间隔太短，则可能因为信噪比不足而得到噪声主导的解。下面的实验只模拟前一类风险，即相位波动快于求解间隔时会发生什么。
            """
        ),
        code(
            """
            intervals = [1, 2, 4, 8]
            residual_rms = []

            for block in intervals:
                if times_h.size // block < 2:
                    continue
                data_avg = average_in_time(data, block)
                model_avg = average_in_time(model_true, block)
                gains_avg = solve_gains(data_avg, model_avg, n_iter=25, ref_ant=0, phase_only=True)
                gains_interp = np.repeat(gains_avg, block, axis=0)[: times_h.size]
                data_corr = data / (
                    gains_interp[:, :, None] * np.conj(gains_interp[:, None, :]) + 1e-12
                )
                residual = cube_to_vec(data_corr - model_true, pairs)
                residual_rms.append(np.sqrt(np.mean(np.abs(residual) ** 2)))

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(intervals[: len(residual_rms)], residual_rms, marker="o", lw=2.0, color="tab:purple")
            ax.set_xlabel("solution interval [number of time samples]")
            ax.set_ylabel("residual visibility RMS")
            ax.set_title("Too-long solution intervals fail to track rapid phase changes")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            真正的数据处理中，这条曲线往往是一个两端都不理想的“碗形”关系：太短的间隔噪声太大，太长的间隔又跟不上真实变化。因此，合适的解间隔总要结合阵列灵敏度、目标场亮度、频段和天气条件来选。
            """
        ),
        md(
            r"""
            ### 8.3.3 自校准、混合成图与失败模式

            经典的 *hybrid mapping* 可以概括成一个循环：

            1. 用当前数据做成像和去卷积；
            2. 从图像中提取或更新天空模型；
            3. 用该模型求解增益；
            4. 改正数据并返回第 1 步。

            这个闭环在以下情况下尤其容易出问题：

            - 初始模型过于贫乏，导致解把扩展结构“吃掉”；
            - 场内总信噪比不足，却强行做过密的解；
            - 幅相自校准开始得太早，导致绝对通量标尺漂移；
            - 把方向依赖残差误当成方向无关相位误差处理。

            因此，一个稳健的经验做法通常是：先做 1GC，再做相位自校准，确认模型和残差稳定后，再谨慎进入幅相自校准。若此时视场仍表现出明显位置相关残差，就该考虑 3GC 了。
            """
        ),
    ],
    "8_4_3gc.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [8. 校准](8_0_introduction.ipynb)
                * 上一节： [8.3 第二代校准（2GC）](8_3_2gc.ipynb)
                * 下一节： [8.5 延伸阅读与参考文献](8_5_further_reading_and_references.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            当误差明显依赖天空方向时，方向无关自校准就不再足够。主波束变化、指向误差、电离层相位屏、对流层水汽以及宽场极化泄漏，都会让“同一面天线只有一个复增益”这个近似失效。

            第三代校准（3GC）的任务，就是在可计算成本可接受的前提下，把这些方向依赖效应纳入模型或求解器，使残差不再随着天空位置系统性增长。
            """
        ),
        md("***"),
        md(
            r"""
            ## 8.4 第三代校准（3GC）：方向依赖自校准

            这一节不追求穷尽所有 3GC 算法，而是建立三个最重要的判断：

            - 什么时候 2GC 的方向无关近似已经失效；
            - 为什么“一个全场单一增益”无法同时解释多个方向上的偏差；
            - 3GC 中常见的几条路线分别解决什么问题。
            """
        ),
        code(
            """
            def baseline_pairs(nant):
                return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


            def baseline_u(ant_x_m, hour_angle_h, wavelength_m=0.214):
                pairs = baseline_pairs(len(ant_x_m))
                hour_angle_rad = np.deg2rad(15.0 * hour_angle_h)
                u = np.zeros((hour_angle_h.size, len(pairs)))
                for ti, ha in enumerate(hour_angle_rad):
                    for bi, (p, q) in enumerate(pairs):
                        u[ti, bi] = (ant_x_m[q] - ant_x_m[p]) * np.sin(ha) / wavelength_m
                return pairs, u


            def direction_visibilities(u, fluxes, ls):
                vis = []
                for flux, ll in zip(fluxes, ls):
                    vis.append(flux * np.exp(-2j * np.pi * u * ll))
                return np.stack(vis, axis=-1)


            def beam_gain(l, offset, sigma):
                return np.exp(-0.5 * ((l - offset) / sigma) ** 2)


            def dirty_image_1d(u, vis, l_grid):
                u_flat = np.concatenate([u.ravel(), -u.ravel()])
                vis_flat = np.concatenate([vis.ravel(), np.conj(vis.ravel())])
                phase = np.exp(2j * np.pi * u_flat[:, None] * l_grid[None, :])
                return np.real(vis_flat @ phase / vis_flat.size)


            ant_x = np.array([0.0, 42.0, 110.0, 205.0, 325.0, 470.0])
            times_h = np.linspace(-2.8, 2.8, 42)
            pairs, u = baseline_u(ant_x, times_h)
            nant = ant_x.size

            fluxes = np.array([1.0, 0.36, 0.22])
            ls = np.array([0.0, 0.018, -0.026])
            source_vis = direction_visibilities(u, fluxes, ls)

            static_offsets = np.array([0.000, 0.0015, -0.0010, 0.0022, -0.0018, 0.0012])
            dynamic_offsets = 0.0012 * np.sin(1.4 * times_h[:, None] + np.linspace(0.0, np.pi, nant)[None, :])
            pointing_offsets = static_offsets[None, :] + dynamic_offsets
            sigma = 0.020

            beam = np.zeros((times_h.size, nant, len(ls)))
            for si, ll in enumerate(ls):
                beam[:, :, si] = beam_gain(ll, pointing_offsets, sigma)

            data = np.zeros((times_h.size, len(pairs)), dtype=complex)
            for bi, (p, q) in enumerate(pairs):
                data[:, bi] = np.sum(beam[:, p, :] * beam[:, q, :] * source_vis[:, bi, :], axis=1)

            model_di = np.sum(source_vis, axis=-1)
            model_dd = np.zeros_like(model_di)
            for bi, (p, q) in enumerate(pairs):
                model_dd[:, bi] = np.sum(beam[:, p, :] * beam[:, q, :] * source_vis[:, bi, :], axis=1)

            residual_di = data - model_di
            residual_dd = data - model_dd
            l_grid = np.linspace(-0.05, 0.05, 700)
            image_di = dirty_image_1d(u, residual_di, l_grid)
            image_dd = dirty_image_1d(u, residual_dd, l_grid)
            """
        ),
        md(
            r"""
            ### 8.4.1 从主波束与指向误差理解方向依赖效应 <a id='cal:sec:p_versus_h'></a>

            下面先看一个最常见的 3GC 来源：主波束加上小的指向偏差。对位于视场中心的源，波束变化往往较弱；但对离轴源，同样大小的指向偏差就会被放大成明显的通量偏差和相位/振幅残差。
            """
        ),
        code(
            """
            l_axis = np.linspace(-0.05, 0.05, 500)
            example_antennas = [0, 2, 4]

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

            for ant in example_antennas:
                axes[0].plot(l_axis, beam_gain(l_axis, static_offsets[ant], sigma), lw=2.0, label=f"antenna {ant}")
            for ll in ls:
                axes[0].axvline(ll, color="black", ls=":", alpha=0.6)
            axes[0].set_xlabel("direction cosine l")
            axes[0].set_ylabel("primary-beam gain")
            axes[0].set_title("Different antennas see different off-axis gains")
            axes[0].legend(loc="upper right")

            for si, ll in enumerate(ls[1:]):
                axes[1].plot(times_h, beam[:, 3, si + 1], lw=2.0, label=f"source at l={ll:+.3f}")
            axes[1].set_xlabel("time [hour]")
            axes[1].set_ylabel("beam gain of antenna 3")
            axes[1].set_title("Off-axis response also varies with time")
            axes[1].legend(loc="upper right")

            plt.tight_layout()
            """
        ),
        md(
            r"""
            一旦不同方向上的增益不再相同，就不可能再用一个全场统一的 $G_p$ 同时解释所有源。这时如果仍强行使用方向无关解，求解器往往会优先照顾最亮的源，而把较弱离轴源附近的残差留在图像里。
            """
        ),
        md(
            r"""
            ### 8.4.2 为什么 2GC 模型无法消除这类残差

            下面比较两种模型：

            - **DI 模型**：假设全场都共享同一组方向无关增益，不考虑主波束差异；
            - **DD 模型**：显式把方向依赖波束项写进每个源的预测可见度。

            对于这里构造的数据，DD 模型是“正确模型”，因此 residual 应接近零；而 DI 模型则会在离轴源附近留下结构化残差。
            """
        ),
        code(
            """
            rms_di = np.sqrt(np.mean(np.abs(residual_di) ** 2))
            rms_dd = np.sqrt(np.mean(np.abs(residual_dd) ** 2))

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

            axes[0].plot(l_grid, image_di, color="tab:red", lw=1.7)
            for ll in ls:
                axes[0].axvline(ll, color="black", ls=":", alpha=0.6)
            axes[0].set_ylabel("residual image")
            axes[0].set_title("Residual after a direction-independent model")

            axes[1].plot(l_grid, image_dd, color="tab:blue", lw=1.7)
            for ll in ls:
                axes[1].axvline(ll, color="black", ls=":", alpha=0.6)
            axes[1].set_ylabel("residual image")
            axes[1].set_xlabel("direction cosine l")
            axes[1].set_title("Residual after a direction-dependent model")

            plt.tight_layout()
            print(f"Visibility RMS residual: DI model = {rms_di:.4e}, DD model = {rms_dd:.4e}")
            """
        ),
        md(
            r"""
            这个对比说明的不是“3GC 一定更好”，而是：**当污染本质上是方向依赖时，不升级模型就无法从根本上减少残差。** 此时继续在 2GC 框架里调更复杂的参数，通常只会把问题隐藏，而不会真正解决。
            """
        ),
        md(
            r"""
            ### 8.4.3 3GC 的几条常见路线

            在真实软件中，3GC 大致有下面几类思路：

            - **物理模型法**：把主波束、电离层屏或指向误差参数化后直接写进 RIME，再联合求解；
            - **差分增益 / peeling**：对少数亮离轴源单独建立方向相关增益；
            - **faceting / image-domain methods**：把大视场分块，每块近似共享局部响应；
            - **A-projection / AW-projection**：在成像卷积核里显式传播主波束和宽场效应；
            - **混合策略**：先用物理模型吸收大尺度效应，再对最亮方向做局部修正。

            哪条路线更合适，取决于误差类型、亮源分布、计算预算以及阵列本身的方向依赖复杂度。
            """
        ),
        md(
            r"""
            ### 8.4.4 求解器与计算代价 <a id='cal:sec:stef'></a>

            进入 3GC 后，计算难点不只在于“未知数更多”，更在于未知数与数据之间的耦合结构更复杂。常见的瓶颈包括：

            - 参数数量随方向数、时间和频率网格迅速增长；
            - 模型预测需要更频繁地在图像域与可见度域之间往返；
            - 若主波束或电离层模型本身带有误差，求解器会出现明显退化。

            因此，像 StEFCal 这类高效增益求解器仍然非常重要，但它们通常只是更大 3GC 工作流中的一个模块，而不是完整答案。真正的工程挑战在于：如何把求解器、成像器和物理先验以稳定的方式组合起来。
            """
        ),
        md(
            r"""
            #### 什么时候应该从 2GC 升级到 3GC？

            若你在完成 1GC 和 2GC 后仍持续看到以下症状，就应认真考虑 3GC：

            - 残差随视场位置明显变化，而不是全场均匀；
            - 图像边缘的动态范围远差于视场中心；
            - 离轴亮源周围反复出现条纹、拉伸或负碗状残差；
            - 同一源在不同时间、频率或偏振下表现出位置相关的系统误差。

            这时继续“微调 2GC”往往收益有限。更合理的做法，是回到误差物理本身，判断是否已经进入了方向依赖校准的适用区间。
            """
        ),
    ],
    "8_5_further_reading_and_references.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [8. 校准](8_0_introduction.ipynb)
                * 上一节： [8.4 第三代校准（3GC）](8_4_3gc.ipynb)
                * 下一节： [第 9 章：实践部分](../9_Practical/9_1_visualisation-inspection.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 8.5 延伸阅读与参考文献 <a id='cal:sec:refs'></a>

            本节按照“从理论到实践”的顺序整理第 8 章最值得继续深读的材料。阅读时建议始终带着三个问题：

            - 这篇材料讨论的是哪一类误差模型？
            - 它默认了怎样的天空模型与求解条件？
            - 它更适合用来理解原理，还是更适合直接指导数据处理？
            """
        ),
        md(
            r"""
            ### 8.5.1 校准与测量方程基础

            - **Thompson, Moran & Swenson, _Interferometry and Synthesis in Radio Astronomy_**  
              射电干涉领域最经典的系统教材之一。阅读校准章节时，建议同时回顾本书第 4 至第 8 章，把可见度、成像与校准放在统一框架里理解。
            - **Smirnov (2011a), “Revisiting the radio interferometer measurement equation. I. A full-sky Jones formalism”**  
              建立现代 RIME 语言的关键论文之一，适合与第 7 章和第 8 章一起阅读。
            - **Smirnov (2011b), “Revisiting the radio interferometer measurement equation. II. Calibration and direction-dependent effects”**  
              把 RIME 明确推进到校准与方向依赖效应，是理解 3GC 的核心文献。
            """
        ),
        md(
            r"""
            ### 8.5.2 自校准、闭合量与经典成图思路

            - **Pearson & Readhead (1984), “Image Formation by Self-Calibration in Radio Astronomy”**  
              自校准经典文献，适合与本章 8.3 节配套阅读。
            - **Readhead & Wilkinson (1978), “The mapping of compact radio sources from VLBI data”**  
              经典 hybrid mapping 思想的重要来源。
            - **Cornwell & Fomalont, synthesis imaging 相关综述**  
              有助于把闭合量、自校准和成像质量控制联系起来。
            """
        ),
        md(
            r"""
            ### 8.5.3 高效求解器与 3GC

            - **Salvini & Wijnholds (2014), “Fast gain calibration in radio astronomy using alternating direction implicit methods”**  
              StEFCal 论文，是理解大阵列快速增益求解的入门材料。
            - **Mitra et al. (2015), “Incorporation of antenna primary beam patterns in radio-interferometric data reduction to produce wide-field, high-dynamic-range images”**  
              适合把主波束模型与宽场高动态范围成像联系起来。
            - **Tasse 及相关方向依赖校准工作**  
              适合继续追踪 faceting、DDFacet、KillMS 等现代 3GC 路线。
            """
        ),
        md(
            r"""
            ### 8.5.4 与第 9 章实践部分的衔接

            完成本章后，建议继续进入 [第 9 章实践部分](../9_Practical/9_1_visualisation-inspection.ipynb)。实践章中最值得重点练习的，不是“记住命令”，而是把以下对应关系建立起来：

            - `listobs` / `plotms` 对应怎样的数据质量判断；
            - `gaincal` / `bandpass` / `applycal` 分别在校正哪类误差；
            - 自校准何时应该开始、何时应该停止；
            - 图像残差究竟在反映天空模型不足，还是误差模型不足。

            若你希望在进入第 9 章前先做一轮手算和编程训练，建议先完成 [校准习题](8_problem_set.ipynb)。
            """
        ),
        md("***"),
    ],
    "8_x_further_reading_and_references.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [8. 校准](8_0_introduction.ipynb)
                * 上一节： [8.4 第三代校准（3GC）](8_4_3gc.ipynb)
                * 下一节： [第 9 章：实践部分](../9_Practical/9_1_visualisation-inspection.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 8.x 兼容导航页：延伸阅读与参考文献

            旧版目录和外部链接中有不少地方仍然指向 `8_x_further_reading_and_references.ipynb`。为了保持兼容，本页保留该入口；正式的章节收尾页已经统一为 [8.5 延伸阅读与参考文献](8_5_further_reading_and_references.ipynb)。

            如果你是顺着当前新版目录阅读，建议直接使用 [8.5 延伸阅读与参考文献](8_5_further_reading_and_references.ipynb)。该页面已经按“基础测量方程、自校准、3GC 与后续实践”重新组织。
            """
        ),
        md(
            r"""
            ### 本页保留的目的

            - 保持旧站点和旧 notebook 内部链接不失效；
            - 把读者统一引导到新的 `8.5` 参考页；
            - 为后续进一步清理旧导航留下过渡空间。
            """
        ),
        md("***"),
    ],
}


NOTEBOOKS.update(
    {
        "8_1_calibration_least_squares_problem.ipynb": [
            md(
                """
                ***

                * [总目录](../0_Introduction/0_introduction.ipynb)
                * [术语表](../0_Introduction/1_glossary.ipynb)
                * [8. 校准](8_0_introduction.ipynb)
                    * 上一节： [第 8 章：校准](8_0_introduction.ipynb)
                    * 下一节： [8.2 第一代校准（1GC）](8_2_1gc.ipynb)

                ***
                """
            ),
            md("导入标准模块:"),
            code(IMPORT_BLOCK),
            md(
                r"""
                校准之所以能够被写成数值问题，是因为我们可以把“观测到的受损可见度”和“由天空模型预测的理想可见度”联系起来。对单极化、方向无关的最简情形，

                $$
                d_{pq}(t) = g_p(t)\,m_{pq}(t)\,g_q^\ast(t) + n_{pq}(t),
                $$

                其中 $d_{pq}$ 是观测数据，$m_{pq}$ 是由天空模型得到的预测可见度，$g_p$ 是天线增益，$n_{pq}$ 是噪声。若把所有基线与时刻上的残差堆叠起来，那么校准就可以被看成一个非线性最小二乘问题：寻找一组参数，使残差平方和最小。

                这一节不追求最一般的推导，而是把三个关键直觉讲清楚：

                - 为什么最小二乘目标函数会自然出现；
                - 为什么全局相位必须通过参考天线来固定；
                - 为什么天空模型不完整时，求解器会把错误“吸收”进增益。
                """
            ),
            md("***"),
            md(
                r"""
                ## 8.1 作为最小二乘问题的校准 <a id='cal:sec:cal_ls'></a>

                本节围绕一个小型东西向阵列的模拟观测展开。我们会：

                - 先生成一个简化的 $uv$ 覆盖；
                - 用一个多点源模型构造理想可见度；
                - 向数据注入时间变化的复增益；
                - 用一个简化版的 Levenberg-Marquardt（LM）求解器恢复增益；
                - 再观察参考天线选择和模型不完整时会发生什么。
                """
            ),
            code(
                """
                def baseline_pairs(nant):
                    return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


                def ew_uv_tracks(ant_x_m, hour_angle_h, dec_deg, wavelength_m=0.214):
                    pairs = baseline_pairs(len(ant_x_m))
                    hour_angle_rad = np.deg2rad(15.0 * hour_angle_h)
                    dec = np.deg2rad(dec_deg)
                    u = np.zeros((hour_angle_h.size, len(pairs)))
                    v = np.zeros_like(u)

                    for ti, ha in enumerate(hour_angle_rad):
                        for bi, (p, q) in enumerate(pairs):
                            baseline_lambda = (ant_x_m[q] - ant_x_m[p]) / wavelength_m
                            u[ti, bi] = baseline_lambda * np.sin(ha)
                            v[ti, bi] = -baseline_lambda * np.sin(dec) * np.cos(ha)

                    return pairs, u, v


                def sky_model_vis(u, fluxes, ls):
                    vis = np.zeros_like(u, dtype=complex)
                    for flux, ll in zip(fluxes, ls):
                        vis += flux * np.exp(-2j * np.pi * u * ll)
                    return vis


                def gain_track(times_h, nant):
                    ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
                    amp = 1.0 + 0.06 * np.sin(0.9 * times_h[:, None] + 0.6 * ant_phase[None, :])
                    phase = 0.45 * np.sin(1.7 * times_h[:, None] + ant_phase[None, :])
                    return amp * np.exp(1j * phase)


                def apply_gains(model_vis, gains, pairs, noise_std=0.0, rng=None):
                    data = np.zeros_like(model_vis, dtype=complex)
                    for bi, (p, q) in enumerate(pairs):
                        data[:, bi] = gains[:, p] * model_vis[:, bi] * np.conj(gains[:, q])

                    if noise_std > 0.0:
                        if rng is None:
                            rng = np.random.default_rng(0)
                        data += noise_std * (
                            rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
                        )

                    return data


                def pack_params(gains, ref_ant=0):
                    log_amp = np.log(np.maximum(np.abs(gains), 1e-12))
                    phase = np.angle(gains)
                    params = list(log_amp)
                    for ant in range(len(gains)):
                        if ant == ref_ant:
                            continue
                        params.append(phase[ant] - phase[ref_ant])
                    return np.array(params, dtype=float)


                def unpack_params(params, nant, ref_ant=0):
                    log_amp = np.array(params[:nant])
                    phase = np.zeros(nant)
                    idx = nant
                    for ant in range(nant):
                        if ant == ref_ant:
                            continue
                        phase[ant] = params[idx]
                        idx += 1
                    return np.exp(log_amp + 1j * phase)


                def predict_vis(model_vec, pairs, gains):
                    pred = np.zeros_like(model_vec, dtype=complex)
                    for bi, (p, q) in enumerate(pairs):
                        pred[bi] = gains[p] * model_vec[bi] * np.conj(gains[q])
                    return pred


                def residual_vector(params, model_vec, data_vec, pairs, nant, ref_ant=0):
                    gains = unpack_params(params, nant, ref_ant=ref_ant)
                    resid = data_vec - predict_vis(model_vec, pairs, gains)
                    return np.concatenate([resid.real, resid.imag])


                def numerical_jacobian(fun, params, eps=1e-6):
                    base = fun(params)
                    jac = np.zeros((base.size, params.size))
                    for idx in range(params.size):
                        step = eps * max(1.0, abs(params[idx]))
                        shifted = params.copy()
                        shifted[idx] += step
                        jac[:, idx] = (fun(shifted) - base) / step
                    return jac, base


                def lm_solve(data_vec, model_vec, pairs, nant, ref_ant=0, n_iter=12, lambda0=1e-2):
                    params = pack_params(np.ones(nant, dtype=complex), ref_ant=ref_ant)
                    lam = lambda0
                    history = []

                    def fun(x):
                        return residual_vector(x, model_vec, data_vec, pairs, nant, ref_ant=ref_ant)

                    for _ in range(n_iter):
                        jac, resid = numerical_jacobian(fun, params)
                        sse = float(resid @ resid)
                        history.append(np.sqrt(np.mean(resid**2)))

                        jtj = jac.T @ jac
                        grad = jac.T @ resid
                        damping = np.diag(np.diag(jtj)) + 1e-9 * np.eye(jtj.shape[0])
                        delta = np.linalg.solve(jtj + lam * damping, -grad)

                        trial = params + delta
                        resid_trial = fun(trial)
                        if float(resid_trial @ resid_trial) < sse:
                            params = trial
                            lam *= 0.7
                        else:
                            lam *= 2.5

                    history.append(np.sqrt(np.mean(fun(params) ** 2)))
                    return unpack_params(params, nant, ref_ant=ref_ant), np.array(history)


                def solve_track(data, model, pairs, nant, ref_ant=0, n_iter=12):
                    gains = np.zeros((data.shape[0], nant), dtype=complex)
                    histories = []
                    for t in range(data.shape[0]):
                        gains[t], hist = lm_solve(
                            data[t],
                            model[t],
                            pairs,
                            nant,
                            ref_ant=ref_ant,
                            n_iter=n_iter,
                        )
                        histories.append(hist)
                    return gains, np.array(histories)


                def correct_vis(data, gains, pairs):
                    corrected = np.zeros_like(data, dtype=complex)
                    for bi, (p, q) in enumerate(pairs):
                        corrected[:, bi] = data[:, bi] / (
                            gains[:, p] * np.conj(gains[:, q]) + 1e-12
                        )
                    return corrected


                def rms_complex(arr):
                    return np.sqrt(np.mean(np.abs(arr) ** 2))
                """
            ),
            md(
                r"""
                ### 8.1.1 先生成一个简化的 $uv$ 覆盖 <a id='cal:sec:uv'></a>

                为了让重点落在求解器上，这里只考虑一个东西向阵列，并把天体源限制在 $m=0$ 的一维方向余弦轴上。即便如此，随着地球自转，阵列仍会在 $uv$ 平面上扫出一组有代表性的轨迹。
                """
            ),
            code(
                """
                nant = 5
                ant_x_m = np.array([0.0, 36.0, 102.0, 210.0, 348.0])
                times_h = np.linspace(-3.5, 3.5, 16)
                dec_deg = 50.0

                pairs, u, v = ew_uv_tracks(ant_x_m, times_h, dec_deg=dec_deg, wavelength_m=0.214)

                fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

                axes[0].scatter(ant_x_m, np.zeros_like(ant_x_m), s=80, color="tab:blue")
                for idx, x in enumerate(ant_x_m):
                    axes[0].text(x + 4.0, 0.02, f"A{idx}", fontsize=10)
                axes[0].set_xlabel("east-west position [m]")
                axes[0].set_ylabel("north-south position [m]")
                axes[0].set_title("Simplified east-west array")

                for bi in range(u.shape[1]):
                    axes[1].plot(u[:, bi], v[:, bi], lw=1.3)
                    axes[1].plot(-u[:, bi], -v[:, bi], lw=1.0, alpha=0.45)
                axes[1].set_xlabel("u [wavelength]")
                axes[1].set_ylabel("v [wavelength]")
                axes[1].set_title("Earth rotation synthesis tracks")

                plt.tight_layout()
                print(f"天线数 = {nant}，基线数 = {len(pairs)}")
                """
            ),
            md(
                r"""
                在这个几何设置上，我们用三个点源来构造天空模型。由于源都取在 $m=0$ 上，模型可见度只显式依赖于 $u$：

                $$
                m_{pq}(u) = \sum_s I_s \exp(-2\pi i u l_s).
                $$

                这种简化并不会改变校准的核心数学结构，但它能让我们更容易看清参数退化和求解行为。
                """
            ),
            md(
                r"""
                ### 8.1.2 最小二乘目标函数与参考天线 <a id='cal:sec:RIME_un'></a>

                令参数向量 $\boldsymbol{\theta}$ 包含各天线的对数振幅和相位，则最小二乘目标函数可以写成

                $$
                \chi^2(\boldsymbol{\theta}) =
                \sum_{t,p<q}
                \left|d_{pq}(t) - g_p(\boldsymbol{\theta}, t)\,
                m_{pq}(t)\,
                g_q^\ast(\boldsymbol{\theta}, t)\right|^2.
                $$

                对该式有一个非常关键的观察：若把所有增益同时乘上同一个相位因子 $e^{i\phi_0}$，那么每条基线上的 $g_p g_q^\ast$ 都不会改变。因此，$\chi^2$ 对全局相位并不敏感，这就是为什么必须通过参考天线把这一自由度固定下来。
                """
            ),
            code(
                """
                fluxes_true = np.array([1.0, 0.35, 0.18])
                ls_true = np.array([0.0, 0.012, -0.021])
                model_vis_true = sky_model_vis(u, fluxes_true, ls_true)

                true_gains = gain_track(times_h, nant)
                data_vis = apply_gains(model_vis_true, true_gains, pairs, noise_std=0.02, rng=RNG)

                t0 = 8
                global_phases = np.linspace(-np.pi, np.pi, 200)
                chi2_global = []
                for phase in global_phases:
                    shifted = true_gains[t0] * np.exp(1j * phase)
                    resid = data_vis[t0] - predict_vis(model_vis_true[t0], pairs, shifted)
                    chi2_global.append(np.sum(np.abs(resid) ** 2))

                fig, ax = plt.subplots(figsize=(7.2, 4.0))
                ax.plot(global_phases, chi2_global, color="tab:purple", lw=2.0)
                ax.set_xlabel("applied global phase shift [rad]")
                ax.set_ylabel("chi-square")
                ax.set_title("A global phase shift leaves the least-squares cost unchanged")
                plt.tight_layout()
                print(f\"chi-square variation = {max(chi2_global) - min(chi2_global):.3e}\")
                """
            ),
            md(
                r"""
                这条近乎水平的曲线表明：如果不固定参考相位，求解器就会在一个平坦方向上游走。实际处理里，常见做法是把一面信号稳定、性能可靠的天线选作参考天线，只固定它的相位而不强行固定绝对振幅。
                """
            ),
            md(
                r"""
                ### 8.1.3 生成受污染可见度与初始残差 <a id='cal:sec:sim'></a>

                下面把时间变化的复增益施加到理想可见度上，并加入热噪声。这样我们就得到了一个最小但完整的校准输入：模型 $m_{pq}(t)$、观测数据 $d_{pq}(t)$，以及一组待估参数 $g_p(t)$。
                """
            ),
            code(
                """
                baseline = (1, 4)
                baseline_idx = pairs.index(baseline)
                raw_residual_rms = rms_complex(data_vis - model_vis_true)

                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")

                axes[0, 0].plot(times_h, np.abs(true_gains[:, 3]), color="tab:blue", lw=2.0)
                axes[0, 0].set_ylabel("|g_3|")
                axes[0, 0].set_title("True amplitude track of antenna 3")

                axes[1, 0].plot(times_h, np.angle(true_gains[:, 3]), color="tab:orange", lw=2.0)
                axes[1, 0].set_ylabel("phase [rad]")
                axes[1, 0].set_xlabel("hour angle [hour]")
                axes[1, 0].set_title("True phase track of antenna 3")

                axes[0, 1].plot(times_h, np.abs(model_vis_true[:, baseline_idx]), color="black", lw=2.0, label="model")
                axes[0, 1].plot(times_h, np.abs(data_vis[:, baseline_idx]), color="tab:red", lw=1.6, label="data")
                axes[0, 1].set_ylabel("amplitude [Jy]")
                axes[0, 1].set_title(f"Baseline {baseline[0]}-{baseline[1]} amplitude")
                axes[0, 1].legend(loc="upper right")

                axes[1, 1].plot(
                    times_h,
                    np.angle(data_vis[:, baseline_idx] / (model_vis_true[:, baseline_idx] + 1e-12)),
                    color="tab:red",
                    lw=1.6,
                )
                axes[1, 1].axhline(0.0, color="black", ls="--")
                axes[1, 1].set_ylabel("phase offset [rad]")
                axes[1, 1].set_xlabel("hour angle [hour]")
                axes[1, 1].set_title(f"Baseline {baseline[0]}-{baseline[1]} phase offset")

                plt.tight_layout()
                print(f"未校准时，对真模型的可见度 RMS 残差 = {raw_residual_rms:.4f}")
                """
            ),
            md(
                r"""
                从这个时刻起，问题就完全变成了数值优化：我们要找到一组参数，使模型在所有基线上的联合误差最小。由于 $g_p m_{pq} g_q^\ast$ 对参数是非线性的，所以通常需要迭代求解，而不是一次线性代数运算就结束。
                """
            ),
            md(
                r"""
                ### 8.1.4 用一个简化的 LM 求解器恢复增益 <a id='cal:sec:LM'></a>

                这里我们实现一个小型的 LM 风格求解器。它的核心思想是：在当前参数点附近，用雅可比矩阵把残差局部线性化，然后通过一个带阻尼的正规方程给出参数更新。阻尼项较大时，它更像稳定但保守的梯度下降；阻尼项较小时，它更接近标准高斯-牛顿步骤。
                """
            ),
            code(
                """
                solved_gains_ref0, histories_ref0 = solve_track(
                    data_vis,
                    model_vis_true,
                    pairs,
                    nant,
                    ref_ant=0,
                    n_iter=12,
                )
                corrected_ref0 = correct_vis(data_vis, solved_gains_ref0, pairs)

                true_gains_ref0 = true_gains * np.exp(-1j * np.angle(true_gains[:, [0]]))
                solved_residual_rms = rms_complex(corrected_ref0 - model_vis_true)
                gain_amp_rms = np.sqrt(
                    np.mean((np.abs(solved_gains_ref0[:, 3]) - np.abs(true_gains_ref0[:, 3])) ** 2)
                )
                gain_phase_rms = np.sqrt(
                    np.mean((np.angle(solved_gains_ref0[:, 3] / true_gains_ref0[:, 3])) ** 2)
                )

                fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")

                axes[0, 0].plot(histories_ref0.mean(axis=0), marker="o", color="tab:purple", lw=1.8)
                axes[0, 0].set_ylabel("RMS residual")
                axes[0, 0].set_title("Mean LM convergence history")

                axes[1, 0].plot(times_h, np.abs(true_gains_ref0[:, 3]), color="black", lw=2.0, label="true")
                axes[1, 0].plot(times_h, np.abs(solved_gains_ref0[:, 3]), color="tab:blue", lw=1.7, label="solved")
                axes[1, 0].set_ylabel("|g_3|")
                axes[1, 0].set_xlabel("hour angle [hour]")
                axes[1, 0].legend(loc="upper right")

                axes[0, 1].plot(times_h, np.angle(true_gains_ref0[:, 3]), color="black", lw=2.0, label="true")
                axes[0, 1].plot(times_h, np.angle(solved_gains_ref0[:, 3]), color="tab:orange", lw=1.7, label="solved")
                axes[0, 1].set_ylabel("phase [rad]")
                axes[0, 1].set_title("Antenna 3 phase relative to reference antenna")
                axes[0, 1].legend(loc="upper right")

                axes[1, 1].plot(times_h, np.abs(data_vis[:, baseline_idx]), color="tab:red", lw=1.5, label="raw data")
                axes[1, 1].plot(times_h, np.abs(corrected_ref0[:, baseline_idx]), color="tab:blue", lw=1.8, label="corrected")
                axes[1, 1].plot(times_h, np.abs(model_vis_true[:, baseline_idx]), color="black", lw=2.0, alpha=0.7, label="model")
                axes[1, 1].set_ylabel("amplitude [Jy]")
                axes[1, 1].set_xlabel("hour angle [hour]")
                axes[1, 1].set_title(f"Corrected visibility on baseline {baseline[0]}-{baseline[1]}")
                axes[1, 1].legend(loc="upper right")

                plt.tight_layout()
                print(f"校准后，对真模型的可见度 RMS 残差 = {solved_residual_rms:.4f}")
                print(f"天线 3 振幅 RMS 误差 = {gain_amp_rms:.4f}")
                print(f"天线 3 相位 RMS 误差 = {gain_phase_rms:.4f} rad")
                """
            ),
            md(
                r"""
                这里最值得注意的是两件事：

                - 残差在很少几步内就快速下降，说明局部线性化在这个问题上是有效的；
                - 我们比较的是“相对于参考天线相位归一化后的真实增益”和求解结果，而不是原始绝对相位轨迹。

                这再次提醒我们：校准解本身带有规范选择，不能脱离参考定义来解读。
                """
            ),
            md(
                r"""
                ### 8.1.5 更换参考天线后会发生什么 <a id='cal:sec:cor'></a>

                下面把参考天线从 0 号改成 2 号。由于我们固定的是全局相位规范，而不是物理信号本身，因此**增益参数的表达会改变，但改正后的数据应该保持不变**。
                """
            ),
            code(
                """
                solved_gains_ref2, histories_ref2 = solve_track(
                    data_vis,
                    model_vis_true,
                    pairs,
                    nant,
                    ref_ant=2,
                    n_iter=12,
                )
                corrected_ref2 = correct_vis(data_vis, solved_gains_ref2, pairs)
                correction_difference = rms_complex(corrected_ref0 - corrected_ref2)

                fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
                axes[0].plot(times_h, np.angle(solved_gains_ref0[:, 3]), color="tab:blue", lw=1.8, label="ref = antenna 0")
                axes[0].plot(times_h, np.angle(solved_gains_ref2[:, 3]), color="tab:green", lw=1.8, label="ref = antenna 2")
                axes[0].set_xlabel("hour angle [hour]")
                axes[0].set_ylabel("phase [rad]")
                axes[0].set_title("Gain phases depend on the chosen reference")
                axes[0].legend(loc="upper right")

                axes[1].plot(
                    times_h,
                    np.abs(corrected_ref0[:, baseline_idx] - corrected_ref2[:, baseline_idx]),
                    color="tab:red",
                    lw=1.8,
                )
                axes[1].set_xlabel("hour angle [hour]")
                axes[1].set_ylabel("|difference| [Jy]")
                axes[1].set_title("Corrected visibilities remain invariant")
                plt.tight_layout()

                print(f"两种参考天线下，改正后可见度的 RMS 差异 = {correction_difference:.4e}")
                """
            ),
            md(
                r"""
                因此，参考天线的选择影响的是“解的表示方式”，而不是最终应该得到的物理校正结果。真正需要小心的是：若参考天线本身不稳定，那么它会把额外噪声传播到整组相位解里。
                """
            ),
            md(
                r"""
                ### 8.1.6 当天空模型不完整时，求解器会吸收错误

                最后做一个在真实处理里非常重要的实验：故意把第三个较弱源从模型里删掉，再重复一次求解。这样做之后，求解器仍会努力减小残差，但它面对的是一个“先天错误的问题”。这时部分天空结构会被错当成增益变化。
                """
            ),
            code(
                """
                fluxes_incomplete = np.array([1.0, 0.35])
                ls_incomplete = np.array([0.0, 0.012])
                model_vis_incomplete = sky_model_vis(u, fluxes_incomplete, ls_incomplete)

                solved_gains_bad, histories_bad = solve_track(
                    data_vis,
                    model_vis_incomplete,
                    pairs,
                    nant,
                    ref_ant=0,
                    n_iter=12,
                )
                corrected_bad = correct_vis(data_vis, solved_gains_bad, pairs)

                residual_good = rms_complex(corrected_ref0 - model_vis_true)
                residual_bad_true = rms_complex(corrected_bad - model_vis_true)
                residual_bad_model = rms_complex(corrected_bad - model_vis_incomplete)

                fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
                axes[0].plot(histories_ref0.mean(axis=0), marker="o", lw=1.8, color="tab:blue", label="complete model")
                axes[0].plot(histories_bad.mean(axis=0), marker="s", lw=1.8, color="tab:red", label="incomplete model")
                axes[0].set_xlabel("LM iteration")
                axes[0].set_ylabel("mean RMS residual")
                axes[0].set_title("A wrong sky model stalls the solver")
                axes[0].legend(loc="upper right")

                axes[1].bar(
                    ["complete->true", "incomplete->true", "incomplete->its model"],
                    [residual_good, residual_bad_true, residual_bad_model],
                    color=["tab:blue", "tab:red", "tab:orange"],
                )
                axes[1].set_ylabel("visibility RMS residual")
                axes[1].set_title("Model incompleteness leaves structured residuals")
                plt.tight_layout()

                print(f"完整模型校准后，对真模型的 RMS 残差 = {residual_good:.4f}")
                print(f"不完整模型校准后，对真模型的 RMS 残差 = {residual_bad_true:.4f}")
                print(f"不完整模型校准后，对其自身模型的 RMS 残差 = {residual_bad_model:.4f}")
                """
            ),
            md(
                r"""
                这正是最小二乘校准最容易被误用的地方。求解器只会最小化你给它的目标函数，而不会替你判断天空模型是否正确。也就是说：

                - **残差下降**不等于**物理解释正确**；
                - **解看起来平滑**不等于**没有把天空结构吃进增益**；
                - 一旦发现残差无法下降到接近噪声，首先要怀疑模型，而不只是继续调求解器参数。

                完成本节后，再去读 [8.2 第一代校准（1GC）](8_2_1gc.ipynb) 会更容易理解：为什么校准源的已知模型如此重要，以及为什么 1GC 的成功往往建立在“模型足够简单、误差近似方向无关”之上。
                """
            ),
        ],
        "8_problem_set.ipynb": [
            md(
                """
                ***
                <a id='beginning'></a>

                * [总目录](../0_Introduction/0_introduction.ipynb)
                * [术语表](../0_Introduction/1_glossary.ipynb)
                * [8. 校准](8_0_introduction.ipynb)
                    * 建议先阅读： [8.1 作为最小二乘问题的校准](8_1_calibration_least_squares_problem.ipynb)

                ***
                """
            ),
            md("导入标准模块:"),
            code(IMPORT_BLOCK),
            md(
                r"""
                ## 校准习题

                这一份习题本不再只围绕单一编程题，而是把第 8 章的核心概念拆成一组训练任务。建议按顺序完成：

                1. 看懂最小二乘目标函数与参考天线；
                2. 观察模型不完整时残差为何无法继续下降；
                3. 自己尝试实现一种交替增益求解器，并与 LM 风格求解做比较。

                习题的目标不是背命令，而是建立“观测数据、天空模型、求解器和残差”之间的因果关系。
                """
            ),
            code(
                """
                def baseline_pairs(nant):
                    return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


                def ew_uv_tracks(ant_x_m, hour_angle_h, dec_deg, wavelength_m=0.214):
                    pairs = baseline_pairs(len(ant_x_m))
                    hour_angle_rad = np.deg2rad(15.0 * hour_angle_h)
                    dec = np.deg2rad(dec_deg)
                    u = np.zeros((hour_angle_h.size, len(pairs)))
                    v = np.zeros_like(u)

                    for ti, ha in enumerate(hour_angle_rad):
                        for bi, (p, q) in enumerate(pairs):
                            baseline_lambda = (ant_x_m[q] - ant_x_m[p]) / wavelength_m
                            u[ti, bi] = baseline_lambda * np.sin(ha)
                            v[ti, bi] = -baseline_lambda * np.sin(dec) * np.cos(ha)

                    return pairs, u, v


                def sky_model_vis(u, fluxes, ls):
                    vis = np.zeros_like(u, dtype=complex)
                    for flux, ll in zip(fluxes, ls):
                        vis += flux * np.exp(-2j * np.pi * u * ll)
                    return vis


                def gain_track(times_h, nant):
                    ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
                    amp = 1.0 + 0.06 * np.sin(0.9 * times_h[:, None] + 0.6 * ant_phase[None, :])
                    phase = 0.45 * np.sin(1.7 * times_h[:, None] + ant_phase[None, :])
                    return amp * np.exp(1j * phase)


                def apply_gains(model_vis, gains, pairs, noise_std=0.0, rng=None):
                    data = np.zeros_like(model_vis, dtype=complex)
                    for bi, (p, q) in enumerate(pairs):
                        data[:, bi] = gains[:, p] * model_vis[:, bi] * np.conj(gains[:, q])
                    if noise_std > 0.0:
                        if rng is None:
                            rng = np.random.default_rng(0)
                        data += noise_std * (
                            rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
                        )
                    return data


                def pack_params(gains, ref_ant=0):
                    log_amp = np.log(np.maximum(np.abs(gains), 1e-12))
                    phase = np.angle(gains)
                    params = list(log_amp)
                    for ant in range(len(gains)):
                        if ant == ref_ant:
                            continue
                        params.append(phase[ant] - phase[ref_ant])
                    return np.array(params, dtype=float)


                def unpack_params(params, nant, ref_ant=0):
                    log_amp = np.array(params[:nant])
                    phase = np.zeros(nant)
                    idx = nant
                    for ant in range(nant):
                        if ant == ref_ant:
                            continue
                        phase[ant] = params[idx]
                        idx += 1
                    return np.exp(log_amp + 1j * phase)


                def predict_vis(model_vec, pairs, gains):
                    pred = np.zeros_like(model_vec, dtype=complex)
                    for bi, (p, q) in enumerate(pairs):
                        pred[bi] = gains[p] * model_vec[bi] * np.conj(gains[q])
                    return pred


                def residual_vector(params, model_vec, data_vec, pairs, nant, ref_ant=0):
                    gains = unpack_params(params, nant, ref_ant=ref_ant)
                    resid = data_vec - predict_vis(model_vec, pairs, gains)
                    return np.concatenate([resid.real, resid.imag])


                def numerical_jacobian(fun, params, eps=1e-6):
                    base = fun(params)
                    jac = np.zeros((base.size, params.size))
                    for idx in range(params.size):
                        step = eps * max(1.0, abs(params[idx]))
                        shifted = params.copy()
                        shifted[idx] += step
                        jac[:, idx] = (fun(shifted) - base) / step
                    return jac, base


                def lm_solve(data_vec, model_vec, pairs, nant, ref_ant=0, n_iter=12, lambda0=1e-2):
                    params = pack_params(np.ones(nant, dtype=complex), ref_ant=ref_ant)
                    lam = lambda0
                    history = []

                    def fun(x):
                        return residual_vector(x, model_vec, data_vec, pairs, nant, ref_ant=ref_ant)

                    for _ in range(n_iter):
                        jac, resid = numerical_jacobian(fun, params)
                        sse = float(resid @ resid)
                        history.append(np.sqrt(np.mean(resid**2)))

                        jtj = jac.T @ jac
                        grad = jac.T @ resid
                        damping = np.diag(np.diag(jtj)) + 1e-9 * np.eye(jtj.shape[0])
                        delta = np.linalg.solve(jtj + lam * damping, -grad)

                        trial = params + delta
                        resid_trial = fun(trial)
                        if float(resid_trial @ resid_trial) < sse:
                            params = trial
                            lam *= 0.7
                        else:
                            lam *= 2.5

                    history.append(np.sqrt(np.mean(fun(params) ** 2)))
                    return unpack_params(params, nant, ref_ant=ref_ant), np.array(history)


                def solve_track(data, model, pairs, nant, ref_ant=0, n_iter=12):
                    gains = np.zeros((data.shape[0], nant), dtype=complex)
                    histories = []
                    for t in range(data.shape[0]):
                        gains[t], hist = lm_solve(
                            data[t],
                            model[t],
                            pairs,
                            nant,
                            ref_ant=ref_ant,
                            n_iter=n_iter,
                        )
                        histories.append(hist)
                    return gains, np.array(histories)


                def correct_vis(data, gains, pairs):
                    corrected = np.zeros_like(data, dtype=complex)
                    for bi, (p, q) in enumerate(pairs):
                        corrected[:, bi] = data[:, bi] / (
                            gains[:, p] * np.conj(gains[:, q]) + 1e-12
                        )
                    return corrected


                def rms_complex(arr):
                    return np.sqrt(np.mean(np.abs(arr) ** 2))
                """
            ),
            md(
                r"""
                ### 练习前的统一数据集

                下面的设置与 [8.1](8_1_calibration_least_squares_problem.ipynb) 保持一致。先执行这段代码，后续各题都基于同一组合成数据。
                """
            ),
            code(
                """
                nant = 5
                ant_x_m = np.array([0.0, 36.0, 102.0, 210.0, 348.0])
                times_h = np.linspace(-3.5, 3.5, 16)
                pairs, u, v = ew_uv_tracks(ant_x_m, times_h, dec_deg=50.0, wavelength_m=0.214)

                fluxes_true = np.array([1.0, 0.35, 0.18])
                ls_true = np.array([0.0, 0.012, -0.021])
                model_vis_true = sky_model_vis(u, fluxes_true, ls_true)

                true_gains = gain_track(times_h, nant)
                data_vis = apply_gains(model_vis_true, true_gains, pairs, noise_std=0.02, rng=RNG)

                solved_ref0, history_ref0 = solve_track(
                    data_vis,
                    model_vis_true,
                    pairs,
                    nant,
                    ref_ant=0,
                    n_iter=12,
                )
                corrected_ref0 = correct_vis(data_vis, solved_ref0, pairs)
                """
            ),
            md(
                r"""
                ### A. 观察阻尼参数与收敛历史

                请修改 `lm_solve()` 中的 `lambda0` 或 `n_iter`，比较下列问题：

                - 阻尼过小会不会让第一步更新过于激进？
                - 阻尼过大时，收敛会不会变得很慢？
                - 对当前数据集，多少步之后收益开始明显变小？

                建议你至少尝试三组 `lambda0`，并画出平均收敛历史。
                """
            ),
            code(
                """
                # 在这里编写你的实验代码。
                # 建议：循环多组 lambda0，重复 solve_track，然后比较 history 的均值曲线。
                """
            ),
            md(
                r"""
                ### B. 参考天线与规范选择

                请把参考天线从 `0` 改成 `2` 或 `4`，验证下面两句话：

                - 求解得到的相位轨迹会改变；
                - 但改正后的可见度不应发生实质变化。

                你可以量化 `corrected_ref0` 与新结果之间的 RMS 差异。
                """
            ),
            code(
                """
                # 在这里编写你的代码。
                # 提示：调用 solve_track(..., ref_ant=2) 后，再用 correct_vis 比较结果。
                """
            ),
            md(
                r"""
                ### C. 故意删掉一个弱源，观察模型不完整效应

                请把第三个弱源从模型中去掉，再比较：

                - 收敛历史是否明显停在更高的平台；
                - 改正后数据对“真模型”的残差是否明显增加；
                - 求解器是否试图用更平滑的增益去吸收模型错误。
                """
            ),
            code(
                """
                # 在这里编写你的代码。
                # 提示：可以复用 8.1 里的思路，构造一个 incomplete model 再重复求解。
                """
            ),
            md(
                r"""
                ### D. 实现一种交替增益更新

                下面给出一个交替求解器的骨架。请你补全 `TODO` 部分，实现一种逐天线更新的方向无关增益求解器，并与 `lm_solve` 的结果比较。

                建议比较：

                - 收敛速度；
                - 对初始值的敏感程度；
                - 在模型不完整时是否同样会把天空误差吸收入增益。
                """
            ),
            code(
                """
                def solve_alternating(data, model, pairs, nant, ref_ant=0, n_iter=20):
                    gains = np.ones((data.shape[0], nant), dtype=complex)

                    # TODO:
                    # 1. 按时间切片循环；
                    # 2. 对每个天线利用其余天线当前解更新 g_p；
                    # 3. 固定参考天线相位；
                    # 4. 返回 gains。

                    return gains
                """
            ),
            md(
                r"""
                ### E. 写一个校准诊断结论

                请用 5 到 8 句话总结你在本习题中看到的现象。建议至少回答：

                - 为什么最小二乘目标函数需要参考天线；
                - 为什么“收敛了”不代表“模型正确”；
                - 在真实数据处理中，哪些残差迹象会让你怀疑 sky model 不完整。
                """
            ),
            code(
                """
                reference_summary = {
                    "raw_to_true_rms": rms_complex(data_vis - model_vis_true),
                    "lm_corrected_to_true_rms": rms_complex(corrected_ref0 - model_vis_true),
                    "mean_final_history": history_ref0[:, -1].mean(),
                }

                print("供核对的参考量：")
                for key, value in reference_summary.items():
                    print(f"  {key}: {value:.4f}")
                """
            ),
        ],
    }
)


def write_notebook(path: Path, cells):
    notebook = {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in NOTEBOOKS.items():
        write_notebook(CHAPTER_DIR / filename, cells)
        print(f"wrote {CHAPTER_DIR / filename}")


if __name__ == "__main__":
    main()
