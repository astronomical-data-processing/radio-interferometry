import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_DIR = ROOT / "9_Practical"

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

RNG = np.random.default_rng(20260421)
"""


NOTEBOOKS = {
    "9_1_visualisation-inspection.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 下一节： [9.2 基础校准流程](9_2_calibration_workflow.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 实践 01：数据检查、可视化与初步质量控制

            到了第 9 章，我们不再把重点放在单独的理论概念，而是把前面各章的内容串成一条实际的数据处理主线。一个稳健的连续谱处理流程通常至少包含六步：

            1. 先看懂观测结构与元数据；
            2. 再识别明显异常与应当 flag 的数据；
            3. 用校准源建立 bandpass、gain 和 flux 标尺；
            4. 把校准解转移到目标场；
            5. 做基础成像与必要的自校准；
            6. 最后做图像质量评估与科学测量。

            这一节对应其中的前两步。重点不是“记住某个软件按钮”，而是建立一个专业习惯：在开始求解之前，先确认你到底拿到了一份什么样的数据，以及它已经暴露出了哪些问题。
            """
        ),
        md("***"),
        md(
            r"""
            ### 9.1.1 一个可复现的“合成测量集”

            真实的 Measurement Set 往往需要 CASA 或其他软件环境才能完整浏览。为了让 notebook 自身可以直接运行，这里我们构造一个“小型合成测量集”，保留最关键的几类信息：

            - 阵列布局与 $uv$ 覆盖；
            - 多个时间采样与多个频率通道；
            - 一个含有中心亮源和离轴弱源的天空模型；
            - 一段注入了 RFI 的频率区间；
            - 一组由单天线异常导致的坏基线数据。

            这并不是为了取代真实软件，而是为了让你在一个完全可执行的环境里练习“怎么看出问题”。
            """
        ),
        code(
            """
            def baseline_pairs(nant):
                return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


            def uv_tracks(ant_xy_m, hour_angle_h, dec_deg, wavelength_m=0.214):
                pairs = baseline_pairs(len(ant_xy_m))
                hour_angle_rad = np.deg2rad(15.0 * hour_angle_h)
                dec = np.deg2rad(dec_deg)
                u = np.zeros((hour_angle_h.size, len(pairs)))
                v = np.zeros_like(u)

                for ti, ha in enumerate(hour_angle_rad):
                    for bi, (p, q) in enumerate(pairs):
                        dx = (ant_xy_m[q, 0] - ant_xy_m[p, 0]) / wavelength_m
                        dy = (ant_xy_m[q, 1] - ant_xy_m[p, 1]) / wavelength_m
                        u[ti, bi] = dx * np.sin(ha) + dy * np.cos(ha)
                        v[ti, bi] = -dx * np.sin(dec) * np.cos(ha) + dy * np.sin(dec) * np.sin(ha)
                return pairs, u, v


            def point_source_visibilities(u, v, freqs_ghz, sources):
                vis = np.zeros(u.shape + (freqs_ghz.size,), dtype=complex)
                ref_freq = np.median(freqs_ghz)
                for flux0, l_src, m_src, alpha in sources:
                    spectrum = flux0 * (freqs_ghz / ref_freq) ** alpha
                    phase = np.exp(-2j * np.pi * (u[..., None] * l_src + v[..., None] * m_src))
                    vis += phase * spectrum[None, None, :]
                return vis


            def simulate_measurement_set():
                ant_xy_m = np.array(
                    [
                        [0.0, 0.0],
                        [38.0, 18.0],
                        [116.0, -24.0],
                        [208.0, 44.0],
                        [330.0, -36.0],
                        [472.0, 10.0],
                    ]
                )
                times_h = np.linspace(-4.0, 4.0, 72)
                freqs_ghz = np.linspace(1.10, 1.90, 64)
                pairs, u, v = uv_tracks(ant_xy_m, times_h, dec_deg=45.0, wavelength_m=0.214)

                sky = [
                    (1.0, 0.0, 0.0, -0.7),
                    (0.24, 0.035, -0.018, -0.9),
                    (0.12, -0.048, 0.024, -0.5),
                ]
                model = point_source_visibilities(u, v, freqs_ghz, sky)

                ntime = times_h.size
                nchan = freqs_ghz.size
                nant = ant_xy_m.shape[0]

                chan = np.linspace(-1.0, 1.0, nchan)
                bandpass_amp = 1.0 + 0.04 * chan[None, :] * np.linspace(-1.0, 1.0, nant)[:, None]
                bandpass_phase = 0.10 * np.sin(np.pi * chan)[None, :] * np.linspace(0.2, 1.0, nant)[:, None]
                bandpass = bandpass_amp * np.exp(1j * bandpass_phase)

                ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
                time_amp = 1.0 + 0.03 * np.sin(0.8 * times_h[:, None] + ant_phase[None, :])
                time_phase = 0.20 * np.sin(1.5 * times_h[:, None] + 0.6 * ant_phase[None, :])
                time_gain = time_amp * np.exp(1j * time_phase)

                data = np.zeros_like(model, dtype=complex)
                for bi, (p, q) in enumerate(pairs):
                    lhs = time_gain[:, p, None] * bandpass[p][None, :]
                    rhs = np.conj(time_gain[:, q, None] * bandpass[q][None, :])
                    data[:, bi, :] = lhs * model[:, bi, :] * rhs

                data += 0.02 * (
                    RNG.normal(size=data.shape) + 1j * RNG.normal(size=data.shape)
                )

                # Inject a short, strong RFI burst over a few channels.
                data[22:34, :, 12:16] += 1.4 * (
                    RNG.normal(size=(12, len(pairs), 4)) + 1j * RNG.normal(size=(12, len(pairs), 4))
                )

                # Simulate one problematic antenna that loses gain for a subset of scans.
                bad_ant = 4
                bad_scans = slice(45, 55)
                for bi, (p, q) in enumerate(pairs):
                    if p == bad_ant or q == bad_ant:
                        data[bad_scans, bi, :] *= 0.35 * np.exp(1j * 0.6)

                return {
                    "ant_xy_m": ant_xy_m,
                    "times_h": times_h,
                    "freqs_ghz": freqs_ghz,
                    "pairs": pairs,
                    "u": u,
                    "v": v,
                    "model": model,
                    "data": data,
                    "bad_ant": bad_ant,
                }


            ms = simulate_measurement_set()
            ntime, nbase, nchan = ms["data"].shape
            nant = ms["ant_xy_m"].shape[0]
            print(f"天线数 = {nant}，基线数 = {nbase}，积分数 = {ntime}，频率通道数 = {nchan}")
            """
        ),
        md(
            r"""
            ### 9.1.2 像 `listobs` 一样先看懂观测结构

            开始处理前，至少要回答下面这些问题：

            - 一共用了多少面天线、多少条基线？
            - 时间采样和频率覆盖是否满足你的科学目标？
            - $uv$ 覆盖是否足以支撑所需的成像分辨率与保真度？
            - 观测中是否已经出现非常可疑的扫描、天线或通道？

            下面的图模仿的是 `listobs`、`plotants` 和 `plotms` 的早期巡检习惯。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

            axes[0].scatter(ms["ant_xy_m"][:, 0], ms["ant_xy_m"][:, 1], s=85, color="tab:blue")
            for ant, (x, y) in enumerate(ms["ant_xy_m"]):
                axes[0].text(x + 6.0, y + 3.0, f"A{ant}", fontsize=10)
            axes[0].set_xlabel("east [m]")
            axes[0].set_ylabel("north [m]")
            axes[0].set_title("Antenna layout")

            for bi in range(ms["u"].shape[1]):
                axes[1].plot(ms["u"][:, bi], ms["v"][:, bi], lw=1.0, alpha=0.8)
                axes[1].plot(-ms["u"][:, bi], -ms["v"][:, bi], lw=0.8, alpha=0.35)
            axes[1].set_xlabel("u [wavelength]")
            axes[1].set_ylabel("v [wavelength]")
            axes[1].set_title("UV coverage from Earth rotation")

            plt.tight_layout()

            longest_baseline = 0.0
            for p, q in ms["pairs"]:
                dx, dy = ms["ant_xy_m"][q] - ms["ant_xy_m"][p]
                longest_baseline = max(longest_baseline, np.sqrt(dx**2 + dy**2))

            print(f"时间覆盖：{ms['times_h'][0]:.1f} h 到 {ms['times_h'][-1]:.1f} h")
            print(f"频率覆盖：{ms['freqs_ghz'][0]:.2f} GHz 到 {ms['freqs_ghz'][-1]:.2f} GHz")
            print(f"最长物理基线：{longest_baseline:.1f} m")
            """
        ),
        md(
            r"""
            只看阵列几何还不够。真正的问题往往藏在“振幅随时间或频率的异常结构”里。因此下一步通常要看：

            - 按时间展开的振幅与相位；
            - 按频率展开的平均 bandpass 形状；
            - 动态谱或时频图中是否有窄带、宽带或脉冲式干扰。
            """
        ),
        code(
            """
            amp = np.abs(ms["data"])
            good_baseline = ms["pairs"].index((0, 2))
            bad_baseline = next(
                bi for bi, pair in enumerate(ms["pairs"]) if ms["bad_ant"] in pair
            )

            mean_amp_vs_time = amp.mean(axis=2)
            mean_amp_vs_freq = amp.mean(axis=0)
            dynamic_spec = amp[:, bad_baseline, :]

            fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")

            axes[0, 0].plot(ms["times_h"], mean_amp_vs_time[:, good_baseline], color="tab:blue", lw=1.7)
            axes[0, 0].set_ylabel("mean amplitude [Jy]")
            axes[0, 0].set_title(f"Normal baseline {ms['pairs'][good_baseline]}")

            axes[1, 0].plot(ms["times_h"], mean_amp_vs_time[:, bad_baseline], color="tab:red", lw=1.7)
            axes[1, 0].set_xlabel("hour angle [hour]")
            axes[1, 0].set_ylabel("mean amplitude [Jy]")
            axes[1, 0].set_title(f"Suspicious baseline {ms['pairs'][bad_baseline]}")

            axes[0, 1].plot(ms["freqs_ghz"], mean_amp_vs_freq[good_baseline], color="tab:blue", lw=1.6)
            axes[0, 1].plot(ms["freqs_ghz"], mean_amp_vs_freq[bad_baseline], color="tab:red", lw=1.6)
            axes[0, 1].set_ylabel("mean amplitude [Jy]")
            axes[0, 1].set_title("Amplitude vs frequency")

            im = axes[1, 1].imshow(
                dynamic_spec,
                aspect="auto",
                origin="lower",
                extent=[
                    ms["freqs_ghz"][0],
                    ms["freqs_ghz"][-1],
                    ms["times_h"][0],
                    ms["times_h"][-1],
                ],
                cmap="magma",
            )
            axes[1, 1].set_xlabel("frequency [GHz]")
            axes[1, 1].set_ylabel("hour angle [hour]")
            axes[1, 1].set_title("Dynamic spectrum on the suspicious baseline")
            plt.colorbar(im, ax=axes[1, 1], shrink=0.85, label="|V| [Jy]")

            plt.tight_layout()
            """
        ),
        md(
            r"""
            上图已经出现了两个典型信号：

            - 某些通道在一小段时间内突然整体抬高，这是典型的 RFI 候选；
            - 与 4 号天线相关的基线在后半段出现系统性振幅下跌，这更像是单天线链路异常，而不是随机噪声。

            这时就不应该立刻去做 `gaincal`，而应先做一次初步 flagging 与质量统计。
            """
        ),
        md(
            r"""
            ### 9.1.3 做一个简单但有用的 QA 与 flag 建议

            这里用一个非常朴素的规则做演示：

            - 若某个时频像素的振幅远高于本基线的中位数水平，则记为 RFI 候选；
            - 若一条基线在一个较长时间段内显著低于正常水平，则记为链路异常候选。

            真正的软件包会更复杂，但这已经足以说明：**先定位异常的类型，再决定是 flag 还是留给后续校准处理。**
            """
        ),
        code(
            """
            median_per_baseline = np.median(amp, axis=(0, 2))
            mad_per_baseline = np.median(np.abs(amp - median_per_baseline[None, :, None]), axis=(0, 2)) + 1e-6

            rfi_like = amp > (median_per_baseline[None, :, None] + 7.0 * mad_per_baseline[None, :, None])
            low_gain_like = amp < (0.45 * median_per_baseline[None, :, None])
            flag_mask = rfi_like | low_gain_like

            flag_fraction_by_baseline = flag_mask.mean(axis=(0, 2))
            top_baselines = np.argsort(flag_fraction_by_baseline)[::-1][:4]
            channel_flag_fraction = flag_mask.mean(axis=(0, 1))
            strongest_channels = np.argsort(channel_flag_fraction)[::-1][:6]

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
            axes[0].bar(
                np.arange(len(ms["pairs"])),
                flag_fraction_by_baseline,
                color=["tab:red" if idx in top_baselines else "tab:blue" for idx in range(len(ms["pairs"]))],
            )
            axes[0].set_xlabel("baseline index")
            axes[0].set_ylabel("flag fraction")
            axes[0].set_title("Flag fraction by baseline")

            axes[1].plot(ms["freqs_ghz"], channel_flag_fraction, color="tab:purple", lw=2.0)
            axes[1].set_xlabel("frequency [GHz]")
            axes[1].set_ylabel("flag fraction")
            axes[1].set_title("Flag fraction by channel")
            plt.tight_layout()

            print("最可疑的基线：")
            for idx in top_baselines:
                print(f"  baseline {idx:02d} {ms['pairs'][idx]} -> flag fraction {flag_fraction_by_baseline[idx]:.3f}")

            print("最可疑的通道：")
            for idx in strongest_channels:
                print(f"  channel {idx:02d} ({ms['freqs_ghz'][idx]:.3f} GHz) -> flag fraction {channel_flag_fraction[idx]:.3f}")
            """
        ),
        md(
            r"""
            ### 9.1.4 与真实软件流程的对应

            如果把上面的 notebook 练习映射到 CASA 之类的软件环境，这一节最接近下面这些任务：

            - `listobs`：看懂观测结构、场、频率设置和扫描；
            - `plotants`：检查天线布局；
            - `plotms`：检查振幅/相位随时间、频率、基线、$uv$ 距离的变化；
            - `flagdata` / `flagmanager`：先做明显异常数据的标记与管理；
            - `plotms` 再次检查：确认 flag 没有过度，也没有漏掉最明显的问题。

            一个很重要的专业判断是：**并不是所有异常都应该立刻被删掉。** 有些结构属于可校准误差，有些才属于必须 flag 的坏数据。后面在 [9.2 基础校准流程](9_2_calibration_workflow.ipynb) 中，我们会看到这两类问题在处理链里的角色并不相同。
            """
        ),
    ],
    "9_2_calibration_workflow.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.1 数据检查、可视化与初步质量控制](9_1_visualisation-inspection.ipynb)
                * 下一节： [9.3 连续谱基础成像](9_3_continuum_imaging.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.2 基础校准流程：从校准源到目标场

            在一个最典型的连续谱处理工作流里，校准部分通常包含三层任务：

            - 先用亮而平滑的 bandpass calibrator 解决频率响应；
            - 再用相位校准源跟踪随时间变化的复增益；
            - 最后把这些解转移到目标场，并检查改正后的残差是否接近噪声。

            这一节对应第 8 章的实践化版本。重点不只是“能求出一个解”，而是理解为什么 bandpass 和 time gain 往往需要分开处理，以及 `applycal` 之后我们究竟该看什么。
            """
        ),
        md("***"),
        code(
            """
            def point_source_model_cube(flux, nsample, nant):
                model = np.full((nsample, nant, nant), flux + 0.0j, dtype=complex)
                for ant in range(nant):
                    model[:, ant, ant] = 0.0
                return model


            def solve_gains(data, model, n_iter=30, ref_ant=0, phase_only=False):
                nsample, nant, _ = data.shape
                gains = np.ones((nsample, nant), dtype=complex)
                eps = 1e-12

                for sample in range(nsample):
                    gt = np.ones(nant, dtype=complex)
                    for _ in range(n_iter):
                        new = gt.copy()
                        for p in range(nant):
                            mask = np.ones(nant, dtype=bool)
                            mask[p] = False
                            num = np.sum(data[sample, p, mask] * gt[mask] * np.conj(model[sample, p, mask]))
                            den = np.sum(
                                np.abs(gt[mask]) ** 2 * np.abs(model[sample, p, mask]) ** 2
                            ) + eps
                            new[p] = num / den

                        if phase_only:
                            new = new / np.maximum(np.abs(new), eps)

                        ref = new[ref_ant]
                        new = new / (ref / max(np.abs(ref), eps))
                        gt = new
                    gains[sample] = gt

                return gains


            def apply_direction_independent_factors(model, factors, noise_std=0.0, rng=None):
                data = factors[:, :, None] * model * np.conj(factors[:, None, :])
                if noise_std > 0.0:
                    if rng is None:
                        rng = np.random.default_rng(0)
                    data += noise_std * (
                        rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
                    )
                for ant in range(data.shape[1]):
                    data[:, ant, ant] = 0.0
                return data


            def rms_complex(arr):
                return np.sqrt(np.mean(np.abs(arr) ** 2))


            nant = 6
            nchan = 48
            ntime = 44
            chans = np.linspace(-1.0, 1.0, nchan)
            times_h = np.linspace(-3.0, 3.0, ntime)

            bandpass_amp = 1.0 + 0.05 * chans[None, :] * np.linspace(-1.0, 1.0, nant)[:, None]
            bandpass_phase = 0.14 * np.sin(np.pi * chans)[None, :] * np.linspace(0.3, 1.0, nant)[:, None]
            true_bandpass = (bandpass_amp * np.exp(1j * bandpass_phase)).T

            ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
            time_amp = 1.0 + 0.04 * np.sin(0.9 * times_h[:, None] + 0.4 * ant_phase[None, :])
            time_phase = 0.22 * np.sin(1.6 * times_h[:, None] + ant_phase[None, :])
            true_time_gains = time_amp * np.exp(1j * time_phase)

            model_bandpass = point_source_model_cube(flux=8.0, nsample=nchan, nant=nant)
            data_bandpass = apply_direction_independent_factors(
                model_bandpass,
                true_bandpass,
                noise_std=0.015,
                rng=RNG,
            )
            solved_bandpass = solve_gains(data_bandpass, model_bandpass, n_iter=35, ref_ant=0, phase_only=False)

            model_phase = point_source_model_cube(flux=3.0, nsample=ntime, nant=nant)
            data_phase = apply_direction_independent_factors(
                model_phase,
                true_time_gains,
                noise_std=0.018,
                rng=RNG,
            )
            solved_time = solve_gains(data_phase, model_phase, n_iter=35, ref_ant=0, phase_only=False)
            """
        ),
        md(
            r"""
            ### 9.2.1 先解 bandpass，再解时间增益

            对多通道数据来说，把频率响应和时间响应混成一个单一问题通常并不高效。更常见的做法是：

            - 用亮带通源在每个通道上独立求复增益，得到 per-channel bandpass；
            - 再对已经 bandpass-correct 的数据沿频率平均，用相位校准源求时间变化的增益。

            下面先看 bandpass 解，再看时间增益解。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")

            axes[0, 0].plot(chans, np.abs(true_bandpass[:, 4]), color="black", lw=2.0, label="true")
            axes[0, 0].plot(chans, np.abs(solved_bandpass[:, 4]), color="tab:blue", lw=1.6, label="solved")
            axes[0, 0].set_ylabel("|B_4|")
            axes[0, 0].set_title("Bandpass amplitude of antenna 4")
            axes[0, 0].legend(loc="upper right")

            axes[1, 0].plot(chans, np.angle(true_bandpass[:, 4]), color="black", lw=2.0, label="true")
            axes[1, 0].plot(chans, np.angle(solved_bandpass[:, 4]), color="tab:blue", lw=1.6, label="solved")
            axes[1, 0].set_xlabel("normalized channel coordinate")
            axes[1, 0].set_ylabel("phase [rad]")

            axes[0, 1].plot(times_h, np.abs(true_time_gains[:, 2]), color="black", lw=2.0, label="true")
            axes[0, 1].plot(times_h, np.abs(solved_time[:, 2]), color="tab:orange", lw=1.6, label="solved")
            axes[0, 1].set_ylabel("|G_2|")
            axes[0, 1].set_title("Time gain amplitude of antenna 2")
            axes[0, 1].legend(loc="upper right")

            axes[1, 1].plot(times_h, np.angle(true_time_gains[:, 2]), color="black", lw=2.0, label="true")
            axes[1, 1].plot(times_h, np.angle(solved_time[:, 2]), color="tab:orange", lw=1.6, label="solved")
            axes[1, 1].set_xlabel("hour angle [hour]")
            axes[1, 1].set_ylabel("phase [rad]")

            plt.tight_layout()

            bp_amp_rms = np.sqrt(np.mean((np.abs(solved_bandpass[:, 4]) - np.abs(true_bandpass[:, 4])) ** 2))
            time_phase_rms = np.sqrt(np.mean((np.angle(solved_time[:, 2] / true_time_gains[:, 2])) ** 2))
            print(f"天线 4 bandpass 振幅 RMS 误差 = {bp_amp_rms:.4f}")
            print(f"天线 2 时间增益相位 RMS 误差 = {time_phase_rms:.4f} rad")
            """
        ),
        md(
            r"""
            在真实处理里，这一步大致对应 `bandpass` 和 `gaincal`。它们本质上都在求解天线基复增益，只是一个把“样本轴”看作频率，另一个把“样本轴”看作时间。
            """
        ),
        md(
            r"""
            ### 9.2.2 把 bandpass 与 time gain 一起应用到目标场

            有了 bandpass 和时间增益后，我们就可以构造一个最简的 `applycal` 过程。这里的目标场采用一个相位中心点源加一个简单谱指数，只为了把注意力放在“校准前后差别”上。
            """
        ),
        code(
            """
            target_flux = 1.6 * (1.0 + 0.25 * chans)
            target_model = np.zeros((ntime, nchan, nant, nant), dtype=complex)
            for ti in range(ntime):
                for ci, flux in enumerate(target_flux):
                    target_model[ti, ci] = point_source_model_cube(flux=flux, nsample=1, nant=nant)[0]

            target_data = np.zeros_like(target_model, dtype=complex)
            for ti in range(ntime):
                for ci in range(nchan):
                    factors = true_time_gains[ti] * true_bandpass[ci]
                    target_data[ti, ci] = (
                        factors[:, None] * target_model[ti, ci] * np.conj(factors[None, :])
                    )
            target_data += 0.02 * (
                RNG.normal(size=target_data.shape) + 1j * RNG.normal(size=target_data.shape)
            )

            corrected_target = np.zeros_like(target_data, dtype=complex)
            for ti in range(ntime):
                for ci in range(nchan):
                    factors = solved_time[ti] * solved_bandpass[ci]
                    corrected_target[ti, ci] = target_data[ti, ci] / (
                        factors[:, None] * np.conj(factors[None, :]) + 1e-12
                    )

            raw_target_rms = rms_complex(target_data - target_model)
            corrected_target_rms = rms_complex(corrected_target - target_model)
            baseline = (1, 5)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

            axes[0].plot(
                times_h,
                np.abs(target_data[:, 20, baseline[0], baseline[1]]),
                color="tab:red",
                lw=1.5,
                label="raw",
            )
            axes[0].plot(
                times_h,
                np.abs(corrected_target[:, 20, baseline[0], baseline[1]]),
                color="tab:blue",
                lw=1.8,
                label="corrected",
            )
            axes[0].plot(
                times_h,
                np.abs(target_model[:, 20, baseline[0], baseline[1]]),
                color="black",
                lw=2.0,
                alpha=0.7,
                label="model",
            )
            axes[0].set_xlabel("hour angle [hour]")
            axes[0].set_ylabel("amplitude [Jy]")
            axes[0].set_title(f"Target baseline {baseline}")
            axes[0].legend(loc="upper right")

            rms_by_channel_raw = np.sqrt(np.mean(np.abs(target_data - target_model) ** 2, axis=(0, 2, 3)))
            rms_by_channel_cor = np.sqrt(np.mean(np.abs(corrected_target - target_model) ** 2, axis=(0, 2, 3)))
            axes[1].plot(chans, rms_by_channel_raw, color="tab:red", lw=1.5, label="before calibration")
            axes[1].plot(chans, rms_by_channel_cor, color="tab:blue", lw=1.8, label="after calibration")
            axes[1].set_xlabel("normalized channel coordinate")
            axes[1].set_ylabel("visibility RMS residual")
            axes[1].set_title("Residual spectrum on the target field")
            axes[1].legend(loc="upper right")

            plt.tight_layout()
            print(f"目标场校准前 RMS 残差 = {raw_target_rms:.4f}")
            print(f"目标场校准后 RMS 残差 = {corrected_target_rms:.4f}")
            """
        ),
        md(
            r"""
            ### 9.2.3 这一步在真实软件里对应什么？

            一个最常见的连续谱校准主线可以概括为：

            - `setjy`：设置通量校准源模型；
            - `bandpass`：求解频率响应；
            - `gaincal`：求解时间依赖的相位和振幅；
            - `fluxscale`：把相对振幅解绑定到绝对通量标尺；
            - `applycal`：把解应用到目标场；
            - `plotms` / quick image：检查改正后数据是否真的改善。

            这里的演示故意把它浓缩成了最小例子，但逻辑和真实处理链是一致的。完成这一步之后，下一节才有意义去做 dirty image、PSF 和 CLEAN；否则成像器面对的是一个仍然被系统误差污染的可见度集。
            """
        ),
    ],
    "9_3_continuum_imaging.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.2 基础校准流程](9_2_calibration_workflow.ipynb)
                * 下一节： [9.4 自校准实战](9_4_self_calibration.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.3 连续谱基础成像：从 dirty image 到 restored image

            校准之后，最常见的下一步就是成像。实践里我们通常关心四个对象：

            - 真实天空模型；
            - 由采样函数卷积后的 dirty image；
            - 与采样函数对应的 PSF；
            - 去卷积并恢复后的 clean / restored image。

            这一节用一个完全可运行的二维直接傅里叶实验，把这条链路压缩成最小原型。它不能替代 `tclean` 之类的工业级实现，但足以帮助我们把“图像里看到的结构”与“采样和去卷积造成的结构”分开。
            """
        ),
        md("***"),
        code(
            """
            def baseline_pairs(nant):
                return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


            def uv_tracks(ant_xy_m, hour_angle_h, dec_deg, wavelength_m=0.214):
                pairs = baseline_pairs(len(ant_xy_m))
                hour_angle_rad = np.deg2rad(15.0 * hour_angle_h)
                dec = np.deg2rad(dec_deg)
                u = []
                v = []
                for ha in hour_angle_rad:
                    for p, q in pairs:
                        dx = (ant_xy_m[q, 0] - ant_xy_m[p, 0]) / wavelength_m
                        dy = (ant_xy_m[q, 1] - ant_xy_m[p, 1]) / wavelength_m
                        uu = dx * np.sin(ha) + dy * np.cos(ha)
                        vv = -dx * np.sin(dec) * np.cos(ha) + dy * np.sin(dec) * np.sin(ha)
                        u.extend([uu, -uu])
                        v.extend([vv, -vv])
                return np.array(u), np.array(v)


            def make_sky(npix=64, cell=0.005):
                coords = (np.arange(npix) - npix // 2) * cell
                l_grid, m_grid = np.meshgrid(coords, coords)
                sky = np.zeros((npix, npix))
                sources = [
                    (1.0, 0.00, 0.00),
                    (0.26, 0.04, -0.025),
                    (0.18, -0.055, 0.035),
                ]
                for flux, l_src, m_src in sources:
                    idx = np.argmin(np.abs(coords - l_src))
                    idy = np.argmin(np.abs(coords - m_src))
                    sky[idy, idx] += flux
                return coords, l_grid, m_grid, sky


            def vis_from_sky(u, v, l_grid, m_grid, sky):
                l_flat = l_grid.ravel()
                m_flat = m_grid.ravel()
                sky_flat = sky.ravel()
                phase = np.exp(-2j * np.pi * (u[:, None] * l_flat[None, :] + v[:, None] * m_flat[None, :]))
                return phase @ sky_flat


            def direct_image(u, v, vis, l_grid, m_grid, weights=None):
                if weights is None:
                    weights = np.ones_like(u)
                l_flat = l_grid.ravel()
                m_flat = m_grid.ravel()
                phase = np.exp(2j * np.pi * (u[:, None] * l_flat[None, :] + v[:, None] * m_flat[None, :]))
                image = (weights * vis) @ phase / np.sum(weights)
                return np.real(image.reshape(l_grid.shape))


            def hogbom_clean(dirty, psf, gain=0.12, niter=120, threshold=0.03):
                residual = dirty.copy()
                model = np.zeros_like(dirty)
                psf_peak = np.unravel_index(np.argmax(np.abs(psf)), psf.shape)

                for _ in range(niter):
                    peak_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
                    peak_val = residual[peak_idx]
                    if np.abs(peak_val) < threshold:
                        break
                    model[peak_idx] += gain * peak_val
                    shifted = np.roll(np.roll(psf, peak_idx[0] - psf_peak[0], axis=0), peak_idx[1] - psf_peak[1], axis=1)
                    residual -= gain * peak_val * shifted
                return model, residual


            def gaussian_kernel(npix, sigma_pix=1.2):
                coords = np.arange(npix) - npix // 2
                x, y = np.meshgrid(coords, coords)
                kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma_pix**2))
                kernel /= kernel.sum()
                return np.fft.ifftshift(kernel)


            def fft_convolve_same(image, kernel):
                image_ft = np.fft.fft2(image)
                kernel_ft = np.fft.fft2(kernel)
                return np.real(np.fft.ifft2(image_ft * kernel_ft))


            ant_xy_m = np.array(
                [
                    [0.0, 0.0],
                    [42.0, 16.0],
                    [116.0, -26.0],
                    [214.0, 40.0],
                    [342.0, -38.0],
                    [468.0, 12.0],
                ]
            )
            times_h = np.linspace(-4.0, 4.0, 18)
            u, v = uv_tracks(ant_xy_m, times_h, dec_deg=42.0, wavelength_m=0.214)

            coords, l_grid, m_grid, true_sky = make_sky(npix=64, cell=0.005)
            model_vis = vis_from_sky(u, v, l_grid, m_grid, true_sky)
            noisy_vis = model_vis + 0.03 * (
                RNG.normal(size=model_vis.size) + 1j * RNG.normal(size=model_vis.size)
            )

            dirty_image = direct_image(u, v, noisy_vis, l_grid, m_grid)
            psf = direct_image(u, v, np.ones_like(noisy_vis), l_grid, m_grid)
            clean_components, residual = hogbom_clean(dirty_image, psf, gain=0.12, niter=130, threshold=0.03)
            restore_kernel = gaussian_kernel(true_sky.shape[0], sigma_pix=1.25)
            restored = fft_convolve_same(clean_components, restore_kernel) + residual

            def off_source_rms(image):
                mask = np.ones_like(image, dtype=bool)
                for ly, lx in [(32, 32), (27, 40), (39, 21)]:
                    mask[max(0, ly - 4):ly + 5, max(0, lx - 4):lx + 5] = False
                return np.sqrt(np.mean(image[mask] ** 2))


            dirty_rms = off_source_rms(dirty_image)
            restored_rms = off_source_rms(restored)
            dirty_dr = dirty_image.max() / dirty_rms
            restored_dr = restored.max() / restored_rms
            """
        ),
        md(
            r"""
            ### 9.3.1 Dirty image、PSF 与 restored image 的最小原型

            下图把成像的四个核心对象放在一起。请特别注意两点：

            - dirty image 中的条纹和负瓣，并不一定属于天空本身；
            - restored image 的改善来自去卷积与恢复波束，而不是“凭空制造信息”。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 2, figsize=(10, 9))

            images = [
                (true_sky, "True sky"),
                (dirty_image, f"Dirty image (DR={dirty_dr:.1f})"),
                (psf, "Point spread function"),
                (restored, f"Restored image (DR={restored_dr:.1f})"),
            ]

            for ax, (image, title) in zip(axes.ravel(), images):
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=[coords[0], coords[-1], coords[0], coords[-1]],
                    cmap="inferno",
                )
                ax.set_xlabel("l")
                ax.set_ylabel("m")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout()
            print(f"Dirty image 离源 RMS = {dirty_rms:.4f}")
            print(f"Restored image 离源 RMS = {restored_rms:.4f}")
            """
        ),
        md(
            r"""
            这个例子里 restored image 的动态范围明显高于 dirty image，但 residual 依然存在。这正是实践中最常见的状态：成像并不是“按一次按钮就到真相”，而是一个不断平衡采样、权重、mask、迭代次数和校准质量的过程。
            """
        ),
        md(
            r"""
            ### 9.3.2 与 `tclean` 等成像任务的对应

            若把这个 notebook 映射到真实软件，通常需要特别关心以下参数：

            - `cell`：像素大小；
            - `imsize`：图像尺寸；
            - `weighting` 与 `robust`：分辨率和灵敏度权衡；
            - `niter` / `threshold`：去卷积停止条件；
            - `mask`：哪些区域允许 CLEAN 继续挖掘分量；
            - `pbcor`：是否做主波束校正。

            从工作流角度看，只有在校准残差已经足够小的情况下，调这些成像参数才真正有意义。否则，图像里最显著的结构可能仍然是校准错误，而不是天体结构。
            """
        ),
    ],
    "9_4_self_calibration.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.3 连续谱基础成像](9_3_continuum_imaging.ipynb)
                * 下一节： [9.5 图像质量评估与测量](9_5_image_assessment_and_measurement.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.4 自校准实战：什么时候开始，什么时候停

            第 8 章已经讨论过 2GC 的原理。这里的重点则是工作流视角：

            - 初始模型通常不完美，但必须先足够好，才值得开始自校准；
            - 第一轮往往先做 phase-only，确认模型稳定后再考虑 amplitude+phase；
            - 改善幅度如果已经很小，或 residual 明显受模型缺失主导，就应当停止，而不是机械地继续迭代。
            """
        ),
        md("***"),
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
                    data += noise_std * (
                        rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape)
                    )
                for ant in range(data.shape[1]):
                    data[:, ant, ant] = 0.0
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
                            den = np.sum(np.abs(gt[mask]) ** 2 * np.abs(model[t, p, mask]) ** 2) + eps
                            new[p] = num / den
                        if phase_only:
                            new = new / np.maximum(np.abs(new), eps)
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


            ant_x = np.array([0.0, 40.0, 106.0, 194.0, 316.0, 462.0])
            times_h = np.linspace(-3.0, 3.0, 52)
            pairs, u = baseline_u(ant_x, times_h)
            nant = ant_x.size

            flux_true = np.array([1.0, 0.30, 0.16])
            l_true = np.array([0.0, 0.012, -0.024])
            vis_true = sky_vis_1d(u, flux_true, l_true)
            model_true = vec_to_cube(vis_true, pairs, nant)

            ant_phase = np.linspace(0.0, np.pi, nant, endpoint=False)
            amp = 1.0 + 0.03 * np.sin(1.1 * times_h[:, None] + 0.5 * ant_phase[None, :])
            phase = 0.60 * np.sin(2.1 * times_h[:, None] + ant_phase[None, :])
            true_gains = amp * np.exp(1j * phase)
            data = apply_gains(model_true, true_gains, noise_std=0.02, rng=RNG)

            flux_initial = np.array([1.0])
            l_initial = np.array([0.0])
            model_initial = vec_to_cube(sky_vis_1d(u, flux_initial, l_initial), pairs, nant)

            flux_improved = np.array([1.0, 0.28, 0.13])
            l_improved = np.array([0.0, 0.012, -0.024])
            model_improved = vec_to_cube(sky_vis_1d(u, flux_improved, l_improved), pairs, nant)

            gains_phase = solve_gains(data, model_initial, n_iter=32, ref_ant=0, phase_only=True)
            data_phase = data / (gains_phase[:, :, None] * np.conj(gains_phase[:, None, :]) + 1e-12)

            gains_full = solve_gains(data_phase, model_improved, n_iter=32, ref_ant=0, phase_only=False)
            data_full = data_phase / (gains_full[:, :, None] * np.conj(gains_full[:, None, :]) + 1e-12)

            l_grid = np.linspace(-0.045, 0.045, 720)
            image_raw = dirty_image_1d(u, cube_to_vec(data, pairs), l_grid)
            image_phase = dirty_image_1d(u, cube_to_vec(data_phase, pairs), l_grid)
            image_full = dirty_image_1d(u, cube_to_vec(data_full, pairs), l_grid)

            rms_raw = off_source_rms(image_raw, l_grid, l_true)
            rms_phase = off_source_rms(image_phase, l_grid, l_true)
            rms_full = off_source_rms(image_full, l_grid, l_true)
            dr_raw = image_raw.max() / rms_raw
            dr_phase = image_phase.max() / rms_phase
            dr_full = image_full.max() / rms_full
            """
        ),
        md(
            r"""
            ### 9.4.1 phase-only 先于 amplitude+phase

            下面的三幅图代表典型的自校准节奏：

            - 未自校准：数据仍带有明显时间相位误差；
            - phase-only：核心结构显著改善，但模型不足的地方仍留下 residual；
            - amplitude+phase：在模型更完整的前提下进一步压低离源噪声。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

            for ax, image, title, color in [
                (axes[0], image_raw, f"Before selfcal (DR={dr_raw:.1f})", "tab:red"),
                (axes[1], image_phase, f"Phase-only selfcal (DR={dr_phase:.1f})", "tab:orange"),
                (axes[2], image_full, f"Amplitude+phase selfcal (DR={dr_full:.1f})", "tab:blue"),
            ]:
                ax.plot(l_grid, image, color=color, lw=1.6)
                for pos in l_true:
                    ax.axvline(pos, color="black", ls=":", alpha=0.6)
                ax.set_ylabel("dirty image")
                ax.set_title(title)

            axes[2].set_xlabel("direction cosine l")
            plt.tight_layout()

            print(f"离源 RMS：raw = {rms_raw:.4f}, phase-only = {rms_phase:.4f}, amp+phase = {rms_full:.4f}")
            print(f"动态范围：raw = {dr_raw:.1f}, phase-only = {dr_phase:.1f}, amp+phase = {dr_full:.1f}")
            """
        ),
        md(
            r"""
            这个结果非常符合实践经验：第一轮 phase-only 往往带来最明显的结构改善，而 amplitude+phase 是否值得做，取决于：

            - 初始模型是否已经足够稳定；
            - 场内总信噪比是否支持更自由的解；
            - 你的目标是进一步提高动态范围，还是已经进入收益递减区。
            """
        ),
        md(
            r"""
            ### 9.4.2 一个简化但实用的停止准则

            自校准不应该无限循环。若后续迭代满足以下任何一种情况，就应认真考虑停止：

            - 离源 RMS 改善已经很小；
            - 峰值增大但弱源结构不再更可信；
            - 改成更复杂解之后，绝对通量或扩展结构开始漂移；
            - residual 明显是位置相关的，这更像是方向依赖问题，而不是方向无关自校准还能继续解决的问题。
            """
        ),
    ],
    "9_5_image_assessment_and_measurement.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.4 自校准实战](9_4_self_calibration.ipynb)
                * 下一节： [9.6 时间平均、频率平均与展宽](9_6_averaging_and_smearing.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.5 图像质量评估与基本测量

            当我们拿到一幅“看起来不错”的图像后，真正的工作并没有结束。至少还要回答下面这些问题：

            - 背景噪声有多大？是否接近热噪声极限？
            - 峰值与动态范围是多少？
            - 某个源的总通量应该按 Jy/beam 还是 Jy 来理解？
            - 扩展结构的积分测量是否考虑了 beam 面积？

            这一节用一个合成 restored image 练习这些最基本、也最容易被忽视的测量动作。
            """
        ),
        md("***"),
        code(
            """
            from matplotlib.patches import Circle


            def make_beam(npix=80, sigma_pix=2.0):
                coords = np.arange(npix) - npix // 2
                x, y = np.meshgrid(coords, coords)
                beam = np.exp(-(x**2 + y**2) / (2.0 * sigma_pix**2))
                beam /= beam.max()
                return beam


            def fft_convolve_same(image, kernel):
                return np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(np.fft.ifftshift(kernel))))


            def normalized_gaussian(l_grid, m_grid, l0, m0, sigma_l, sigma_m):
                profile = np.exp(
                    -0.5 * (((l_grid - l0) / sigma_l) ** 2 + ((m_grid - m0) / sigma_m) ** 2)
                )
                profile /= profile.sum()
                return profile


            def robust_rms(image, mask=None):
                if mask is None:
                    values = image.ravel()
                else:
                    values = image[mask]
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                return 1.4826 * mad


            npix = 80
            cell = 0.8
            coords = (np.arange(npix) - npix // 2) * cell
            l_grid, m_grid = np.meshgrid(coords, coords)

            sky = np.zeros((npix, npix))
            core_idx = np.argmin(np.abs(coords - 0.0))
            sky[core_idx, core_idx] += 0.85
            sky += 0.42 * normalized_gaussian(l_grid, m_grid, 14.0, -9.0, 3.6, 2.2)
            weak_x = np.argmin(np.abs(coords - 12.0))
            weak_y = np.argmin(np.abs(coords - (-18.0)))
            sky[weak_y, weak_x] += 0.18

            beam = make_beam(npix=npix, sigma_pix=2.1)
            restored = fft_convolve_same(sky, beam)
            restored += 0.004 * RNG.normal(size=restored.shape)

            source_mask = (np.abs(l_grid) < 6.0) & (np.abs(m_grid) < 6.0)
            background_mask = ~(
                ((l_grid - 0.0) ** 2 + (m_grid - 0.0) ** 2 < 7.0**2)
                | ((l_grid - 14.0) ** 2 + (m_grid + 9.0) ** 2 < 9.0**2)
                | ((l_grid + 18.0) ** 2 + (m_grid - 12.0) ** 2 < 6.0**2)
            )

            beam_area_pix = beam.sum()
            noise_rms = robust_rms(restored, mask=background_mask)
            peak = restored.max()
            dynamic_range = peak / noise_rms
            integrated_core = restored[source_mask].sum() / beam_area_pix

            extended_mask = ((l_grid - 14.0) ** 2 + (m_grid + 9.0) ** 2) < 9.0**2
            integrated_extended = restored[extended_mask].sum() / beam_area_pix
            """
        ),
        md(
            r"""
            ### 9.5.1 先做最基本的 QA 量

            这里我们先估计三类量：

            - **noise RMS**：背景噪声量级；
            - **peak brightness**：图像峰值（通常以 Jy/beam 表示）；
            - **dynamic range**：峰值与背景噪声之比。

            然后再进入源测量。这样做的好处是：你会先知道这幅图到底有多可靠，再去解释它的精细结构。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

            im = axes[0].imshow(
                restored,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="inferno",
            )
            axes[0].add_patch(Circle((0.0, 0.0), radius=7.0, fill=False, ec="cyan", lw=1.8))
            axes[0].add_patch(Circle((14.0, -9.0), radius=8.0, fill=False, ec="lime", lw=1.8))
            axes[0].set_xlabel("relative RA pixel coordinate")
            axes[0].set_ylabel("relative Dec pixel coordinate")
            axes[0].set_title("Restored image with measurement apertures")
            plt.colorbar(im, ax=axes[0], shrink=0.82, label="Jy/beam")

            axes[1].hist(restored[background_mask].ravel(), bins=35, color="tab:purple", alpha=0.85)
            axes[1].axvline(noise_rms, color="black", ls="--", lw=1.6, label=f"RMS={noise_rms:.4f}")
            axes[1].axvline(-noise_rms, color="black", ls="--", lw=1.6)
            axes[1].set_xlabel("pixel value [Jy/beam]")
            axes[1].set_ylabel("number of pixels")
            axes[1].set_title("Background histogram")
            axes[1].legend(loc="upper right")

            plt.tight_layout()
            print(f"Background RMS = {noise_rms:.4f} Jy/beam")
            print(f"Peak brightness = {peak:.4f} Jy/beam")
            print(f"Dynamic range = {dynamic_range:.1f}")
            print(f"Beam area = {beam_area_pix:.2f} pixel")
            """
        ),
        md(
            r"""
            ### 9.5.2 把 Jy/beam 正确转换成积分通量

            一幅 restored image 常用的单位是 Jy/beam，因此如果你直接把一片区域内的像素值相加，得到的并不是 Jy，而是“Jy/beam 的像素和”。要得到积分通量，必须除以 beam 面积（以像素计）。
            """
        ),
        code(
            """
            print(f"核心源积分通量（近似） = {integrated_core:.4f} Jy")
            print(f"扩展源积分通量（近似） = {integrated_extended:.4f} Jy")
            """
        ),
        md(
            r"""
            ### 9.5.3 一个最基本的图像 QA 清单

            在进入科学解释之前，至少应完成下面这些检查：

            - 背景是否近似零均值，噪声直方图是否近似对称；
            - peak / RMS 是否达到预期动态范围；
            - 负瓣或条纹 residual 是否主要围绕亮源出现；
            - 对扩展源做积分通量时，是否已经正确考虑 beam 面积；
            - 若后续要做源尺寸或形态测量，是否已经评估波束卷积的影响。
            """
        ),
    ],
    "9_6_averaging_and_smearing.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.5 图像质量评估与测量](9_5_image_assessment_and_measurement.ipynb)
                * 下一节： [9.7 谱线数据处理](9_7_spectral_line_processing.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.6 时间平均、频率平均与展宽

            在真实处理里，为了减小数据量，人们经常会沿时间或频率做平均。但平均并不是免费的：

            - 离相位中心越远的源，条纹旋转越快；
            - 基线越长，条纹旋转越快；
            - 通道越宽、积分时间越长，平均造成的相位涂抹就越严重。

            因此平均会把离轴源变暗、拉宽，最终降低宽场成像的保真度。这一节给出几个最常见的定量趋势。
            """
        ),
        md("***"),
        code(
            """
            c = 299792458.0
            omega_earth = 7.2921159e-5


            def bandwidth_attenuation(theta_rad, baseline_m, channel_width_hz, freq_hz):
                argument = np.pi * baseline_m * theta_rad * channel_width_hz / c
                return np.sinc(argument / np.pi)


            def time_attenuation(theta_rad, baseline_m, tint_s, freq_hz, dec_deg=45.0):
                wavelength = c / freq_hz
                projected_rate = omega_earth * (baseline_m / wavelength) * np.cos(np.deg2rad(dec_deg))
                argument = np.pi * projected_rate * theta_rad * tint_s
                return np.sinc(argument / np.pi)


            field_angle_arcmin = np.linspace(0.0, 45.0, 240)
            theta = np.deg2rad(field_angle_arcmin / 60.0)
            baseline_set = [150.0, 1000.0, 5000.0]
            freq_hz = 1.4e9

            tint_set = [2.0, 10.0, 60.0]
            channel_width_set = [0.5e6, 2.0e6, 8.0e6]
            """
        ),
        md(
            r"""
            ### 9.6.1 基线越长、离轴越远，平均损失越严重
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

            for baseline_m in baseline_set:
                axes[0].plot(
                    field_angle_arcmin,
                    bandwidth_attenuation(theta, baseline_m, channel_width_hz=0.5e6, freq_hz=freq_hz),
                    lw=2.0,
                    label=f"{baseline_m:.0f} m",
                )
                axes[1].plot(
                    field_angle_arcmin,
                    time_attenuation(theta, baseline_m, tint_s=8.0, freq_hz=freq_hz),
                    lw=2.0,
                    label=f"{baseline_m:.0f} m",
                )

            axes[0].set_xlabel("field radius [arcmin]")
            axes[0].set_ylabel("amplitude attenuation")
            axes[0].set_title("Bandwidth smearing")
            axes[0].legend(loc="upper right")

            axes[1].set_xlabel("field radius [arcmin]")
            axes[1].set_ylabel("amplitude attenuation")
            axes[1].set_title("Time smearing")
            axes[1].legend(loc="upper right")

            plt.tight_layout()
            """
        ),
        md(
            r"""
            这两个趋势都说明一个共同事实：**宽场成像最怕把长基线数据平均得过粗。** 因为长基线本来就对离轴源最敏感，平均以后会最快丢失保真度。
            """
        ),
        md(
            r"""
            ### 9.6.2 通道宽度与积分时间的实际权衡
            """
        ),
        code(
            """
            test_radius = np.deg2rad(20.0 / 60.0)
            baseline_m = 5000.0

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

            for dnu in channel_width_set:
                att = bandwidth_attenuation(theta, baseline_m, channel_width_hz=dnu, freq_hz=freq_hz)
                axes[0].plot(field_angle_arcmin, att, lw=2.0, label=f"{dnu/1e6:.3f} MHz")

            for tint in tint_set:
                att = time_attenuation(theta, baseline_m, tint_s=tint, freq_hz=freq_hz)
                axes[1].plot(field_angle_arcmin, att, lw=2.0, label=f"{tint:.0f} s")

            axes[0].set_xlabel("field radius [arcmin]")
            axes[0].set_ylabel("amplitude attenuation")
            axes[0].set_title("Effect of channel width")
            axes[0].legend(loc="upper right")

            axes[1].set_xlabel("field radius [arcmin]")
            axes[1].set_ylabel("amplitude attenuation")
            axes[1].set_title("Effect of integration time")
            axes[1].legend(loc="upper right")

            plt.tight_layout()

            print("在 20 arcmin 视场半径、5000 m 基线上的衰减：")
            for dnu in channel_width_set:
                print(f"  Δν = {dnu/1e6:.3f} MHz -> {bandwidth_attenuation(test_radius, baseline_m, dnu, freq_hz):.3f}")
            for tint in tint_set:
                print(f"  Δt = {tint:.0f} s -> {time_attenuation(test_radius, baseline_m, tint, freq_hz):.3f}")
            """
        ),
        md(
            r"""
            ### 9.6.3 一个简单的“可接受视场”估算

            若把“幅度衰减不超过 5%”作为一个经验阈值，那么给定基线长度、通道宽度和积分时间后，可以倒推出允许的最大视场半径。这个估算虽然粗糙，但在做 `split` 或 `mstransform` 平均前很有用。
            """
        ),
        code(
            """
            def max_radius_for_threshold(func, baseline_m, threshold, **kwargs):
                radii = np.deg2rad(np.linspace(0.0, 60.0, 2000) / 60.0)
                attenuation = func(radii, baseline_m, **kwargs)
                valid = np.where(attenuation >= threshold)[0]
                if valid.size == 0:
                    return 0.0
                return np.rad2deg(radii[valid[-1]]) * 60.0


            radius_bw = max_radius_for_threshold(
                bandwidth_attenuation,
                baseline_m=5000.0,
                threshold=0.95,
                channel_width_hz=2.0e6,
                freq_hz=freq_hz,
            )
            radius_time = max_radius_for_threshold(
                time_attenuation,
                baseline_m=5000.0,
                threshold=0.95,
                tint_s=24.0,
                freq_hz=freq_hz,
            )

            print(f"带宽展宽 5% 阈值对应的最大视场半径约为 {radius_bw:.1f} arcmin")
            print(f"时间展宽 5% 阈值对应的最大视场半径约为 {radius_time:.1f} arcmin")
            """
        ),
        md(
            r"""
            这也是为什么宽场、高动态范围数据并不适合“为了省空间而随手平均”。一旦平均过头，后面再怎么校准和 CLEAN，都不能把已经在相关器或预处理里损失掉的离轴信息完全找回来。
            """
        ),
    ],
    "9_7_spectral_line_processing.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.6 时间平均、频率平均与展宽](9_6_averaging_and_smearing.ipynb)
                * 下一节： [9.8 宽带与宽场高级成像](9_8_wideband_and_widefield_imaging.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.7 谱线数据处理：从连续谱扣除到 PV 图和线宽诊断

            谱线实践和连续谱实践最根本的差别在于：我们不再只关心“总带宽里平均有多亮”，而要保留频率轴上的细结构，并把它解释成速度、柱密度和运动学信息。

            一个更接近真实工作的基础流程通常包括：

            - 先识别 line-free channels，并决定连续谱应采用常数还是低阶基线模型；
            - 在频率轴上估计并去除连续谱；
            - 得到 line-only data cube；
            - 浏览 channel map；
            - 用一个代表性 visibility spectrum 解释为什么真实流程更偏向 `uv-domain continuum subtraction`；
            - 用平滑辅助的 3D mask 构造 moment 0 / moment 1 图；
            - 抽取 PV diagram；
            - 从积分谱线中估计系统速度、`W20/W50` 和单高斯摘要量；
            - 在必要时升级到双分量近似，并把线通量连回柱密度与总 H I 质量。

            这一节用一个完全自包含的合成 H I 风格数据立方体演示整个过程。它并不替代真实的 `uvcontsub`、`tclean` cube mode、`immoments` 或 `impv`，但会把谱线页从“最小原型”加厚到“更接近专业训练”的层次。
            """
        ),
        md("***"),
        code(
            """
            c_kms = 299792.458
            rest_freq_ghz = 1.42040575


            def normalized_gaussian(x, y, x0, y0, sigma_x, sigma_y):
                profile = np.exp(
                    -0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
                )
                profile /= profile.sum()
                return profile


            def make_beam(npix=56, sigma_pix=1.6):
                coords = np.arange(npix) - npix // 2
                xx, yy = np.meshgrid(coords, coords)
                beam = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
                beam /= beam.max()
                return beam


            def fft_convolve_same(image, kernel):
                image_ft = np.fft.fft2(image)
                kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
                return np.real(np.fft.ifft2(image_ft * kernel_ft))


            def robust_rms(values):
                median = np.median(values)
                return 1.4826 * np.median(np.abs(values - median))


            def fit_linear_baseline(cube, velocity_axis, line_free_mask):
                design = np.column_stack(
                    [np.ones(line_free_mask.sum()), velocity_axis[line_free_mask]]
                )
                data = cube[line_free_mask].reshape(line_free_mask.sum(), -1)
                coeff, _, _, _ = np.linalg.lstsq(design, data, rcond=None)
                baseline = coeff[0][None, :] + velocity_axis[:, None] * coeff[1][None, :]
                return baseline.reshape((velocity_axis.size, cube.shape[1], cube.shape[2]))


            def spectral_smooth(cube):
                padded = np.pad(cube, ((1, 1), (0, 0), (0, 0)), mode="edge")
                return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


            def linewidth_at_fraction(velocity_axis, spectrum, fraction):
                positive = np.clip(spectrum, a_min=0.0, a_max=None)
                peak_idx = int(np.argmax(positive))
                peak = positive[peak_idx]
                level = fraction * peak

                left_candidates = np.where(positive[: peak_idx + 1] < level)[0]
                if left_candidates.size == 0:
                    v_left = velocity_axis[0]
                else:
                    i0 = left_candidates[-1]
                    i1 = min(i0 + 1, peak_idx)
                    if positive[i1] == positive[i0]:
                        v_left = velocity_axis[i1]
                    else:
                        v_left = velocity_axis[i0] + (
                            (level - positive[i0])
                            * (velocity_axis[i1] - velocity_axis[i0])
                            / (positive[i1] - positive[i0])
                        )

                right_candidates = np.where(positive[peak_idx:] < level)[0]
                if right_candidates.size == 0:
                    v_right = velocity_axis[-1]
                else:
                    i1 = peak_idx + right_candidates[0]
                    i0 = max(peak_idx, i1 - 1)
                    if positive[i1] == positive[i0]:
                        v_right = velocity_axis[i0]
                    else:
                        v_right = velocity_axis[i0] + (
                            (level - positive[i0])
                            * (velocity_axis[i1] - velocity_axis[i0])
                            / (positive[i1] - positive[i0])
                        )

                return v_right - v_left, 0.5 * (v_left + v_right), level


            def fit_single_gaussian_grid(velocity_axis, spectrum, center_guess):
                positive = np.clip(spectrum, a_min=0.0, a_max=None)
                best = None

                for center in np.linspace(center_guess - 18.0, center_guess + 18.0, 121):
                    for sigma in np.linspace(4.0, 28.0, 90):
                        basis = np.exp(-0.5 * ((velocity_axis - center) / sigma) ** 2)
                        amplitude = np.dot(positive, basis) / np.dot(basis, basis)
                        amplitude = max(amplitude, 0.0)
                        model = amplitude * basis
                        chi2 = np.mean((positive - model) ** 2)
                        if best is None or chi2 < best[0]:
                            best = (chi2, amplitude, center, sigma, model)

                _, _, center_best, sigma_best, _ = best
                for center in np.linspace(center_best - 4.0, center_best + 4.0, 121):
                    for sigma in np.linspace(max(2.5, sigma_best - 4.0), sigma_best + 4.0, 121):
                        basis = np.exp(-0.5 * ((velocity_axis - center) / sigma) ** 2)
                        amplitude = np.dot(positive, basis) / np.dot(basis, basis)
                        amplitude = max(amplitude, 0.0)
                        model = amplitude * basis
                        chi2 = np.mean((positive - model) ** 2)
                        if chi2 < best[0]:
                            best = (chi2, amplitude, center, sigma, model)

                _, amplitude, center, sigma, model = best
                return amplitude, center, sigma, model


            npix = 56
            cell_arcsec = 1.5
            coords = (np.arange(npix) - npix // 2) * cell_arcsec
            x_grid, y_grid = np.meshgrid(coords, coords)

            vel_kms = np.linspace(-90.0, 90.0, 49)
            freqs_ghz = rest_freq_ghz * (1.0 - vel_kms / c_kms)
            dv = np.abs(vel_kms[1] - vel_kms[0])

            continuum_map = np.zeros((npix, npix))
            center = np.argmin(np.abs(coords - 0.0))
            continuum_map[center, center] += 0.36
            continuum_map += 0.14 * normalized_gaussian(x_grid, y_grid, 16.0, -12.0, 3.0, 2.5)
            continuum_cube = continuum_map[None, :, :] * (1.0 + 0.0013 * vel_kms[:, None, None])

            disk_surface = 0.95 * normalized_gaussian(x_grid, y_grid, 3.0, -2.0, 8.0, 5.0)
            cloud_surface = 0.22 * normalized_gaussian(x_grid, y_grid, -11.0, 9.0, 3.8, 2.8)
            velocity_field = 8.0 + 1.8 * x_grid
            cloud_velocity = 28.0 + 0.6 * x_grid
            sigma_v = 7.0

            line_cube = np.zeros((vel_kms.size, npix, npix))
            for ci, vel in enumerate(vel_kms):
                disk_component = disk_surface * np.exp(
                    -0.5 * ((vel - velocity_field) / sigma_v) ** 2
                )
                cloud_component = cloud_surface * np.exp(
                    -0.5 * ((vel - cloud_velocity) / 4.5) ** 2
                )
                line_cube[ci] = disk_component + cloud_component

            beam = make_beam(npix=npix, sigma_pix=1.7)
            beam_area_pix = beam.sum()

            restored_cube = np.zeros_like(line_cube)
            for ci in range(vel_kms.size):
                restored_cube[ci] = fft_convolve_same(continuum_cube[ci] + line_cube[ci], beam)

            restored_cube += 0.0035 * RNG.normal(size=restored_cube.shape)

            line_free_conservative = np.abs(vel_kms) > 48.0
            line_free_loose = np.abs(vel_kms) > 24.0
            continuum_mean_loose = restored_cube[line_free_loose].mean(axis=0)
            continuum_mean_conservative = restored_cube[line_free_conservative].mean(axis=0)
            continuum_linear = fit_linear_baseline(
                restored_cube, vel_kms, line_free_conservative
            )

            cube_sub_loose = restored_cube - continuum_mean_loose[None, :, :]
            cube_sub_mean = restored_cube - continuum_mean_conservative[None, :, :]
            cube_sub = restored_cube - continuum_linear

            noise_rms = robust_rms(cube_sub[line_free_conservative])
            source_aperture = (
                (x_grid - 2.0) ** 2 / 18.0**2 + (y_grid + 1.0) ** 2 / 13.0**2
            ) < 1.0
            background_aperture = ((x_grid + 20.0) ** 2 + (y_grid - 18.0) ** 2) < 7.5**2
            spectrum_raw = restored_cube[:, source_aperture].sum(axis=1) / beam_area_pix
            spectrum_loose = cube_sub_loose[:, source_aperture].sum(axis=1) / beam_area_pix
            spectrum_mean = cube_sub_mean[:, source_aperture].sum(axis=1) / beam_area_pix
            spectrum_line = cube_sub[:, source_aperture].sum(axis=1) / beam_area_pix
            """
        ),
        md(
            r"""
            ### 9.7.1 先定义 line-free channels，再决定基线模型

            谱线处理里最容易犯的两个错误是：

            - 没有先想清楚哪些通道可以视为“只有连续谱，没有谱线”；
            - 明明频谱基线存在斜率，却仍然只用常数平均去扣除连续谱。

            下面先看积分谱，再比较三种连续谱扣除策略：

            - 过于激进的 line-free 选择加上均值法；
            - 保守 line-free 选择加上均值法；
            - 保守 line-free 选择加上线性基线拟合。

            真实处理里更常见的是在可见度域做 `uvcontsub`，并允许低阶多项式拟合。这里的目的，是让你看清楚“通道选择”和“基线模型”是两个独立判断。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

            axes[0].plot(vel_kms, spectrum_raw, color="tab:purple", lw=2.0, label="raw spectrum")
            axes[0].axvspan(-48.0, 48.0, color="tab:orange", alpha=0.16, label="conservative line window")
            axes[0].axvspan(-48.0, -24.0, color="tab:red", alpha=0.12, label="risky wing channels if chosen as line-free")
            axes[0].axvspan(24.0, 48.0, color="tab:red", alpha=0.12)
            axes[0].set_xlabel("velocity [km/s]")
            axes[0].set_ylabel("integrated flux density [Jy]")
            axes[0].set_title("Inspect the global spectrum before subtraction")
            axes[0].legend(loc="upper right")

            axes[1].plot(vel_kms, spectrum_raw, color="tab:gray", lw=1.4, label="raw")
            axes[1].plot(vel_kms, spectrum_loose, color="tab:red", lw=1.5, label="mean subtraction, loose line-free")
            axes[1].plot(vel_kms, spectrum_mean, color="tab:orange", lw=1.6, label="mean subtraction, conservative line-free")
            axes[1].plot(vel_kms, spectrum_line, color="tab:blue", lw=2.0, label="linear baseline, conservative line-free")
            axes[1].axhline(0.0, color="black", ls="--")
            axes[1].set_xlabel("velocity [km/s]")
            axes[1].set_ylabel("integrated flux density [Jy]")
            axes[1].set_title("Line-free choice and baseline model both matter")
            axes[1].legend(loc="upper right")

            plt.tight_layout()
            print(f"保守 line-free 通道数 = {line_free_conservative.sum()} / {vel_kms.size}")
            print(f"激进 line-free 通道数 = {line_free_loose.sum()} / {vel_kms.size}")
            print(
                "线外中位残差 [Jy]："
                f" 激进均值法 = {np.median(spectrum_loose[line_free_conservative]):+.4f},"
                f" 保守均值法 = {np.median(spectrum_mean[line_free_conservative]):+.4f},"
                f" 线性基线 = {np.median(spectrum_line[line_free_conservative]):+.4f}"
            )
            print(f"continuum-subtracted cube 的噪声 RMS ≈ {noise_rms:.4f} Jy/beam")
            """
        ),
        md(
            r"""
            从这个对比可以看到两件事：

            - 若把线翼误划进 line-free channels，谱线本身会被过度扣除；
            - 即使通道选得对，若基线存在频谱斜率，常数均值法仍会留下系统残差。

            下面的后续步骤都采用“保守 line-free + 线性基线模型”得到的 `cube_sub`。这一步之后，数据结构才真正变成了可用于谱线分析的 line-only data cube。
            """
        ),
        md(
            r"""
            ### 9.7.2 一个紧凑的 `uv-domain continuum subtraction` 概念实验

            真实工作里更偏向在可见度域做连续谱扣除，不只是因为软件历史，而是因为每条基线、每个相关乘积都可以在成像之前独立处理频谱基线，从而避免把 **beam、deconvolution 和频谱拟合误差** 混在一起。

            下面用一个代表性的复可见度频谱做一个最小实验：分别对实部和虚部在线外通道上拟合线性基线，再看扣除后的残差。这和真实 `uvcontsub` 当然还不是一回事，但思路是一致的。
            """
        ),
        code(
            """
            vis_amp_cont = 1.45 + 0.0010 * vel_kms
            vis_phase_cont = 0.22 + 0.0009 * vel_kms
            vis_cont = vis_amp_cont * np.exp(1j * vis_phase_cont)

            vis_line = 0.42 * np.exp(-0.5 * ((vel_kms - 12.0) / 8.0) ** 2) * np.exp(
                1j * (0.75 + 0.009 * vel_kms)
            )
            vis_line += 0.12 * np.exp(-0.5 * ((vel_kms - 33.0) / 4.2) ** 2) * np.exp(
                1j * (-0.35 + 0.004 * vel_kms)
            )
            vis_raw = vis_cont + vis_line + 0.015 * (
                RNG.normal(size=vel_kms.size) + 1j * RNG.normal(size=vel_kms.size)
            )

            coeff_real = np.polyfit(
                vel_kms[line_free_conservative], vis_raw.real[line_free_conservative], 1
            )
            coeff_imag = np.polyfit(
                vel_kms[line_free_conservative], vis_raw.imag[line_free_conservative], 1
            )
            vis_model = np.polyval(coeff_real, vel_kms) + 1j * np.polyval(coeff_imag, vel_kms)
            vis_resid = vis_raw - vis_model

            fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

            axes[0].plot(vel_kms, vis_raw.real, color="tab:blue", lw=1.8, label="raw real")
            axes[0].plot(vel_kms, vis_model.real, color="black", ls="--", lw=1.2, label="fitted baseline")
            axes[0].set_xlabel("velocity [km/s]")
            axes[0].set_ylabel("real visibility [Jy]")
            axes[0].set_title("Real part")
            axes[0].legend(loc="upper right")

            axes[1].plot(vel_kms, vis_raw.imag, color="tab:purple", lw=1.8, label="raw imag")
            axes[1].plot(vel_kms, vis_model.imag, color="black", ls="--", lw=1.2, label="fitted baseline")
            axes[1].set_xlabel("velocity [km/s]")
            axes[1].set_ylabel("imag visibility [Jy]")
            axes[1].set_title("Imaginary part")
            axes[1].legend(loc="upper right")

            axes[2].plot(vel_kms, np.abs(vis_raw), color="tab:gray", lw=1.5, label="raw amplitude")
            axes[2].plot(vel_kms, np.abs(vis_resid), color="tab:red", lw=1.8, label="residual amplitude")
            axes[2].axhline(
                np.median(np.abs(vis_resid[line_free_conservative])),
                color="black",
                ls="--",
                lw=1.0,
                label="median line-free residual",
            )
            axes[2].set_xlabel("velocity [km/s]")
            axes[2].set_ylabel("visibility amplitude [Jy]")
            axes[2].set_title("Amplitude after complex baseline subtraction")
            axes[2].legend(loc="upper right")

            plt.tight_layout()
            print(
                "可见度域线外残差振幅中位数 ≈ "
                f"{np.median(np.abs(vis_resid[line_free_conservative])):.4f} Jy"
            )
            """
        ),
        md(
            r"""
            这个练习想强调的不是“图像域做法一定错”，而是：**在真实数据处理中，连续谱扣除最好尽量发生在成像之前。** 这样做更容易把频谱基线问题和成像/卷积问题分离开。
            """
        ),
        md(
            r"""
            ### 9.7.3 浏览 channel map：谱线不是一张图，而是一组随速度变化的切片

            连续谱图像只有二维，而谱线数据的基本对象是三维立方体。最直接的入口就是 channel map：观察不同速度通道上，发射在空间上如何移动、增强或减弱。
            """
        ),
        code(
            """
            channel_targets = [-35.0, -5.0, 20.0, 45.0]
            channel_indices = [np.argmin(np.abs(vel_kms - target)) for target in channel_targets]

            fig, axes = plt.subplots(2, 2, figsize=(10, 9))
            for ax, idx in zip(axes.ravel(), channel_indices):
                im = ax.imshow(
                    cube_sub[idx],
                    origin="lower",
                    extent=[coords[0], coords[-1], coords[0], coords[-1]],
                    cmap="inferno",
                    vmin=-2.0 * noise_rms,
                    vmax=np.max(cube_sub[channel_indices]) * 0.95,
                )
                ax.set_title(f"Channel map at {vel_kms[idx]:+.1f} km/s")
                ax.set_xlabel("RA offset [arcsec]")
                ax.set_ylabel("Dec offset [arcsec]")
                plt.colorbar(im, ax=ax, shrink=0.82, label="Jy/beam")

            plt.tight_layout()
            """
        ),
        md(
            r"""
            这些切片能直接回答一个很重要的问题：发射在不同速度通道上是否沿空间方向系统移动。若答案是肯定的，那么 moment 1 图和 PV 图通常都会呈现出清晰的速度梯度。
            """
        ),
        md(
            r"""
            ### 9.7.4 用平滑辅助的 3D mask 构造 moment 0 与 moment 1 图

            若不加筛选就直接对整立方体积分，噪声会严重污染矩图。因此实践里一般会先构造一个 mask，只保留显著探测到发射的体素。

            这里采用一个更接近真实工作的简化版思路：

            - 先对 cube 做空间与频率平滑；
            - 在平滑 cube 上找 2.2-sigma 的种子；
            - 要求至少出现在相邻通道中，避免单通道噪声峰；
            - 再回到原始 cube，用较低阈值扩展成最终 mask。

            这已经是一个简化版的 `moment masking` 思想，比“直接 3-sigma 硬阈值”更稳健，也更接近专业工作流。
            """
        ),
        code(
            """
            smoothing_beam = make_beam(npix=npix, sigma_pix=2.6)
            smooth_cube = np.zeros_like(cube_sub)
            for ci in range(vel_kms.size):
                smooth_cube[ci] = fft_convolve_same(cube_sub[ci], smoothing_beam)
            smooth_cube = spectral_smooth(smooth_cube)

            smoothed_rms = robust_rms(smooth_cube[line_free_conservative])
            seed_mask = smooth_cube > (2.2 * smoothed_rms)
            neighbor_count = seed_mask.astype(int)
            neighbor_count[1:] += seed_mask[:-1]
            neighbor_count[:-1] += seed_mask[1:]
            spectral_support = neighbor_count >= 2
            mask = spectral_support & (cube_sub > (1.2 * noise_rms))

            positive_cube = np.clip(cube_sub, a_min=0.0, a_max=None)
            masked_positive_cube = np.where(mask, positive_cube, 0.0)
            moment0_raw = positive_cube.sum(axis=0) * dv
            moment0 = masked_positive_cube.sum(axis=0) * dv
            weight_sum = masked_positive_cube.sum(axis=0)
            weighted_velocity = (masked_positive_cube * vel_kms[:, None, None]).sum(axis=0)
            moment1 = np.full_like(weight_sum, np.nan, dtype=float)
            np.divide(weighted_velocity, weight_sum, out=moment1, where=weight_sum > 0.0)
            mask_coverage = mask.sum(axis=0)

            raw_background_bias = moment0_raw[background_aperture].mean()
            masked_background_bias = moment0[background_aperture].mean()

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))

            im_mask = axes[0, 0].imshow(
                mask_coverage,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="viridis",
            )
            axes[0, 0].set_title("Mask coverage: detected channels per pixel")
            axes[0, 0].set_xlabel("RA offset [arcsec]")
            axes[0, 0].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im_mask, ax=axes[0, 0], shrink=0.82, label="channels")

            im0_raw = axes[0, 1].imshow(
                moment0_raw,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="magma",
            )
            axes[0, 1].set_title("Moment 0 without masking")
            axes[0, 1].set_xlabel("RA offset [arcsec]")
            axes[0, 1].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im0_raw, ax=axes[0, 1], shrink=0.82, label="Jy km/s / beam")

            im0 = axes[1, 0].imshow(
                moment0,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="magma",
            )
            axes[1, 0].set_title("Moment 0 with smoothed 3D mask")
            axes[1, 0].set_xlabel("RA offset [arcsec]")
            axes[1, 0].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im0, ax=axes[1, 0], shrink=0.82, label="Jy km/s / beam")

            im1 = axes[1, 1].imshow(
                moment1,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="coolwarm",
                vmin=np.nanpercentile(moment1, 5),
                vmax=np.nanpercentile(moment1, 95),
            )
            axes[1, 1].set_title("Moment 1 from masked cube")
            axes[1, 1].set_xlabel("RA offset [arcsec]")
            axes[1, 1].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im1, ax=axes[1, 1], shrink=0.82, label="km/s")

            plt.tight_layout()
            print(f"平滑后 cube 的 RMS ≈ {smoothed_rms:.4f} Jy/beam")
            print(f"进入最终 mask 的体素数 = {mask.sum()}")
            print(
                "背景区域 moment0 平均偏差 [Jy km/s/beam]："
                f" 未掩膜 = {raw_background_bias:+.4f},"
                f" 掩膜后 = {masked_background_bias:+.4f}"
            )
            """
        ),
        md(
            r"""
            这里最值得注意的并不是“我们画出了两张矩图”，而是：**构造掩膜本身已经是谱线分析的一部分。** 未掩膜的 moment 0 图会在空白区域积累正噪声偏差，而平滑辅助的 3D mask 能明显压低这种偏差。
            """
        ),
        md(
            r"""
            ### 9.7.5 沿主轴抽取一个 PV diagram

            对旋转盘、外流、双峰结构或速度梯度而言，仅靠 channel map 和 moment 1 往往还不够。一个非常经典、也非常有解释力的诊断量，就是位置-速度图（PV diagram）。

            这里沿主轴方向抽取一条有有限宽度的切片。若存在系统性的速度梯度，PV 图通常会呈现出倾斜的脊线。
            """
        ),
        code(
            """
            slit_center_dec = -1.5
            slit_halfwidth = 3.0
            row_select = np.abs(coords - slit_center_dec) <= slit_halfwidth
            pv_slice = cube_sub[:, row_select, :].mean(axis=1)
            pv_mask = mask[:, row_select, :].any(axis=1).astype(float)

            fig, ax = plt.subplots(figsize=(9.0, 5.0))
            im = ax.imshow(
                pv_slice,
                origin="lower",
                aspect="auto",
                extent=[coords[0], coords[-1], vel_kms[0], vel_kms[-1]],
                cmap="cividis",
                vmin=-2.0 * noise_rms,
                vmax=np.max(pv_slice) * 0.95,
            )
            ax.contour(
                coords,
                vel_kms,
                pv_mask,
                levels=[0.5],
                colors="white",
                linewidths=1.0,
            )
            ax.axhline(0.0, color="white", ls="--", lw=0.9, alpha=0.7)
            ax.set_xlabel("major-axis offset [arcsec]")
            ax.set_ylabel("velocity [km/s]")
            ax.set_title("PV diagram extracted along the major axis")
            plt.colorbar(im, ax=ax, shrink=0.86, label="Jy/beam")
            plt.tight_layout()
            """
        ),
        md(
            r"""
            和 moment 1 相比，PV 图往往更容易让你判断：速度梯度是不是单调的、有没有离散云团、是否存在和主盘不同的附加运动学成分。
            """
        ),
        md(
            r"""
            ### 9.7.6 做一个更完整的基础谱线测量：`W20/W50` 与单高斯摘要

            下面基于 aperture 内的积分谱线，估计几类最常见的量：

            - **系统速度**：谱线强度加权的平均速度；
            - **等效 FWHM**：由二阶矩换算出的高斯等效线宽；
            - **`W50` / `W20`**：在峰值 50% 和 20% 处测得的轮廓宽度；
            - **单高斯摘要**：用一个单峰高斯给出紧凑的中心和宽度摘要。

            这里特别保留 `W20/W50`，是因为很多真实全局 H I 轮廓并不接近单高斯，强行只报一个“高斯 FWHM”会把重要形状信息抹掉。
            """
        ),
        code(
            """
            positive_spec = np.clip(spectrum_line, a_min=0.0, a_max=None)
            systemic_velocity = np.sum(vel_kms * positive_spec) / np.sum(positive_spec)
            sigma_line = np.sqrt(np.sum(positive_spec * (vel_kms - systemic_velocity) ** 2) / np.sum(positive_spec))
            fwhm_equiv = 2.355 * sigma_line
            w50, v50_center, level50 = linewidth_at_fraction(vel_kms, positive_spec, 0.50)
            w20, v20_center, level20 = linewidth_at_fraction(vel_kms, positive_spec, 0.20)
            gauss_amp, gauss_center, gauss_sigma, gauss_model = fit_single_gaussian_grid(
                vel_kms, positive_spec, center_guess=systemic_velocity
            )
            gauss_fwhm = 2.355 * gauss_sigma
            integrated_flux = np.sum(positive_spec) * dv

            fig, ax = plt.subplots(figsize=(8.0, 4.0))
            ax.plot(vel_kms, spectrum_line, color="tab:blue", lw=2.0)
            ax.plot(vel_kms, gauss_model, color="tab:orange", lw=1.8, label="single-Gaussian summary")
            ax.axhline(level50, color="tab:green", ls="--", lw=1.0, label=f"W50 = {w50:.1f} km/s")
            ax.axhline(level20, color="tab:red", ls=":", lw=1.0, label=f"W20 = {w20:.1f} km/s")
            ax.axvline(systemic_velocity, color="black", ls="--", label=f"v_sys = {systemic_velocity:.1f} km/s")
            ax.set_xlabel("velocity [km/s]")
            ax.set_ylabel("integrated flux density [Jy]")
            ax.set_title("Continuum-subtracted integrated line spectrum")
            ax.legend(loc="upper right")
            plt.tight_layout()

            print(f"系统速度 ≈ {systemic_velocity:.2f} km/s")
            print(f"高斯等效 FWHM ≈ {fwhm_equiv:.2f} km/s")
            print(f"W50 ≈ {w50:.2f} km/s，中心速度 ≈ {v50_center:.2f} km/s")
            print(f"W20 ≈ {w20:.2f} km/s，中心速度 ≈ {v20_center:.2f} km/s")
            print(f"单高斯摘要中心 ≈ {gauss_center:.2f} km/s，FWHM ≈ {gauss_fwhm:.2f} km/s")
            print(f"积分线通量 ≈ {integrated_flux:.2f} Jy km/s")
            """
        ),
        md(
            r"""
            这一组量的用途并不完全相同：

            - 二阶矩和单高斯摘要适合快速压缩成“一个中心、一个宽度”；
            - `W50/W20` 更接近传统全局轮廓测量，尤其适用于非高斯、双峰或带肩部结构的谱线；
            - 若线型明显复杂，下一步通常不应停在摘要量，而要继续做多分量拟合或运动学建模。
            """
        ),
        md(
            r"""
            ### 9.7.7 当单高斯不够时：双分量近似与残差诊断

            当前这个合成谱线刻意包含了一个主盘分量和一个偏红端弱云团，因此单高斯虽然给出了不错的摘要量，但不一定能把轮廓细节吃干净。

            下面做一个教学用途的两步近似：

            - 先用单高斯给出主摘要；
            - 再对正残差做一次次级高斯拟合；
            - 比较单高斯和双分量近似的残差。

            这不是严格的联合非线性拟合，更不能替代真实科学分析中的完整似然建模；但它足以让学生看到“摘要量”和“模型解释”之间的差别。
            """
        ),
        code(
            """
            secondary_input = np.clip(positive_spec - gauss_model, a_min=0.0, a_max=None)
            sec_amp, sec_center, sec_sigma, sec_component = fit_single_gaussian_grid(
                vel_kms, secondary_input, center_guess=gauss_center + 12.0
            )
            double_model = gauss_model + sec_component

            single_residual = positive_spec - gauss_model
            double_residual = positive_spec - double_model
            single_resid_rms = np.sqrt(np.mean(single_residual**2))
            double_resid_rms = np.sqrt(np.mean(double_residual**2))
            residual_improvement = 100.0 * (single_resid_rms - double_resid_rms) / single_resid_rms
            secondary_fwhm = 2.355 * sec_sigma
            secondary_flux = np.sum(sec_component) * dv

            fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.5), sharex=True)

            axes[0].plot(vel_kms, positive_spec, color="tab:blue", lw=2.0, label="data")
            axes[0].plot(vel_kms, gauss_model, color="tab:orange", lw=1.5, label="single Gaussian")
            axes[0].plot(vel_kms, sec_component, color="tab:green", lw=1.3, label="secondary component")
            axes[0].plot(vel_kms, double_model, color="black", ls="--", lw=1.4, label="two-component approximation")
            axes[0].set_ylabel("integrated flux density [Jy]")
            axes[0].set_title("Single-Gaussian summary versus two-component approximation")
            axes[0].legend(loc="upper right")

            axes[1].plot(vel_kms, single_residual, color="tab:orange", lw=1.6, label="single-Gaussian residual")
            axes[1].plot(vel_kms, double_residual, color="tab:green", lw=1.6, label="two-component residual")
            axes[1].axhline(0.0, color="black", ls="--", lw=1.0)
            axes[1].set_xlabel("velocity [km/s]")
            axes[1].set_ylabel("residual [Jy]")
            axes[1].legend(loc="upper right")

            plt.tight_layout()
            print(f"单高斯残差 RMS ≈ {single_resid_rms:.4f} Jy")
            print(f"双分量近似残差 RMS ≈ {double_resid_rms:.4f} Jy")
            print(f"残差 RMS 改善 ≈ {residual_improvement:.1f}%")
            print(f"次级分量中心速度 ≈ {sec_center:.2f} km/s，FWHM ≈ {secondary_fwhm:.2f} km/s")
            print(f"次级分量积分线通量 ≈ {secondary_flux:.2f} Jy km/s")
            """
        ),
        md(
            r"""
            这个结果应当教会学生一件很重要的事：**单高斯“好用”，不等于单高斯“充分”。** 当残差仍有系统结构时，就要考虑肩部、双峰、云团或更复杂的运动学解释。
            """
        ),
        md(
            r"""
            ### 9.7.8 从积分强度到物理量：柱密度与总 H I 质量的入口

            一旦拿到了 moment 0 和积分线通量，最自然的下一步就是把它们转成物理量。这里给出两个最常见、也最适合作为教学入口的量：

            - **H I 柱密度**：先把 `Jy/beam km/s` 转成 `K km/s`，再用 optically thin 近似得到 $N_{\mathrm{HI}}$；
            - **总 H I 质量**：用距离和积分线通量估计 $M_{\mathrm{HI}}$。

            这里的前提都需要明确说清楚：我们假设发射是 optically thin 的，并且距离已知。为了让数量级更接近真实教材里的常见例子，下面还会**额外假定一个实际成像后的 restoring beam**，而不是直接把这个合成实验里的数值卷积核当成真实望远镜波束。真实科学分析里，还必须继续检查自吸收、距离误差、漏通量和 aperture 选择等系统项。
            """
        ),
        code(
            """
            beam_major_arcsec = 30.0
            beam_minor_arcsec = 24.0
            distance_mpc = 5.2

            jybeam_to_tb = 1.222e6 / (
                rest_freq_ghz**2 * beam_major_arcsec * beam_minor_arcsec
            )
            moment0_tb = moment0 * jybeam_to_tb
            nhi_map = 1.823e18 * moment0_tb
            source_mask_2d = moment0 > (0.25 * np.nanmax(moment0))
            nhi_masked = np.where(source_mask_2d, nhi_map, np.nan)

            peak_nhi = np.nanmax(nhi_masked)
            mean_nhi = np.nanmean(nhi_masked)
            hi_mass = 2.356e5 * distance_mpc**2 * integrated_flux
            hi_mass_secondary = 2.356e5 * distance_mpc**2 * secondary_flux
            inclination_deg = 58.0
            global_vrot = 0.5 * w50 / np.sin(np.deg2rad(inclination_deg))

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

            im0 = axes[0].imshow(
                moment0,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="magma",
            )
            axes[0].set_title("Moment 0 used for physical estimates")
            axes[0].set_xlabel("RA offset [arcsec]")
            axes[0].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im0, ax=axes[0], shrink=0.82, label="Jy km/s / beam")

            im1 = axes[1].imshow(
                np.log10(nhi_masked),
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="viridis",
            )
            axes[1].set_title("log10 N_HI [cm^-2]")
            axes[1].set_xlabel("RA offset [arcsec]")
            axes[1].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im1, ax=axes[1], shrink=0.82, label="log10 cm^-2")

            plt.tight_layout()
            print(f"假定 restoring beam = {beam_major_arcsec:.0f}'' x {beam_minor_arcsec:.0f}''")
            print(f"假定距离 = {distance_mpc:.1f} Mpc")
            print(f"峰值 H I 柱密度 ≈ {peak_nhi:.2e} cm^-2")
            print(f"源区平均 H I 柱密度 ≈ {mean_nhi:.2e} cm^-2")
            print(f"总 H I 质量 ≈ {hi_mass:.2e} Msun")
            print(f"若把次级谱线分量单独看待，其 H I 质量量级 ≈ {hi_mass_secondary:.2e} Msun")
            print(
                f"若把 W50 当成未分辨盘的全局轮廓，且倾角取 {inclination_deg:.0f} deg，"
                f"则粗略旋转速度入口 ≈ {global_vrot:.2f} km/s"
            )
            """
        ),
        md(
            r"""
            这里的数值只应被理解成“分析入口”，而不是最终科学结论。真正进入论文级工作时，还要继续追问：

            - beam 稀释有没有把峰值柱密度压低？
            - aperture 与 mask 有没有漏掉低面亮度扩展发射？
            - 如果线型明显复杂，总质量和次级分量通量的分配是否依赖模型假设？
            - 若要讨论动力学，是否已经需要 tilted-ring、3D cube fitting 或更系统的 PV 解读？
            """
        ),
        md(
            r"""
            ### 9.7.9 与真实软件流程的对应

            若把这个练习映射到真实软件环境，最常见的谱线处理链大致是：

            - `plotms` / `plotbandpass`：先确认 bandpass 与 line-free channels；
            - `uvcontsub`：在可见度域做连续谱扣除，必要时加入低阶多项式；
            - `tclean`（cube mode）：逐通道成像得到 data cube；
            - `immoments` 或更稳健的外部工具：在 mask 基础上构造矩图；
            - `impv` 或自定义切片：抽取 PV diagram；
            - `specfit` 或更完整的模型：比较单分量和多分量线型；
            - 自定义物理量换算：把 moment 0 和积分线通量连回 $N_{\mathrm{HI}}$、$M_{\mathrm{HI}}$ 和粗略动力学量。

            这里最重要的专业判断是：**谱线处理不只是“连续谱成像再多一维”。** 它需要更谨慎的频谱基线处理、更明确的 line-free 选择、更有意识的掩膜构造，以及把空间结构和速度结构一起读出来的习惯。
            """
        ),
    ],
    "9_8_wideband_and_widefield_imaging.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.7 谱线数据处理](9_7_spectral_line_processing.ipynb)
                * 下一节： [9.9 偏振成像与 RM 入门](9_9_polarization_imaging.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.8 宽带与宽场高级成像：MFS、频谱指数与主波束校正

            当带宽变宽、视场变大以后，成像就不再只是“把所有通道平均一下再 CLEAN”。至少有三件事会同时发生：

            - 天体本身的亮度会随频率变化；
            - 主波束会随频率收缩，因此离轴源会被额外染上一个假的频谱斜率；
            - 一旦做主波束校正，图像边缘噪声会被同步放大。

            因此宽带和宽场其实是耦合问题。真实软件里，这通常会把你带到 `MFS`、`MT-MFS`、`widebandpbcor`、`A-projection` 或 `AW-projection` 这些关键词上。

            这一节不试图复刻工业级算法，而是用一个完全自包含的实验，把下面几件事讲透：

            - 为什么简单 band average 会让离轴源的频谱被主波束污染；
            - 为什么 `nterms > 1` 的 MFS 思想是有必要的；
            - 为什么主波束校正会改善频谱，但同时放大图像边缘噪声；
            - 为什么宽带/宽场成像的参数选择必须和科学目标一起考虑。
            """
        ),
        md("***"),
        code(
            """
            def normalized_gaussian(x, y, x0, y0, sigma_x, sigma_y):
                profile = np.exp(
                    -0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
                )
                profile /= profile.sum()
                return profile


            def make_restoring_beam(npix=72, sigma_pix=1.6):
                coords = np.arange(npix) - npix // 2
                xx, yy = np.meshgrid(coords, coords)
                beam = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
                beam /= beam.max()
                return beam


            def fft_convolve_same(image, kernel):
                image_ft = np.fft.fft2(image)
                kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
                return np.real(np.fft.ifft2(image_ft * kernel_ft))


            def primary_beam_gain(radius_arcsec, fwhm_arcsec):
                return np.exp(-4.0 * np.log(2.0) * (radius_arcsec / fwhm_arcsec) ** 2)


            def aperture_mask(x, y, x0, y0, radius_arcsec):
                return ((x - x0) ** 2 + (y - y0) ** 2) <= radius_arcsec**2


            def aperture_flux_spectrum(cube, mask, beam_area_pix):
                return np.nansum(cube[:, mask], axis=1) / beam_area_pix


            def fit_powerlaw_spectrum(freqs_ghz, spectrum, ref_freq_ghz):
                x = np.log(freqs_ghz / ref_freq_ghz)
                y = np.log(spectrum)
                slope, intercept = np.polyfit(x, y, 1)
                model = np.exp(intercept) * (freqs_ghz / ref_freq_ghz) ** slope
                return np.exp(intercept), slope, model


            def fit_powerlaw_map(cube, freqs_ghz, ref_freq_ghz, min_signal, valid_mask):
                i0_map = np.full(cube.shape[1:], np.nan, dtype=float)
                alpha_map = np.full_like(i0_map, np.nan)
                x = np.log(freqs_ghz / ref_freq_ghz)
                for iy in range(cube.shape[1]):
                    for ix in range(cube.shape[2]):
                        if not valid_mask[iy, ix]:
                            continue
                        spec = cube[:, iy, ix]
                        if np.any(~np.isfinite(spec)) or np.any(spec <= min_signal):
                            continue
                        _, alpha, model = fit_powerlaw_spectrum(freqs_ghz, spec, ref_freq_ghz)
                        alpha_map[iy, ix] = alpha
                        i0_map[iy, ix] = model[np.argmin(np.abs(freqs_ghz - ref_freq_ghz))]
                return i0_map, alpha_map


            def radius_at_pb_gain(radius_grid_arcsec, pb_profile, gain_level):
                valid = np.where(pb_profile >= gain_level)[0]
                if valid.size == 0:
                    return 0.0
                return radius_grid_arcsec[valid[-1]]


            npix = 72
            cell_arcsec = 2.0
            coords = (np.arange(npix) - npix // 2) * cell_arcsec
            x_grid, y_grid = np.meshgrid(coords, coords)
            radius_grid = np.sqrt(x_grid**2 + y_grid**2)

            freqs_ghz = np.linspace(1.0, 2.0, 12)
            ref_freq_ghz = np.median(freqs_ghz)
            noise_std = 0.0015

            sources = [
                {"name": "core", "flux_ref": 0.95, "alpha": -0.70, "x0": 0.0, "y0": 0.0, "sx": 3.0, "sy": 2.4},
                {"name": "steep_offaxis", "flux_ref": 0.55, "alpha": -1.15, "x0": 30.0, "y0": -18.0, "sx": 3.4, "sy": 2.2},
                {"name": "flat_knot", "flux_ref": 0.24, "alpha": 0.15, "x0": -24.0, "y0": 24.0, "sx": 2.4, "sy": 2.4},
                {"name": "ridge", "flux_ref": 0.32, "alpha": -0.40, "x0": 18.0, "y0": 26.0, "sx": 8.0, "sy": 3.0},
            ]

            true_cube = np.zeros((freqs_ghz.size, npix, npix))
            true_ref_image = np.zeros((npix, npix))
            true_alpha_image = np.zeros((npix, npix))

            for src in sources:
                profile = normalized_gaussian(
                    x_grid, y_grid, src["x0"], src["y0"], src["sx"], src["sy"]
                )
                true_ref_image += src["flux_ref"] * profile
                true_alpha_image += src["alpha"] * profile / np.max(profile)
                for fi, freq in enumerate(freqs_ghz):
                    true_cube[fi] += src["flux_ref"] * (freq / ref_freq_ghz) ** src["alpha"] * profile

            restoring_beam = make_restoring_beam(npix=npix, sigma_pix=1.6)
            beam_area_pix = restoring_beam.sum()

            pb_fwhm_ref_arcsec = 96.0
            pb_cube = np.zeros_like(true_cube)
            true_cube_restored = np.zeros_like(true_cube)
            observed_cube = np.zeros_like(true_cube)
            for fi, freq in enumerate(freqs_ghz):
                pb_fwhm = pb_fwhm_ref_arcsec * (ref_freq_ghz / freq)
                pb_cube[fi] = primary_beam_gain(radius_grid, pb_fwhm)
                true_cube_restored[fi] = fft_convolve_same(true_cube[fi], restoring_beam)
                observed_cube[fi] = fft_convolve_same(true_cube[fi] * pb_cube[fi], restoring_beam)

            observed_cube += noise_std * RNG.normal(size=observed_cube.shape)

            pb_threshold = 0.28
            cube_pbcor = np.where(pb_cube > pb_threshold, observed_cube / pb_cube, np.nan)
            band_avg_obs = observed_cube.mean(axis=0)
            ref_index = int(np.argmin(np.abs(freqs_ghz - ref_freq_ghz)))
            true_ref_restored = true_cube_restored[ref_index]
            band_avg_true = true_cube_restored.mean(axis=0)
            valid_counts_pbcor = np.sum(np.isfinite(cube_pbcor), axis=0)
            band_avg_pbcor = np.full_like(band_avg_obs, np.nan)
            np.divide(
                np.nansum(cube_pbcor, axis=0),
                valid_counts_pbcor,
                out=band_avg_pbcor,
                where=valid_counts_pbcor > 0,
            )

            source_masks = {
                "core": aperture_mask(x_grid, y_grid, 0.0, 0.0, 8.0),
                "steep_offaxis": aperture_mask(x_grid, y_grid, 30.0, -18.0, 8.0),
                "flat_knot": aperture_mask(x_grid, y_grid, -24.0, 24.0, 7.0),
            }

            true_spec_off = aperture_flux_spectrum(
                true_cube_restored, source_masks["steep_offaxis"], beam_area_pix
            )
            obs_spec_off = aperture_flux_spectrum(observed_cube, source_masks["steep_offaxis"], beam_area_pix)
            pbcor_spec_off = aperture_flux_spectrum(cube_pbcor, source_masks["steep_offaxis"], beam_area_pix)
            band_flux_true_off = np.nansum(band_avg_true[source_masks["steep_offaxis"]]) / beam_area_pix
            band_flux_obs_off = np.nansum(band_avg_obs[source_masks["steep_offaxis"]]) / beam_area_pix
            band_flux_pbcor_off = np.nansum(band_avg_pbcor[source_masks["steep_offaxis"]]) / beam_area_pix
            """
        ),
        md(
            r"""
            ### 9.8.1 宽带和宽场为什么会耦合

            若只看中心源，宽带成像似乎只是“多了一点频谱信息”；但一旦把视场拉大，主波束的频率依赖就会把离轴源额外压暗，并在人为制造一个偏陡的频谱斜率。

            下面把三张图放在一起：

            - 参考频率下的真实天空；
            - 不做主波束校正、直接对所有通道平均的宽带图像；
            - 先做主波束校正、再做带宽平均的图像。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

            for ax, image, title in [
                (axes[0], true_ref_restored, "True sky at reference frequency"),
                (axes[1], band_avg_obs, "Naive band average without PB correction"),
                (axes[2], band_avg_pbcor, "Band average after PB correction"),
            ]:
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=[coords[0], coords[-1], coords[0], coords[-1]],
                    cmap="inferno",
                )
                ax.set_title(title)
                ax.set_xlabel("RA offset [arcsec]")
                ax.set_ylabel("Dec offset [arcsec]")
                plt.colorbar(im, ax=ax, shrink=0.80, label="Jy/beam")

            plt.tight_layout()
            print(f"若不受主波束影响，宽带平均后的离轴源积分通量 ≈ {band_flux_true_off:.3f} Jy")
            print(f"未做 PB 校正的宽带平均积分通量 ≈ {band_flux_obs_off:.3f} Jy")
            print(f"做了 PB 校正后的宽带平均积分通量 ≈ {band_flux_pbcor_off:.3f} Jy")
            """
        ),
        md(
            r"""
            这里最核心的判断是：**离轴源变暗并不一定意味着天体本身更陡，也可能只是主波束在高频端缩小了。** 这就是为什么宽带和宽场效应必须一起处理。
            """
        ),
        md(
            r"""
            ### 9.8.2 频变主波束会如何污染频谱指数

            下面挑出一个离轴陡谱源，直接比较三条谱线：

            - 真实频谱；
            - 未做主波束校正时在图像域测得的频谱；
            - 做了主波束校正后的频谱。

            同时再画出中心和离轴位置的主波束增益随频率变化的趋势。这样能很直观地看到“假的谱指数”是怎么来的。
            """
        ),
        code(
            """
            _, alpha_true_off, model_true_off = fit_powerlaw_spectrum(
                freqs_ghz, true_spec_off, ref_freq_ghz
            )
            _, alpha_obs_off, model_obs_off = fit_powerlaw_spectrum(
                freqs_ghz, np.clip(obs_spec_off, 1e-6, None), ref_freq_ghz
            )
            _, alpha_pbcor_off, model_pbcor_off = fit_powerlaw_spectrum(
                freqs_ghz, np.clip(pbcor_spec_off, 1e-6, None), ref_freq_ghz
            )

            center_gain = np.array([pb_cube[fi, npix // 2, npix // 2] for fi in range(freqs_ghz.size)])
            offaxis_gain = np.array([
                pb_cube[fi, np.argmin(np.abs(coords - (-18.0))), np.argmin(np.abs(coords - 30.0))]
                for fi in range(freqs_ghz.size)
            ])

            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

            axes[0].plot(freqs_ghz, true_spec_off, color="black", lw=1.8, label=f"true spectrum, alpha={alpha_true_off:.2f}")
            axes[0].plot(freqs_ghz, obs_spec_off, color="tab:red", lw=1.8, label=f"observed without PB correction, alpha={alpha_obs_off:.2f}")
            axes[0].plot(freqs_ghz, pbcor_spec_off, color="tab:blue", lw=1.8, label=f"after PB correction, alpha={alpha_pbcor_off:.2f}")
            axes[0].set_xlabel("frequency [GHz]")
            axes[0].set_ylabel("integrated flux density [Jy]")
            axes[0].set_title("Off-axis source spectrum")
            axes[0].legend(loc="upper right")

            axes[1].plot(freqs_ghz, center_gain, color="tab:green", lw=1.8, label="beam gain at field center")
            axes[1].plot(freqs_ghz, offaxis_gain, color="tab:purple", lw=1.8, label="beam gain at off-axis source")
            axes[1].set_xlabel("frequency [GHz]")
            axes[1].set_ylabel("primary beam gain")
            axes[1].set_title("Frequency-dependent primary beam")
            axes[1].legend(loc="lower left")

            plt.tight_layout()
            print(f"离轴源真实频谱指数 ≈ {alpha_true_off:.2f}")
            print(f"未做 PB 校正时测得的频谱指数 ≈ {alpha_obs_off:.2f}")
            print(f"做了 PB 校正后的频谱指数 ≈ {alpha_pbcor_off:.2f}")
            """
        ),
        md(
            r"""
            这个实验基本就说明了 `widebandpbcor` 这类步骤存在的必要性：如果不先把主波束的频率依赖剥掉，你测到的“谱指数”里会混进明显的仪器项。
            """
        ),
        md(
            r"""
            ### 9.8.3 用简化的 `nterms=2` 思想恢复参考频率亮度和谱指数

            `MT-MFS` 的核心精神，是在宽带数据里同时解出“参考频率亮度”和“频谱变化项”。下面不复刻完整算法，只做一个简化版的逐像素幂律拟合：

            - 对未做主波束校正的 cube 拟合一个表观谱指数图；
            - 对做了主波束校正的 cube 再拟合一次；
            - 比较不同源位置的谱指数恢复结果。
            """
        ),
        code(
            """
            valid_mask_obs = np.nanmax(observed_cube, axis=0) > (6.0 * noise_std)
            cube_pbcor_finite_max = np.max(
                np.where(np.isfinite(cube_pbcor), cube_pbcor, -np.inf), axis=0
            )
            valid_mask_pbcor = (
                np.all(pb_cube > 0.35, axis=0)
                & np.isfinite(cube_pbcor_finite_max)
                & (cube_pbcor_finite_max > (6.0 * noise_std))
            )

            i0_obs, alpha_map_obs = fit_powerlaw_map(
                np.clip(observed_cube, 1e-6, None),
                freqs_ghz,
                ref_freq_ghz,
                min_signal=2.5 * noise_std,
                valid_mask=valid_mask_obs,
            )
            i0_pbcor, alpha_map_pbcor = fit_powerlaw_map(
                cube_pbcor,
                freqs_ghz,
                ref_freq_ghz,
                min_signal=2.5 * noise_std,
                valid_mask=valid_mask_pbcor,
            )

            fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))

            im0 = axes[0].imshow(
                i0_obs,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="inferno",
            )
            axes[0].set_title("Reference-frequency intensity without PB correction")
            axes[0].set_xlabel("RA offset [arcsec]")
            axes[0].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im0, ax=axes[0], shrink=0.80, label="Jy/beam")

            im1 = axes[1].imshow(
                alpha_map_obs,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="coolwarm",
                vmin=-1.8,
                vmax=0.4,
            )
            axes[1].set_title("Apparent alpha map without PB correction")
            axes[1].set_xlabel("RA offset [arcsec]")
            axes[1].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im1, ax=axes[1], shrink=0.80, label="alpha")

            im2 = axes[2].imshow(
                alpha_map_pbcor,
                origin="lower",
                extent=[coords[0], coords[-1], coords[0], coords[-1]],
                cmap="coolwarm",
                vmin=-1.8,
                vmax=0.4,
            )
            axes[2].set_title("Alpha map after PB correction")
            axes[2].set_xlabel("RA offset [arcsec]")
            axes[2].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im2, ax=axes[2], shrink=0.80, label="alpha")

            plt.tight_layout()

            source_truth = {"core": -0.70, "steep_offaxis": -1.15, "flat_knot": 0.15}
            for name, truth in source_truth.items():
                est_obs = np.nanmean(alpha_map_obs[source_masks[name]])
                est_pb = np.nanmean(alpha_map_pbcor[source_masks[name]])
                print(f"{name} 真实 alpha = {truth:+.2f}，未校正图像域估计 = {est_obs:+.2f}，PB 校正后估计 = {est_pb:+.2f}")
            """
        ),
        md(
            r"""
            这个练习对应的正是现代宽带成像里 `nterms=2` 或 `nterms=3` 的基本动机：**不仅要成像，还要把频谱项一起恢复出来。** 如果只做单频近似，宽带数据里的信息会被浪费掉，甚至被主波束污染。
            """
        ),
        md(
            r"""
            ### 9.8.4 主波束校正会把边缘噪声一起放大

            主波束校正几乎总是必要的，但它不是免费的。因为你在除以主波束增益时，也同时在除以噪声。

            下面用一个径向剖面把这个代价画出来：参考频率下的主波束增益、以及做主波束校正后噪声随视场半径增长的趋势。
            """
        ),
        code(
            """
            radius_arcsec = np.linspace(0.0, 70.0, 300)
            pb_profile_ref = primary_beam_gain(radius_arcsec, pb_fwhm_ref_arcsec)
            pb_profile_low = primary_beam_gain(radius_arcsec, pb_fwhm_ref_arcsec * (ref_freq_ghz / freqs_ghz[0]))
            pb_profile_high = primary_beam_gain(radius_arcsec, pb_fwhm_ref_arcsec * (ref_freq_ghz / freqs_ghz[-1]))
            corrected_noise_ref = noise_std / np.maximum(pb_profile_ref, 1e-3)

            radius_pb50 = radius_at_pb_gain(radius_arcsec, pb_profile_ref, 0.50)
            radius_pb25 = radius_at_pb_gain(radius_arcsec, pb_profile_ref, 0.25)

            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

            axes[0].plot(radius_arcsec, pb_profile_low, color="tab:blue", lw=1.8, label=f"{freqs_ghz[0]:.1f} GHz")
            axes[0].plot(radius_arcsec, pb_profile_ref, color="black", lw=1.8, label=f"{ref_freq_ghz:.1f} GHz")
            axes[0].plot(radius_arcsec, pb_profile_high, color="tab:red", lw=1.8, label=f"{freqs_ghz[-1]:.1f} GHz")
            axes[0].axhline(0.5, color="gray", ls="--", lw=1.0)
            axes[0].axhline(0.25, color="gray", ls=":", lw=1.0)
            axes[0].set_xlabel("field radius [arcsec]")
            axes[0].set_ylabel("primary beam gain")
            axes[0].set_title("Frequency-dependent primary beam profile")
            axes[0].legend(loc="upper right")

            axes[1].plot(radius_arcsec, corrected_noise_ref, color="tab:purple", lw=2.0)
            axes[1].axvline(radius_pb50, color="gray", ls="--", lw=1.0, label=f"PB=0.5 at {radius_pb50:.1f}''")
            axes[1].axvline(radius_pb25, color="gray", ls=":", lw=1.0, label=f"PB=0.25 at {radius_pb25:.1f}''")
            axes[1].set_xlabel("field radius [arcsec]")
            axes[1].set_ylabel("effective noise after PB correction [Jy/beam]")
            axes[1].set_title("Noise amplification after PB correction")
            axes[1].legend(loc="upper left")

            plt.tight_layout()
            print(f"参考频率下 PB=0.50 的视场半径约为 {radius_pb50:.1f} arcsec")
            print(f"参考频率下 PB=0.25 的视场半径约为 {radius_pb25:.1f} arcsec")
            print(
                f"若把参考频率下的图像边缘推进到 PB=0.25，等效噪声会放大到中心的 "
                f"{corrected_noise_ref[np.argmin(np.abs(radius_arcsec - radius_pb25))] / noise_std:.2f} 倍"
            )
            """
        ),
        md(
            r"""
            这就是为什么真实数据处理中常常需要同时指定：

            - 成像视场多大；
            - 主波束校正做到什么 cutoff；
            - 最终科学分析只在什么半径内报告结果。

            如果科学目标是离轴微弱源或频谱指数图，那么这些参数不能靠“默认值”处理。
            """
        ),
        md(
            r"""
            ### 9.8.5 与真实软件流程的对应

            若把这个教学实验映射到真实软件环境，最常见的工作流大致是：

            - `tclean` + `specmode='mfs'`：做宽带连续谱成像；
            - `nterms=2/3`：让成像同时解参考频率亮度和频谱项；
            - `widebandpbcor` 或等价步骤：校正主波束引入的频谱偏差；
            - `wprojplanes`、`A-projection`、`AW-projection`：在更宽场、更高动态范围情形下处理方向相关效应；
            - 参数选择：把 `cell`、`imsize`、`pb cutoff`、`nterms` 和科学目标联动起来。

            这个 notebook 依然只是简化版，但它想传达的专业判断已经很明确：**宽带频谱、主波束和宽场效应是同一个问题族，而不是三个彼此独立的小修饰项。**
            """
        ),
    ],
    "9_9_polarization_imaging.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.8 宽带与宽场高级成像](9_8_wideband_and_widefield_imaging.ipynb)
                * 下一节： [9.10 短间距、单碟与联合成像](9_10_short_spacing_and_feather.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.9 偏振成像与 RM 入门：Stokes 成像、偏振偏差与 leakage

            偏振数据处理通常是很多初学者第一次真正意识到“干涉成像不只是在做 Stokes I 图”的地方。到了偏振问题，至少有四件事必须同时考虑：

            - 你不仅要成 `I`，还要理解 `Q/U/V` 各自代表什么；
            - 线偏振强度 $P=\sqrt{Q^2+U^2}$ 在低信噪比下会有正偏差；
            - 很小的极化泄漏（leakage / D-term）就足以把强 Stokes I 源污染成“假偏振源”；
            - 一旦跨频带研究偏振角，就会遇到 Faraday rotation 和 rotation measure (`RM`)。

            这一节不复刻完整的交叉手校准链，而是用一个完全自包含的合成实验，把下面几件关键判断建立起来：

            - Stokes `I/Q/U/V` 图像在数值上分别长什么样；
            - 为什么偏振强度需要 debias；
            - leakage 为什么会制造假偏振，以及为什么 `D-term` 校准是必要的；
            - 简单 Faraday screen 下，为什么偏振角会随 $\lambda^2$ 近似线性变化。
            """
        ),
        md("***"),
        code(
            """
            c_light = 0.299792458


            def normalized_gaussian(x, y, x0, y0, sigma_x, sigma_y):
                profile = np.exp(
                    -0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
                )
                profile /= profile.sum()
                return profile


            def make_restoring_beam(npix=72, sigma_pix=1.7):
                coords = np.arange(npix) - npix // 2
                xx, yy = np.meshgrid(coords, coords)
                beam = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
                beam /= beam.max()
                return beam


            def fft_convolve_same(image, kernel):
                image_ft = np.fft.fft2(image)
                kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
                return np.real(np.fft.ifft2(image_ft * kernel_ft))


            def aperture_mask(x, y, x0, y0, radius_arcsec):
                return ((x - x0) ** 2 + (y - y0) ** 2) <= radius_arcsec**2


            def aperture_spectrum(cube, mask, beam_area_pix):
                return np.nansum(cube[:, mask], axis=1) / beam_area_pix


            def linear_polarization(q_image, u_image, noise_std):
                p_raw = np.sqrt(q_image**2 + u_image**2)
                p_debiased = np.sqrt(np.clip(p_raw**2 - noise_std**2, 0.0, None))
                return p_raw, p_debiased


            def evpa_image(q_image, u_image, min_p):
                evpa = 0.5 * np.rad2deg(np.arctan2(u_image, q_image))
                p_linear = np.sqrt(q_image**2 + u_image**2)
                return np.where(p_linear > min_p, evpa, np.nan)


            def fit_rotation_measure(lambda2, q_spec, u_spec, min_p):
                p_spec = np.sqrt(q_spec**2 + u_spec**2)
                valid = p_spec > min_p
                chi = np.full_like(lambda2, np.nan, dtype=float)
                model = np.full_like(lambda2, np.nan, dtype=float)
                if np.count_nonzero(valid) < 3:
                    return np.nan, np.nan, chi, valid, model
                two_chi_unwrapped = np.unwrap(np.angle(q_spec[valid] + 1j * u_spec[valid]))
                chi_valid = 0.5 * two_chi_unwrapped
                slope, intercept = np.polyfit(lambda2[valid], chi_valid, 1)
                chi[valid] = chi_valid
                model = intercept + slope * lambda2
                return intercept, slope, chi, valid, model


            npix = 72
            cell_arcsec = 2.0
            coords = (np.arange(npix) - npix // 2) * cell_arcsec
            x_grid, y_grid = np.meshgrid(coords, coords)
            radius_grid = np.sqrt(x_grid**2 + y_grid**2)

            freqs_ghz = np.linspace(1.0, 2.0, 18)
            lambda2 = (c_light / freqs_ghz) ** 2
            ref_index = int(np.argmin(np.abs(freqs_ghz - np.median(freqs_ghz))))
            ref_freq_ghz = freqs_ghz[ref_index]
            lambda2_ref = lambda2[ref_index]

            sources = [
                {
                    "name": "core",
                    "flux_ref": 1.10,
                    "alpha": -0.65,
                    "x0": 0.0,
                    "y0": 0.0,
                    "sx": 3.2,
                    "sy": 2.6,
                    "pol_frac": 0.05,
                    "chi0_deg": 18.0,
                    "rm": 35.0,
                    "v_frac": 0.015,
                },
                {
                    "name": "unpolarized_hotspot",
                    "flux_ref": 0.72,
                    "alpha": -0.95,
                    "x0": -24.0,
                    "y0": 22.0,
                    "sx": 2.8,
                    "sy": 2.8,
                    "pol_frac": 0.0,
                    "chi0_deg": 0.0,
                    "rm": 0.0,
                    "v_frac": 0.0,
                },
                {
                    "name": "polarized_lobe",
                    "flux_ref": 0.48,
                    "alpha": -1.05,
                    "x0": 28.0,
                    "y0": -16.0,
                    "sx": 4.0,
                    "sy": 2.8,
                    "pol_frac": 0.22,
                    "chi0_deg": -28.0,
                    "rm": -120.0,
                    "v_frac": 0.0,
                },
            ]

            true_i_cube = np.zeros((freqs_ghz.size, npix, npix))
            true_q_cube = np.zeros_like(true_i_cube)
            true_u_cube = np.zeros_like(true_i_cube)
            true_v_cube = np.zeros_like(true_i_cube)

            for src in sources:
                profile = normalized_gaussian(
                    x_grid, y_grid, src["x0"], src["y0"], src["sx"], src["sy"]
                )
                chi0_rad = np.deg2rad(src["chi0_deg"])
                for fi, freq in enumerate(freqs_ghz):
                    i_amp = src["flux_ref"] * (freq / ref_freq_ghz) ** src["alpha"]
                    chi = chi0_rad + src["rm"] * (lambda2[fi] - lambda2_ref)
                    q_amp = i_amp * src["pol_frac"] * np.cos(2.0 * chi)
                    u_amp = i_amp * src["pol_frac"] * np.sin(2.0 * chi)
                    v_amp = i_amp * src["v_frac"]
                    true_i_cube[fi] += i_amp * profile
                    true_q_cube[fi] += q_amp * profile
                    true_u_cube[fi] += u_amp * profile
                    true_v_cube[fi] += v_amp * profile

            restoring_beam = make_restoring_beam(npix=npix, sigma_pix=1.7)
            beam_area_pix = restoring_beam.sum()

            noise_i = 0.0014
            noise_quv = 0.0009
            observed_i_cube = np.zeros_like(true_i_cube)
            observed_q_cube = np.zeros_like(true_i_cube)
            observed_u_cube = np.zeros_like(true_i_cube)
            observed_v_cube = np.zeros_like(true_i_cube)

            for fi in range(freqs_ghz.size):
                observed_i_cube[fi] = (
                    fft_convolve_same(true_i_cube[fi], restoring_beam)
                    + noise_i * RNG.normal(size=(npix, npix))
                )
                observed_q_cube[fi] = (
                    fft_convolve_same(true_q_cube[fi], restoring_beam)
                    + noise_quv * RNG.normal(size=(npix, npix))
                )
                observed_u_cube[fi] = (
                    fft_convolve_same(true_u_cube[fi], restoring_beam)
                    + noise_quv * RNG.normal(size=(npix, npix))
                )
                observed_v_cube[fi] = (
                    fft_convolve_same(true_v_cube[fi], restoring_beam)
                    + noise_quv * RNG.normal(size=(npix, npix))
                )

            dq_leak = 0.018
            du_leak = -0.012
            q_leaky_cube = observed_q_cube + dq_leak * observed_i_cube
            u_leaky_cube = observed_u_cube + du_leak * observed_i_cube
            q_corrected_cube = q_leaky_cube - dq_leak * observed_i_cube
            u_corrected_cube = u_leaky_cube - du_leak * observed_i_cube

            source_masks = {
                "core": aperture_mask(x_grid, y_grid, 0.0, 0.0, 8.0),
                "unpolarized_hotspot": aperture_mask(x_grid, y_grid, -24.0, 22.0, 8.0),
                "polarized_lobe": aperture_mask(x_grid, y_grid, 28.0, -16.0, 8.0),
            }

            ref_i = observed_i_cube[ref_index]
            ref_q = observed_q_cube[ref_index]
            ref_u = observed_u_cube[ref_index]
            ref_v = observed_v_cube[ref_index]
            p_raw_ref, p_debiased_ref = linear_polarization(ref_q, ref_u, noise_quv)
            p_leaky_ref = linear_polarization(
                q_leaky_cube[ref_index], u_leaky_cube[ref_index], noise_quv
            )[1]
            p_corrected_ref = linear_polarization(
                q_corrected_cube[ref_index], u_corrected_cube[ref_index], noise_quv
            )[1]
            evpa_ref = evpa_image(ref_q, ref_u, min_p=5.0 * noise_quv)

            integrated = {}
            for name, mask in source_masks.items():
                integrated[name] = {
                    "I": aperture_spectrum(observed_i_cube, mask, beam_area_pix),
                    "Q": aperture_spectrum(observed_q_cube, mask, beam_area_pix),
                    "U": aperture_spectrum(observed_u_cube, mask, beam_area_pix),
                    "V": aperture_spectrum(observed_v_cube, mask, beam_area_pix),
                    "Q_leaky": aperture_spectrum(q_leaky_cube, mask, beam_area_pix),
                    "U_leaky": aperture_spectrum(u_leaky_cube, mask, beam_area_pix),
                    "Q_corrected": aperture_spectrum(q_corrected_cube, mask, beam_area_pix),
                    "U_corrected": aperture_spectrum(u_corrected_cube, mask, beam_area_pix),
                }
            """
        ),
        md(
            r"""
            ### 9.9.1 先看懂 Stokes `I/Q/U/V` 图像分别在说什么

            在偏振成像里，最容易混淆的事情之一是：`Q/U/V` 不是 “强度更弱一点的 I 图”，而是不同偏振态在仪器基上的投影。

            下面在参考频率切出一组图，分别看：

            - `I`：总强度；
            - `Q/U`：线偏振在两个正交基上的投影；
            - `V`：圆偏振；
            - debiased 的线偏振强度；
            - 偏振角 `EVPA`。
            """
        ),
        code(
            """
            extent = [coords[0], coords[-1], coords[0], coords[-1]]
            fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4))

            panels = [
                (ref_i, "Stokes I", "inferno", None, None),
                (ref_q, "Stokes Q", "coolwarm", -0.015, 0.015),
                (ref_u, "Stokes U", "coolwarm", -0.015, 0.015),
                (ref_v, "Stokes V", "coolwarm", -0.015, 0.015),
                (p_debiased_ref, "Debiased linear polarization", "magma", 0.0, 0.020),
                (evpa_ref, "EVPA [deg]", "twilight", -90.0, 90.0),
            ]

            for ax, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(title)
                ax.set_xlabel("RA offset [arcsec]")
                ax.set_ylabel("Dec offset [arcsec]")
                plt.colorbar(im, ax=ax, shrink=0.82)

            plt.tight_layout()

            core_i_ref = integrated["core"]["I"][ref_index]
            core_p_ref = np.sqrt(
                integrated["core"]["Q"][ref_index] ** 2 + integrated["core"]["U"][ref_index] ** 2
            )
            core_v_ref = np.abs(integrated["core"]["V"][ref_index])
            lobe_i_ref = integrated["polarized_lobe"]["I"][ref_index]
            lobe_p_ref = np.sqrt(
                integrated["polarized_lobe"]["Q"][ref_index] ** 2
                + integrated["polarized_lobe"]["U"][ref_index] ** 2
            )

            print(f"core 的线偏振分数 ≈ {100.0 * core_p_ref / core_i_ref:.1f}%")
            print(f"core 的圆偏振分数 ≈ {100.0 * core_v_ref / core_i_ref:.1f}%")
            print(f"polarized_lobe 的线偏振分数 ≈ {100.0 * lobe_p_ref / lobe_i_ref:.1f}%")
            """
        ),
        md(
            r"""
            这一步的重点不是“记住哪种颜色代表哪种偏振”，而是建立一个判断：**偏振分析从来都不只是再多画几张图，而是在同一个数据立方里追踪多个物理自由度。**
            """
        ),
        md(
            r"""
            ### 9.9.2 为什么线偏振强度需要 debias

            由于 $P=\sqrt{Q^2+U^2}$ 永远不小于零，即使纯噪声区域里 `Q/U` 的期望值是零，直接算出来的 `P` 也会偏正。这个问题在弱偏振源附近尤其关键。

            下面专门看离源背景区域，比较 raw `P` 和 debiased `P` 的分布。
            """
        ),
        code(
            """
            background_mask = radius_grid > 55.0

            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

            axes[0].hist(ref_q[background_mask] / noise_quv, bins=28, alpha=0.65, label="Q / sigma")
            axes[0].hist(ref_u[background_mask] / noise_quv, bins=28, alpha=0.65, label="U / sigma")
            axes[0].set_xlabel("background pixel value / sigma")
            axes[0].set_ylabel("number of pixels")
            axes[0].set_title("Q/U remain symmetric around zero")
            axes[0].legend(loc="upper right")

            axes[1].hist(p_raw_ref[background_mask] / noise_quv, bins=28, alpha=0.70, label="raw P / sigma")
            axes[1].hist(
                p_debiased_ref[background_mask] / noise_quv,
                bins=28,
                alpha=0.70,
                label="debiased P / sigma",
            )
            axes[1].set_xlabel("background pixel value / sigma")
            axes[1].set_ylabel("number of pixels")
            axes[1].set_title("Raw P is positively biased")
            axes[1].legend(loc="upper right")

            plt.tight_layout()

            print(f"离源区域 raw P 的中位数 ≈ {1e3 * np.median(p_raw_ref[background_mask]):.2f} mJy/beam")
            print(
                f"离源区域 debiased P 的中位数 ≈ "
                f"{1e3 * np.median(p_debiased_ref[background_mask]):.2f} mJy/beam"
            )
            """
        ),
        md(
            r"""
            在真实数据里，这个步骤会直接影响：

            - 弱偏振源是否被误判为探测到；
            - 偏振分数 $m_L=P/I$ 是否被系统性高估；
            - 后续的 `EVPA` 与 `RM` 分析是否建立在可靠像素之上。
            """
        ),
        md(
            r"""
            ### 9.9.3 leakage / D-term 为什么会制造“假偏振”

            偏振校准里最经典的系统误差之一，就是把一部分强 Stokes `I` 信号泄漏进 `Q/U`。这类方向无关的一阶污染，常常用 `D-term` 来描述。

            下面人为加入一个简化 leakage：

            - `Q <- Q + 0.018 I`
            - `U <- U - 0.012 I`

            然后比较参考频率下的线偏振图像，以及一个本来不偏振的热点在校正前后的表观偏振分数。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))

            for ax, image, title in [
                (axes[0], p_debiased_ref, "Observed debiased P"),
                (axes[1], p_leaky_ref, "After adding leakage"),
                (axes[2], p_corrected_ref, "After leakage correction"),
            ]:
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    vmin=0.0,
                    vmax=0.028,
                )
                ax.set_title(title)
                ax.set_xlabel("RA offset [arcsec]")
                ax.set_ylabel("Dec offset [arcsec]")
                plt.colorbar(im, ax=ax, shrink=0.82, label="Jy/beam")

            plt.tight_layout()

            hotspot_i_ref = integrated["unpolarized_hotspot"]["I"][ref_index]
            hotspot_p_ref = np.sqrt(
                integrated["unpolarized_hotspot"]["Q"][ref_index] ** 2
                + integrated["unpolarized_hotspot"]["U"][ref_index] ** 2
            )
            hotspot_p_leaky_ref = np.sqrt(
                integrated["unpolarized_hotspot"]["Q_leaky"][ref_index] ** 2
                + integrated["unpolarized_hotspot"]["U_leaky"][ref_index] ** 2
            )
            hotspot_p_corrected_ref = np.sqrt(
                integrated["unpolarized_hotspot"]["Q_corrected"][ref_index] ** 2
                + integrated["unpolarized_hotspot"]["U_corrected"][ref_index] ** 2
            )

            print(
                f"非偏振热点在原始 Q/U 图中的表观线偏振分数 ≈ "
                f"{100.0 * hotspot_p_ref / hotspot_i_ref:.2f}%"
            )
            print(
                f"加入 leakage 后表观线偏振分数 ≈ "
                f"{100.0 * hotspot_p_leaky_ref / hotspot_i_ref:.2f}%"
            )
            print(
                f"做 leakage 校正后表观线偏振分数 ≈ "
                f"{100.0 * hotspot_p_corrected_ref / hotspot_i_ref:.2f}%"
            )
            """
        ),
        md(
            r"""
            这正是偏振校准为什么不能被当成“细节修饰项”的原因。对于强源附近的弱偏振结构来说，哪怕只有百分之几的 leakage，也足以在科学解释上造成根本误导。
            """
        ),
        md(
            r"""
            ### 9.9.4 用 $\lambda^2$ 线性关系做一个最小 RM 拟合

            在简单 Faraday screen 近似下，偏振角满足：

            $$
            \chi(\lambda^2) = \chi_0 + \mathrm{RM}\,\lambda^2 .
            $$

            这意味着只要你能在多个频率上稳定测到 `Q/U`，就可以把偏振角展开到 $\lambda^2$ 空间里做直线拟合。下面分别看：

            - 一个弱偏振 core：加入 leakage 后，`RM` 会被明显拉偏；
            - 一个高偏振 lobe：在这个简化例子里，`RM` 可以被较稳定地恢复出来。
            """
        ),
        code(
            """
            core_fit_obs = fit_rotation_measure(
                lambda2,
                integrated["core"]["Q"],
                integrated["core"]["U"],
                min_p=0.003,
            )
            core_fit_leaky = fit_rotation_measure(
                lambda2,
                integrated["core"]["Q_leaky"],
                integrated["core"]["U_leaky"],
                min_p=0.003,
            )
            core_fit_corrected = fit_rotation_measure(
                lambda2,
                integrated["core"]["Q_corrected"],
                integrated["core"]["U_corrected"],
                min_p=0.003,
            )
            lobe_fit_obs = fit_rotation_measure(
                lambda2,
                integrated["polarized_lobe"]["Q"],
                integrated["polarized_lobe"]["U"],
                min_p=0.003,
            )

            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))

            for chi, valid, model, label, color in [
                (core_fit_obs[2], core_fit_obs[3], core_fit_obs[4], f"observed, RM={core_fit_obs[1]:+.1f}", "tab:blue"),
                (core_fit_leaky[2], core_fit_leaky[3], core_fit_leaky[4], f"with leakage, RM={core_fit_leaky[1]:+.1f}", "tab:red"),
                (core_fit_corrected[2], core_fit_corrected[3], core_fit_corrected[4], f"corrected, RM={core_fit_corrected[1]:+.1f}", "tab:green"),
            ]:
                axes[0].plot(lambda2[valid], np.rad2deg(chi[valid]), "o", color=color)
                axes[0].plot(lambda2, np.rad2deg(model), color=color, lw=1.8, label=label)

            axes[0].set_xlabel("lambda^2 [m^2]")
            axes[0].set_ylabel("EVPA [deg]")
            axes[0].set_title("Core RM fit and leakage bias")
            axes[0].legend(loc="upper right")

            axes[1].plot(
                lambda2[lobe_fit_obs[3]],
                np.rad2deg(lobe_fit_obs[2][lobe_fit_obs[3]]),
                "o",
                color="tab:purple",
            )
            axes[1].plot(
                lambda2,
                np.rad2deg(lobe_fit_obs[4]),
                color="tab:purple",
                lw=1.8,
                label=f"observed, RM={lobe_fit_obs[1]:+.1f}",
            )
            axes[1].set_xlabel("lambda^2 [m^2]")
            axes[1].set_ylabel("EVPA [deg]")
            axes[1].set_title("Polarized lobe RM fit")
            axes[1].legend(loc="upper right")

            plt.tight_layout()

            print(
                f"core 真实 RM = +35.0 rad/m^2，"
                f"观测估计 = {core_fit_obs[1]:+.1f}，"
                f"带 leakage = {core_fit_leaky[1]:+.1f}，"
                f"校正后 = {core_fit_corrected[1]:+.1f}"
            )
            print(
                f"polarized_lobe 真实 RM = -120.0 rad/m^2，"
                f"观测估计 = {lobe_fit_obs[1]:+.1f}"
            )
            """
        ),
        md(
            r"""
            这里必须强调一个专业边界：这个例子只对应“单一 Faraday 分量、角度随 $\lambda^2$ 近似线性”的最简情形。若存在多分量或 Faraday-thick 结构，真实工作里往往要进入 `RM synthesis` 和 Faraday depth 分析，而不是只做一条直线。
            """
        ),
        md(
            r"""
            ### 9.9.5 与真实软件流程的对应

            若把这个教学实验映射回真实数据处理，通常会落到下面这些动作上：

            - 先完成总强度校准，再进入交叉手 (`cross-hand`) 偏振校准；
            - 处理 `XY phase` / `R-L phase`、cross-hand delay 和 `D-term`；
            - 成 Stokes `I/Q/U/V` 图，再推导 debiased `P`、偏振角和偏振分数；
            - 对宽频带数据，在足够信噪比和频率采样下做 `RM fit` 或 `RM synthesis`；
            - 对最终物理解释，始终检查 leakage、带宽去极化、beam depolarization 和主波束偏振响应。

            这个 notebook 仍然是入口级练习，但它已经把偏振数据处理中最常见的几个“坑”串到了同一条判断链上：**没有可靠的 `Q/U`、没有偏差控制、没有 leakage 校准，就谈不上可靠的偏振科学量。**
            """
        ),
    ],
    "9_10_short_spacing_and_feather.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.9 偏振成像与 RM 入门](9_9_polarization_imaging.ipynb)
                * 下一节： [9.11 交叉手相位校准与 RM synthesis](9_11_polarization_calibration_and_rm_synthesis.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.10 短间距、单碟与联合成像：missing flux、negative bowl 与 feather

            到了这一步，我们终于遇到现代毫米波和厘米波阵列处理中最常见、也最容易让初学者困惑的问题之一：**为什么图像分辨率看起来很好，但总通量明显不对？**

            根本原因是：干涉阵并不直接测天空图像，而是在 $uv$ 平面上取样。如果最短基线不够短，那么最靠近零空间频率的那一圈信息就没有被测到。于是：

            - 大尺度结构会被低估，甚至几乎消失；
            - 图像里常常出现围绕源分布的 `negative bowl`；
            - 单看干涉阵图像时，容易误以为源本身就只有紧致成分。

            这也是为什么真实工作流里，常常需要把干涉阵与单碟数据联合起来。单碟擅长恢复大尺度总功率，但角分辨率差；干涉阵擅长恢复小尺度细节，但缺零间距与短间距。两者互补，才更接近真实天空。

            这一节用一个完全自包含的二维实验，把四件事情讲清楚：

            - 缺短间距时，图像会怎样丢失大尺度通量；
            - 单碟为什么能补回总通量，但会显著模糊紧致结构；
            - `feather` 这种频域加权组合在做什么；
            - 为什么 feather 不是“万能恢复”，参数和数据质量仍然决定结果。
            """
        ),
        md("***"),
        code(
            """
            def flux_normalized_gaussian(
                x,
                y,
                x0,
                y0,
                sigma_x,
                sigma_y,
                cell_arcsec,
                theta_deg=0.0,
            ):
                theta = np.deg2rad(theta_deg)
                xr = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
                yr = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
                profile = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
                profile /= profile.sum() * (cell_arcsec**2)
                return profile


            def gaussian_kernel(npix, sigma_pix):
                coords = np.arange(npix) - npix // 2
                xx, yy = np.meshgrid(coords, coords)
                kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
                kernel /= kernel.sum()
                return kernel


            def fft_convolve_same(image, kernel):
                image_ft = np.fft.fft2(image)
                kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
                return np.real(np.fft.ifft2(image_ft * kernel_ft))


            def integrated_flux(image, mask, cell_arcsec):
                return np.sum(image[mask]) * (cell_arcsec**2)


            def radial_profile(image, radius_grid, bin_edges):
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                values = np.full(centers.shape, np.nan, dtype=float)
                for i, (r0, r1) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                    mask = (radius_grid >= r0) & (radius_grid < r1)
                    if np.any(mask):
                        values[i] = np.mean(image[mask])
                return centers, values


            def aperture_curve(image, radius_grid, radii, cell_arcsec):
                return np.array(
                    [integrated_flux(image, radius_grid <= rr, cell_arcsec) for rr in radii]
                )


            def feather_combine(int_image, sd_image, k_radius, k_join):
                w_sd = np.exp(-(k_radius / k_join) ** 2)
                ft_combined = (
                    (1.0 - w_sd) * np.fft.fft2(int_image)
                    + w_sd * np.fft.fft2(sd_image)
                )
                return np.real(np.fft.ifft2(ft_combined)), w_sd


            npix = 96
            cell_arcsec = 2.0
            coords = (np.arange(npix) - npix // 2) * cell_arcsec
            x_grid, y_grid = np.meshgrid(coords, coords)
            radius_grid = np.sqrt(x_grid**2 + y_grid**2)

            true_sky = np.zeros((npix, npix))
            true_sky += 1.00 * flux_normalized_gaussian(
                x_grid, y_grid, 0.0, 0.0, 16.0, 14.0, cell_arcsec
            )
            true_sky += 0.35 * flux_normalized_gaussian(
                x_grid, y_grid, -10.0, 18.0, 14.0, 4.0, cell_arcsec, theta_deg=28.0
            )
            true_sky += 0.55 * flux_normalized_gaussian(
                x_grid, y_grid, 0.0, 0.0, 3.4, 2.8, cell_arcsec
            )
            true_sky += 0.20 * flux_normalized_gaussian(
                x_grid, y_grid, 28.0, -16.0, 4.5, 3.2, cell_arcsec, theta_deg=-10.0
            )
            true_sky += 0.14 * flux_normalized_gaussian(
                x_grid, y_grid, -26.0, -22.0, 5.0, 4.0, cell_arcsec, theta_deg=20.0
            )

            kx = np.fft.fftfreq(npix, d=cell_arcsec)
            ky = np.fft.fftfreq(npix, d=cell_arcsec)
            kx_grid, ky_grid = np.meshgrid(kx, ky)
            k_radius = np.sqrt(kx_grid**2 + ky_grid**2)

            mrs_arcsec = 56.0
            k_short = 1.0 / mrs_arcsec
            uv_transfer = k_radius / (k_radius + k_short)
            uv_transfer[0, 0] = 0.0

            restoring_beam = gaussian_kernel(npix, sigma_pix=1.2)
            single_dish_beam = gaussian_kernel(npix, sigma_pix=5.0)
            restoring_fwhm_arcsec = 2.355 * 1.2 * cell_arcsec
            single_dish_fwhm_arcsec = 2.355 * 5.0 * cell_arcsec

            interferometer_image = fft_convolve_same(
                np.real(np.fft.ifft2(np.fft.fft2(true_sky) * uv_transfer)),
                restoring_beam,
            )
            interferometer_image += 1.5e-4 * RNG.normal(size=(npix, npix))

            single_dish_image = fft_convolve_same(true_sky, single_dish_beam)
            single_dish_image += 8.0e-5 * RNG.normal(size=(npix, npix))

            conservative_join = 0.012
            balanced_join = 0.018
            aggressive_join = 0.024

            feather_conservative, feather_weight_conservative = feather_combine(
                interferometer_image, single_dish_image, k_radius, conservative_join
            )
            feather_balanced, feather_weight_balanced = feather_combine(
                interferometer_image, single_dish_image, k_radius, balanced_join
            )
            feather_aggressive, feather_weight_aggressive = feather_combine(
                interferometer_image, single_dish_image, k_radius, aggressive_join
            )

            science_mask = radius_grid <= 62.0
            diffuse_mask = (radius_grid >= 10.0) & (radius_grid <= 45.0)
            compact_mask = radius_grid <= 8.0
            negative_bowl_mask = (radius_grid >= 42.0) & (radius_grid <= 60.0)

            image_vmin = -6.0e-5
            image_vmax = true_sky.max()
            """
        ),
        md(
            r"""
            ### 9.10.1 干涉阵为什么会“看见细节，却看不见总通量”

            下面这组三图对应的是同一片天空：

            - `true sky`：真实天空亮度分布；
            - `interferometer only`：缺短间距后得到的干涉阵图像；
            - `single-dish only`：单碟总功率图像。

            这个实验里，我们把干涉阵的大尺度转折尺度设在约 `56 arcsec`。这不是严格阵列公式推导，而是一个教学上的等效 `MRS`（maximum recoverable scale）近似，用来直观表示“从这附近开始，大尺度信息会越来越不可靠”。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))

            for ax, image, title in [
                (axes[0], true_sky, "True sky"),
                (axes[1], interferometer_image, "Interferometer only"),
                (axes[2], single_dish_image, "Single-dish only"),
            ]:
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=[coords[0], coords[-1], coords[0], coords[-1]],
                    cmap="magma",
                    vmin=image_vmin,
                    vmax=image_vmax,
                )
                ax.set_title(title)
                ax.set_xlabel("RA offset [arcsec]")
                ax.set_ylabel("Dec offset [arcsec]")
                plt.colorbar(im, ax=ax, shrink=0.82, label="Jy / arcsec^2")

            plt.tight_layout()

            true_total_flux = integrated_flux(true_sky, science_mask, cell_arcsec)
            int_total_flux = integrated_flux(interferometer_image, science_mask, cell_arcsec)
            sd_total_flux = integrated_flux(single_dish_image, science_mask, cell_arcsec)

            print(f"设定的等效最大可恢复尺度 MRS ≈ {mrs_arcsec:.1f} arcsec")
            print(f"干涉阵恢复束 FWHM ≈ {restoring_fwhm_arcsec:.1f} arcsec")
            print(f"单碟束 FWHM ≈ {single_dish_fwhm_arcsec:.1f} arcsec")
            print(f"真实天空在科学区域内的总通量 ≈ {true_total_flux:.3f} Jy")
            print(f"干涉阵单独成像得到的总通量 ≈ {int_total_flux:.3f} Jy")
            print(f"单碟图像得到的总通量 ≈ {sd_total_flux:.3f} Jy")
            """
        ),
        md(
            r"""
            这里最关键的判断是：**干涉阵图像中的“亮源还在”并不意味着通量没有丢。** 真正先消失的往往是大尺度背景、扩展包层和低空间频率结构。
            """
        ),
        md(
            r"""
            ### 9.10.2 missing flux 与 negative bowl 在图像上怎么表现

            缺短间距的后果通常至少有两个：

            - 积分通量显著偏低；
            - 源外围出现系统性的负碗结构。

            下面分别画出径向平均亮度剖面和累计孔径通量曲线。这样可以直接看到：

            - 干涉阵图像在大半径处会被拉到负值附近；
            - 单碟图像在总通量上更接近真实天空，但紧致峰值被明显压低。
            """
        ),
        code(
            """
            radial_bins = np.linspace(0.0, 70.0, 20)
            radial_centers, true_profile = radial_profile(true_sky, radius_grid, radial_bins)
            _, int_profile = radial_profile(interferometer_image, radius_grid, radial_bins)
            _, sd_profile = radial_profile(single_dish_image, radius_grid, radial_bins)

            aperture_radii = np.linspace(6.0, 60.0, 16)
            true_curve = aperture_curve(true_sky, radius_grid, aperture_radii, cell_arcsec)
            int_curve = aperture_curve(interferometer_image, radius_grid, aperture_radii, cell_arcsec)
            sd_curve = aperture_curve(single_dish_image, radius_grid, aperture_radii, cell_arcsec)

            diffuse_true_flux = integrated_flux(true_sky, diffuse_mask, cell_arcsec)
            diffuse_int_flux = integrated_flux(interferometer_image, diffuse_mask, cell_arcsec)
            diffuse_sd_flux = integrated_flux(single_dish_image, diffuse_mask, cell_arcsec)
            bowl_median = np.median(interferometer_image[negative_bowl_mask])

            fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1))

            axes[0].plot(radial_centers, true_profile, color="black", lw=1.8, label="true sky")
            axes[0].plot(radial_centers, int_profile, color="tab:red", lw=1.8, label="interferometer only")
            axes[0].plot(radial_centers, sd_profile, color="tab:blue", lw=1.8, label="single-dish only")
            axes[0].axhline(0.0, color="gray", ls="--", lw=1.0)
            axes[0].set_xlabel("radius [arcsec]")
            axes[0].set_ylabel("mean surface brightness [Jy / arcsec^2]")
            axes[0].set_title("Radial profile")
            axes[0].legend(loc="upper right")

            axes[1].plot(aperture_radii, true_curve, color="black", lw=1.8, label="true sky")
            axes[1].plot(aperture_radii, int_curve, color="tab:red", lw=1.8, label="interferometer only")
            axes[1].plot(aperture_radii, sd_curve, color="tab:blue", lw=1.8, label="single-dish only")
            axes[1].set_xlabel("aperture radius [arcsec]")
            axes[1].set_ylabel("enclosed flux [Jy]")
            axes[1].set_title("Cumulative aperture flux")
            axes[1].legend(loc="lower right")

            plt.tight_layout()

            print(f"扩展成分在真实天空中的积分通量 ≈ {diffuse_true_flux:.3f} Jy")
            print(f"扩展成分在干涉阵图像中的积分通量 ≈ {diffuse_int_flux:.3f} Jy")
            print(f"扩展成分在单碟图像中的积分通量 ≈ {diffuse_sd_flux:.3f} Jy")
            print(f"干涉阵图像负碗区域的中位亮度 ≈ {bowl_median:.2e} Jy / arcsec^2")
            """
        ),
        md(
            r"""
            这就是为什么真实数据处理中，看到负碗时不能只把它当成“去卷积还没收干净”。很多时候，它反映的是**低空间频率本来就没有被测到**。
            """
        ),
        md(
            r"""
            ### 9.10.3 用 feather 把低频交给单碟，把高频交给干涉阵

            `feather` 的核心思想非常直接：

            - 在低空间频率处，更相信单碟；
            - 在高空间频率处，更相信干涉阵；
            - 在中间区域做平滑过渡。

            下面先看一个“balanced feather”结果，也就是这节里推荐作为默认教学展示的组合。
            """
        ),
        code(
            """
            feather_total_flux = integrated_flux(feather_balanced, science_mask, cell_arcsec)
            feather_diffuse_flux = integrated_flux(feather_balanced, diffuse_mask, cell_arcsec)
            feather_compact_flux = integrated_flux(feather_balanced, compact_mask, cell_arcsec)

            fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))

            for ax, image, title in [
                (axes[0, 0], true_sky, "True sky"),
                (axes[0, 1], interferometer_image, "Interferometer only"),
                (axes[1, 0], single_dish_image, "Single-dish only"),
                (axes[1, 1], feather_balanced, "Balanced feather"),
            ]:
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=[coords[0], coords[-1], coords[0], coords[-1]],
                    cmap="magma",
                    vmin=image_vmin,
                    vmax=image_vmax,
                )
                ax.set_title(title)
                ax.set_xlabel("RA offset [arcsec]")
                ax.set_ylabel("Dec offset [arcsec]")
                plt.colorbar(im, ax=ax, shrink=0.82, label="Jy / arcsec^2")

            plt.tight_layout()

            print(f"balanced feather 后的总通量 ≈ {feather_total_flux:.3f} Jy")
            print(f"balanced feather 后的扩展成分通量 ≈ {feather_diffuse_flux:.3f} Jy")
            print(f"balanced feather 后紧致核心 8'' 孔径通量 ≈ {feather_compact_flux:.3f} Jy")
            print(
                f"干涉阵单独成像只保留了总通量的 {100.0 * int_total_flux / true_total_flux:.1f}% ，"
                f"balanced feather 恢复到 {100.0 * feather_total_flux / true_total_flux:.1f}%"
            )
            print(
                f"balanced feather 的中心峰值约为单碟图像的 "
                f"{feather_balanced[npix // 2, npix // 2] / single_dish_image[npix // 2, npix // 2]:.2f} 倍"
            )
            """
        ),
        md(
            r"""
            这里要注意一个非常重要的专业边界：`feather` 确实能把低频补回来，但它**不是联合去卷积的完全替代品**。紧致结构的最终峰值、PSF 一致性和不同数据集之间的相对标定，依然会限制结果质量。
            """
        ),
        md(
            r"""
            ### 9.10.4 feather 权重也有取舍：保守、平衡、激进

            如果让单碟在频域里占得太少，缺失的大尺度仍会补不回来；如果让单碟占得太多，又会把中尺度和小尺度结构拉得过平。

            下面比较三种过渡尺度：

            - conservative：单碟只负责更低频的一小段；
            - balanced：这节推荐的默认方案；
            - aggressive：让单碟在更宽的频率范围内参与。
            """
        ),
        code(
            """
            k_axis = np.linspace(0.0, np.max(k_radius), 400)
            w_cons_axis = np.exp(-(k_axis / conservative_join) ** 2)
            w_bal_axis = np.exp(-(k_axis / balanced_join) ** 2)
            w_aggr_axis = np.exp(-(k_axis / aggressive_join) ** 2)

            strategy_images = {
                "conservative": feather_conservative,
                "balanced": feather_balanced,
                "aggressive": feather_aggressive,
            }
            strategy_total = np.array(
                [
                    integrated_flux(strategy_images[name], science_mask, cell_arcsec)
                    for name in ["conservative", "balanced", "aggressive"]
                ]
            )
            strategy_peak = np.array(
                [strategy_images[name][npix // 2, npix // 2] for name in ["conservative", "balanced", "aggressive"]]
            )

            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))

            axes[0].plot(
                k_axis,
                k_axis / (k_axis + k_short),
                color="black",
                lw=1.8,
                label="interferometer transfer",
            )
            axes[0].plot(k_axis, w_cons_axis, color="tab:blue", lw=1.8, label="single-dish weight: conservative")
            axes[0].plot(k_axis, w_bal_axis, color="tab:green", lw=1.8, label="single-dish weight: balanced")
            axes[0].plot(k_axis, w_aggr_axis, color="tab:red", lw=1.8, label="single-dish weight: aggressive")
            axes[0].set_xlabel("spatial frequency [cycles / arcsec]")
            axes[0].set_ylabel("weight / response")
            axes[0].set_title("Frequency-domain weighting")
            axes[0].legend(loc="upper right")

            labels = ["conservative", "balanced", "aggressive"]
            x_pos = np.arange(len(labels))
            width = 0.36
            axes[1].bar(
                x_pos - width / 2,
                100.0 * strategy_total / true_total_flux,
                width=width,
                color="tab:orange",
                label="total flux recovery [% of true]",
            )
            axes[1].bar(
                x_pos + width / 2,
                100.0 * strategy_peak / interferometer_image[npix // 2, npix // 2],
                width=width,
                color="tab:purple",
                label="central peak [% of interferometer-only]",
            )
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(labels, rotation=10)
            axes[1].set_ylabel("relative percentage [%]")
            axes[1].set_title("Trade-off between flux recovery and compactness")
            axes[1].legend(loc="upper right")

            plt.tight_layout()

            for name in ["conservative", "balanced", "aggressive"]:
                image = strategy_images[name]
                total_flux = integrated_flux(image, science_mask, cell_arcsec)
                diffuse_flux = integrated_flux(image, diffuse_mask, cell_arcsec)
                peak_value = image[npix // 2, npix // 2]
                print(
                    f"{name} feather: 总通量 = {total_flux:.3f} Jy, "
                    f"扩展成分 = {diffuse_flux:.3f} Jy, "
                    f"中心峰值 = {peak_value:.4f} Jy / arcsec^2"
                )
            """
        ),
        md(
            r"""
            这个比较想传达的不是“存在一个永远最优的 join 参数”，而是一个更实际的工作习惯：**你必须根据科学目标判断，是更在意总通量恢复，还是更在意紧致结构和分辨率。**
            """
        ),
        md(
            r"""
            ### 9.10.5 与真实软件流程的对应

            若把这个教学实验映射回真实处理环境，最常见的关键词大致包括：

            - `MRS`、最短基线和缺零间距；
            - `TP`、`single-dish`、`total power`；
            - `feather`、`sdintimaging` 或其他联合去卷积方法；
            - `7m + 12m + TP` 这种多阵列联合；
            - flux scale matching、beam matching 与频谱窗口一致性检查。

            真正的专业处理里，你还需要继续检查：

            - 单碟与干涉阵是否在同一亮度标尺上；
            - 两边的 beam、像素网格和单位是否兼容；
            - 是否应该只做 feather，还是要做更完整的 joint deconvolution；
            - 光谱线数据里，联合成像是否还要同时考虑 continuum subtraction 与每通道噪声差异。

            因此，这个 notebook 的核心结论可以概括为一句话：**短间距问题不是成像后的“小修小补”，而是会直接决定你能否可靠测到大尺度真实通量。**
            """
        ),
    ],
    "9_11_polarization_calibration_and_rm_synthesis.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.10 短间距、单碟与联合成像](9_10_short_spacing_and_feather.ipynb)
                * 下一节： [9.12 宽场方向相关成像](9_12_direction_dependent_widefield_imaging.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.11 偏振高级实践：交叉手相位校准与 RM synthesis

            `9.9` 已经把偏振成像的入口建立起来了，但真实数据处理往往不会停在 “画出 `Q/U` 图和做一条 RM 直线拟合” 这一步。继续往下走，你很快就会遇到两个更专业、也更常见的问题：

            - 如果交叉手相位（`cross-hand phase`、`XY phase` 或 `R-L phase`）没有校准好，整个偏振角会被系统性旋转，而且这种旋转还可能随频率变化；
            - 如果真实源在 Faraday 深度上并不只有一个分量，那么简单的 $\chi(\lambda^2)$ 直线拟合就会失效，这时需要转向 `RM synthesis`。

            这一节把这两件事连成一条更接近真实工作的判断链：

            - 先用一个已知偏振校准源求解频率依赖的交叉手相位；
            - 再把这个解应用到一个 Faraday-thin 目标上，比较校准前后的 RM 偏差；
            - 最后构造一个双 Faraday 分量目标，说明为什么单一 RM 拟合不够，以及 `RM synthesis` 为什么有必要。
            """
        ),
        md("***"),
        code(
            """
            c_light = 0.299792458


            def complex_polarization_spectrum(
                freqs_ghz, lambda2, ref_freq_ghz, lambda2_ref, components
            ):
                spectrum = np.zeros(freqs_ghz.size, dtype=complex)
                for comp in components:
                    amp = comp["p_ref"] * (freqs_ghz / ref_freq_ghz) ** comp.get("alpha", 0.0)
                    phase = 2.0 * (
                        np.deg2rad(comp["chi0_deg"]) + comp["phi"] * (lambda2 - lambda2_ref)
                    )
                    spectrum += amp * np.exp(1j * phase)
                return spectrum


            def wrap_evpa_deg(angle_deg):
                return ((angle_deg + 90.0) % 180.0) - 90.0


            def evpa_deg(spectrum):
                return wrap_evpa_deg(0.5 * np.rad2deg(np.angle(spectrum)))


            def fit_linear_rm(lambda2, spectrum, min_p):
                p_spec = np.abs(spectrum)
                valid = p_spec > min_p
                chi = np.full_like(lambda2, np.nan, dtype=float)
                model = np.full_like(lambda2, np.nan, dtype=float)
                if np.count_nonzero(valid) < 3:
                    return np.nan, np.nan, chi, model, valid, np.nan
                chi_valid = 0.5 * np.unwrap(np.angle(spectrum[valid]))
                slope, intercept = np.polyfit(lambda2[valid], chi_valid, 1)
                chi[valid] = chi_valid
                model[valid] = intercept + slope * lambda2[valid]
                residual_rms_deg = np.sqrt(
                    np.mean(np.rad2deg(chi_valid - model[valid]) ** 2)
                )
                return intercept, slope, chi, model, valid, residual_rms_deg


            def rm_synthesis(lambda2, spectrum, phi_axis, weights=None):
                if weights is None:
                    weights = np.ones_like(lambda2)
                lambda2_0 = np.sum(weights * lambda2) / np.sum(weights)
                phase = np.exp(-2j * np.outer(phi_axis, lambda2 - lambda2_0))
                fdf = phase @ (weights * spectrum) / np.sum(weights)
                rmsf = phase @ weights / np.sum(weights)
                return lambda2_0, fdf, rmsf


            def extract_faraday_peaks(phi_axis, amplitude, min_sep, nmax):
                peaks = []
                for idx in np.argsort(amplitude)[::-1]:
                    if not peaks or all(abs(phi_axis[idx] - peak[0]) > min_sep for peak in peaks):
                        peaks.append((phi_axis[idx], amplitude[idx]))
                    if len(peaks) >= nmax:
                        break
                return peaks


            def rm_resolution(lambda2):
                return 2.0 * np.sqrt(3.0) / (lambda2.max() - lambda2.min())


            freqs_ghz = np.linspace(1.0, 2.0, 64)
            lambda2 = (c_light / freqs_ghz) ** 2
            ref_index = int(np.argmin(np.abs(freqs_ghz - np.median(freqs_ghz))))
            ref_freq_ghz = freqs_ghz[ref_index]
            lambda2_ref = lambda2[ref_index]

            calibrator_true = complex_polarization_spectrum(
                freqs_ghz,
                lambda2,
                ref_freq_ghz,
                lambda2_ref,
                [{"p_ref": 0.075, "chi0_deg": 22.0, "phi": 55.0, "alpha": -0.2}],
            )
            simple_target_true = complex_polarization_spectrum(
                freqs_ghz,
                lambda2,
                ref_freq_ghz,
                lambda2_ref,
                [{"p_ref": 0.052, "chi0_deg": -12.0, "phi": 85.0, "alpha": -0.5}],
            )
            complex_target_true = complex_polarization_spectrum(
                freqs_ghz,
                lambda2,
                ref_freq_ghz,
                lambda2_ref,
                [
                    {"p_ref": 0.040, "chi0_deg": 10.0, "phi": 35.0, "alpha": -0.3},
                    {"p_ref": 0.040, "chi0_deg": -20.0, "phi": 165.0, "alpha": -0.6},
                ],
            )

            crosshand_phase_true_deg = (
                35.0
                + 75.0 * (freqs_ghz - ref_freq_ghz)
                + 18.0 * np.sin(2.0 * np.pi * (freqs_ghz - 1.0) / 0.45)
            )
            crosshand_phase_true_rad = np.deg2rad(crosshand_phase_true_deg)
            noise_std = 6.0e-4

            calibrator_obs = calibrator_true * np.exp(1j * crosshand_phase_true_rad)
            calibrator_obs += noise_std * (
                RNG.normal(size=freqs_ghz.size) + 1j * RNG.normal(size=freqs_ghz.size)
            )

            simple_target_obs = simple_target_true * np.exp(1j * crosshand_phase_true_rad)
            simple_target_obs += noise_std * (
                RNG.normal(size=freqs_ghz.size) + 1j * RNG.normal(size=freqs_ghz.size)
            )

            complex_target_obs = complex_target_true * np.exp(1j * crosshand_phase_true_rad)
            complex_target_obs += noise_std * (
                RNG.normal(size=freqs_ghz.size) + 1j * RNG.normal(size=freqs_ghz.size)
            )

            crosshand_phase_solution_rad = np.unwrap(
                np.angle(calibrator_obs * np.conj(calibrator_true))
            )
            crosshand_phase_residual_deg = np.rad2deg(
                np.angle(
                    np.exp(1j * (crosshand_phase_solution_rad - np.unwrap(crosshand_phase_true_rad)))
                )
            )

            calibrator_corrected = calibrator_obs * np.exp(-1j * crosshand_phase_solution_rad)
            simple_target_corrected = simple_target_obs * np.exp(-1j * crosshand_phase_solution_rad)
            complex_target_corrected = complex_target_obs * np.exp(-1j * crosshand_phase_solution_rad)

            phi_axis = np.linspace(-300.0, 300.0, 1801)
            broad_mask = np.ones_like(freqs_ghz, dtype=bool)
            narrow_mask = (freqs_ghz >= 1.30) & (freqs_ghz <= 1.55)
            """
        ),
        md(
            r"""
            ### 9.11.1 先把交叉手相位校准好，否则偏振角会整段被旋转

            对线偏振复量 $P = Q + iU$ 来说，频率依赖的交叉手相位误差本质上就是一个复平面旋转项。如果不先把它解掉：

            - `EVPA` 会被系统性扭曲；
            - 单一 RM 拟合会出现偏差；
            - `RM synthesis` 的相位参考也会受到污染。

            下面用一个已知偏振校准源来解这个相位项，并比较真实相位偏移、求解结果和校准前后的 `EVPA`。
            """
        ),
        code(
            """
            calibrator_fit_true = fit_linear_rm(lambda2, calibrator_true, min_p=0.002)
            calibrator_fit_obs = fit_linear_rm(lambda2, calibrator_obs, min_p=0.002)
            calibrator_fit_corr = fit_linear_rm(lambda2, calibrator_corrected, min_p=0.002)

            fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))

            axes[0].plot(freqs_ghz, crosshand_phase_true_deg, color="black", lw=1.8, label="true cross-hand phase")
            axes[0].plot(
                freqs_ghz,
                np.rad2deg(crosshand_phase_solution_rad),
                color="tab:blue",
                lw=1.8,
                label="solved from calibrator",
            )
            axes[0].set_xlabel("frequency [GHz]")
            axes[0].set_ylabel("phase [deg]")
            axes[0].set_title("Frequency-dependent cross-hand phase")
            axes[0].legend(loc="upper left")

            axes[1].plot(freqs_ghz, evpa_deg(calibrator_true), color="black", lw=1.8, label="true EVPA")
            axes[1].plot(freqs_ghz, evpa_deg(calibrator_obs), color="tab:red", lw=1.8, label="observed before correction")
            axes[1].plot(freqs_ghz, evpa_deg(calibrator_corrected), color="tab:blue", lw=1.8, label="after correction")
            axes[1].set_xlabel("frequency [GHz]")
            axes[1].set_ylabel("EVPA [deg]")
            axes[1].set_title("Calibrator EVPA before and after correction")
            axes[1].legend(loc="best")

            plt.tight_layout()

            print(
                f"交叉手相位在这段频带内约变化 "
                f"{crosshand_phase_true_deg.min():.1f} 到 {crosshand_phase_true_deg.max():.1f} deg"
            )
            print(
                f"交叉手相位求解残差 RMS ≈ {np.std(crosshand_phase_residual_deg):.2f} deg"
            )
            print(
                f"校准源真实 RM = +55.0 rad/m^2，"
                f"校准前估计 = {calibrator_fit_obs[1]:+.1f}，"
                f"校准后估计 = {calibrator_fit_corr[1]:+.1f}"
            )
            """
        ),
        md(
            r"""
            这里要建立的专业判断是：**即使你已经有了 `Q/U` 数据，偏振角也并不自动可信。** 交叉手相位没有处理好，后面一整条偏振分析链都会偏掉。
            """
        ),
        md(
            r"""
            ### 9.11.2 把解应用到 Faraday-thin 目标：RM 偏差会明显收敛

            下面构造一个单一 Faraday 分量目标。它本来满足简单的 $\chi(\lambda^2)$ 线性关系，所以非常适合拿来检验交叉手相位校准有没有真正生效。
            """
        ),
        code(
            """
            simple_fit_true = fit_linear_rm(lambda2, simple_target_true, min_p=0.002)
            simple_fit_obs = fit_linear_rm(lambda2, simple_target_obs, min_p=0.002)
            simple_fit_corr = fit_linear_rm(lambda2, simple_target_corrected, min_p=0.002)

            fig, ax = plt.subplots(figsize=(7.8, 4.2))

            ax.plot(
                lambda2[simple_fit_true[4]],
                np.rad2deg(simple_fit_true[2][simple_fit_true[4]]),
                "o",
                color="black",
                label="true EVPA",
            )
            ax.plot(
                lambda2[simple_fit_obs[4]],
                np.rad2deg(simple_fit_obs[2][simple_fit_obs[4]]),
                "o",
                color="tab:red",
                label=f"before correction, RM={simple_fit_obs[1]:+.1f}",
            )
            ax.plot(
                lambda2[simple_fit_corr[4]],
                np.rad2deg(simple_fit_corr[2][simple_fit_corr[4]]),
                "o",
                color="tab:blue",
                label=f"after correction, RM={simple_fit_corr[1]:+.1f}",
            )
            ax.plot(lambda2[simple_fit_true[4]], np.rad2deg(simple_fit_true[3][simple_fit_true[4]]), color="black", lw=1.6)
            ax.plot(lambda2[simple_fit_obs[4]], np.rad2deg(simple_fit_obs[3][simple_fit_obs[4]]), color="tab:red", lw=1.6)
            ax.plot(lambda2[simple_fit_corr[4]], np.rad2deg(simple_fit_corr[3][simple_fit_corr[4]]), color="tab:blue", lw=1.6)
            ax.set_xlabel("lambda^2 [m^2]")
            ax.set_ylabel("EVPA [deg]")
            ax.set_title("Faraday-thin target before and after cross-hand correction")
            ax.legend(loc="upper right")

            plt.tight_layout()

            print(
                f"simple target 真实 RM = +85.0 rad/m^2，"
                f"校准前估计 = {simple_fit_obs[1]:+.1f}，"
                f"校准后估计 = {simple_fit_corr[1]:+.1f}"
            )
            print(
                f"校准后 RM 偏差约从 {abs(simple_fit_obs[1] - 85.0):.1f} "
                f"降到 {abs(simple_fit_corr[1] - 85.0):.1f} rad/m^2"
            )
            """
        ),
        md(
            r"""
            这一步说明：**交叉手相位校准解决的不只是“角度好不好看”的问题，而是会直接影响你最终报告的 RM 数值。**
            """
        ),
        md(
            r"""
            ### 9.11.3 一旦源里有多个 Faraday 分量，单一 RM 拟合就会变得不可靠

            下面把目标换成两个 Faraday-thin 分量的叠加。这里即使交叉手相位已经校准好，$\chi(\lambda^2)$ 也不再严格线性：

            - `EVPA` 会出现明显弯折；
            - 线偏振强度会随频率振荡；
            - 用一条直线去拟合时，得到的 RM 往往没有清晰物理意义。
            """
        ),
        code(
            """
            complex_fit_corr = fit_linear_rm(lambda2, complex_target_corrected, min_p=0.005)

            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))

            axes[0].plot(
                lambda2[complex_fit_corr[4]],
                np.rad2deg(complex_fit_corr[2][complex_fit_corr[4]]),
                "o",
                color="tab:purple",
                label="corrected EVPA",
            )
            axes[0].plot(
                lambda2[complex_fit_corr[4]],
                np.rad2deg(complex_fit_corr[3][complex_fit_corr[4]]),
                color="black",
                lw=1.8,
                label=f"single-RM fit, RM={complex_fit_corr[1]:+.1f}",
            )
            axes[0].set_xlabel("lambda^2 [m^2]")
            axes[0].set_ylabel("EVPA [deg]")
            axes[0].set_title("A single RM no longer fits well")
            axes[0].legend(loc="upper right")

            axes[1].plot(freqs_ghz, np.abs(complex_target_true), color="black", lw=1.8, label="true |P|")
            axes[1].plot(
                freqs_ghz,
                np.abs(complex_target_corrected),
                color="tab:blue",
                lw=1.8,
                label="corrected observed |P|",
            )
            axes[1].set_xlabel("frequency [GHz]")
            axes[1].set_ylabel("|Q+iU|")
            axes[1].set_title("Frequency-dependent interference of Faraday components")
            axes[1].legend(loc="upper right")

            plt.tight_layout()

            print(
                f"双分量目标若仍做单一 RM 拟合，会得到 RM ≈ {complex_fit_corr[1]:+.1f} rad/m^2"
            )
            print(
                f"但这条直线的 EVPA 残差 RMS 仍有 ≈ {complex_fit_corr[5]:.1f} deg"
            )
            """
        ),
        md(
            r"""
            这正是 `RM synthesis` 的工作背景。它不是为了“更复杂而复杂”，而是因为单一 RM 模型已经不足以描述真实偏振频谱。
            """
        ),
        md(
            r"""
            ### 9.11.4 用 RM synthesis 在 Faraday 深度空间里分开多分量

            现在对同一个双分量目标做 `RM synthesis`。我们同时比较：

            - 全带宽情形：Faraday 深度分辨率较好，两个分量可以被区分；
            - 窄带宽情形：Faraday 深度分辨率明显变差，两个分量会混成一个宽峰。
            """
        ),
        code(
            """
            _, fdf_broad, rmsf_broad = rm_synthesis(
                lambda2[broad_mask], complex_target_corrected[broad_mask], phi_axis
            )
            _, fdf_narrow, rmsf_narrow = rm_synthesis(
                lambda2[narrow_mask], complex_target_corrected[narrow_mask], phi_axis
            )

            broad_peaks = extract_faraday_peaks(phi_axis, np.abs(fdf_broad), min_sep=35.0, nmax=3)
            narrow_peaks = extract_faraday_peaks(phi_axis, np.abs(fdf_narrow), min_sep=35.0, nmax=3)
            broad_resolution = rm_resolution(lambda2[broad_mask])
            narrow_resolution = rm_resolution(lambda2[narrow_mask])

            fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))

            axes[0].plot(phi_axis, np.abs(fdf_broad), color="tab:blue", lw=1.8)
            axes[0].axvline(35.0, color="gray", ls="--", lw=1.0)
            axes[0].axvline(165.0, color="gray", ls="--", lw=1.0)
            axes[0].set_xlabel("Faraday depth [rad / m^2]")
            axes[0].set_ylabel("|F(phi)|")
            axes[0].set_title("Broad-band RM synthesis")

            axes[1].plot(phi_axis, np.abs(fdf_narrow), color="tab:red", lw=1.8)
            axes[1].axvline(35.0, color="gray", ls="--", lw=1.0)
            axes[1].axvline(165.0, color="gray", ls="--", lw=1.0)
            axes[1].set_xlabel("Faraday depth [rad / m^2]")
            axes[1].set_ylabel("|F(phi)|")
            axes[1].set_title("Narrow-band RM synthesis")

            axes[2].plot(phi_axis, np.abs(rmsf_broad), color="tab:blue", lw=1.8, label=f"broad RMSF, FWHM~{broad_resolution:.1f}")
            axes[2].plot(phi_axis, np.abs(rmsf_narrow), color="tab:red", lw=1.8, label=f"narrow RMSF, FWHM~{narrow_resolution:.1f}")
            axes[2].set_xlim(-250.0, 250.0)
            axes[2].set_xlabel("Faraday depth [rad / m^2]")
            axes[2].set_ylabel("|RMSF|")
            axes[2].set_title("Faraday-depth resolution")
            axes[2].legend(loc="upper right")

            plt.tight_layout()

            print(
                "全带宽 RM synthesis 的两个主峰约位于 "
                + ", ".join([f"{peak[0]:+.1f}" for peak in broad_peaks[:2]])
                + " rad/m^2"
            )
            print(
                "窄带宽 RM synthesis 的主峰约位于 "
                + ", ".join([f"{peak[0]:+.1f}" for peak in narrow_peaks[:2]])
                + " rad/m^2"
            )
            print(f"全带宽 Faraday 深度分辨率约 ≈ {broad_resolution:.1f} rad/m^2")
            print(f"窄带宽 Faraday 深度分辨率约 ≈ {narrow_resolution:.1f} rad/m^2")
            """
        ),
        md(
            r"""
            这里最重要的结论有两个：

            - `RM synthesis` 的能力不仅取决于信噪比，也强烈取决于 $\lambda^2$ 覆盖；
            - 带宽一旦变窄，Faraday 深度分辨率会迅速恶化，多个分量就可能在 Faraday 空间里混成一个结构。
            """
        ),
        md(
            r"""
            ### 9.11.5 与真实软件流程的对应

            若把这个教学实验映射回真实偏振数据处理，通常会落到下面这些动作上：

            - 在总强度校准之后，继续求解 cross-hand delay、`XY phase` / `R-L phase` 和 `D-term`；
            - 用偏振校准源把交叉手相位随频率的旋转校准干净；
            - 对 Faraday-thin 源，可以先做 `RM fit` 作为快速入口；
            - 对多分量或 Faraday-thick 源，应进入 `RM synthesis`、`RM-CLEAN`、Faraday depth 建模或 `QU-fitting`；
            - 始终检查带宽去极化、beam depolarization、主波束偏振响应和频率采样是否足以支撑物理解释。

            这个 notebook 传达的专业判断是：**交叉手相位校准和 RM synthesis 不是两个孤立技巧，而是同一条偏振分析链上前后相接的关键环节。**
            """
        ),
    ],
    "9_12_direction_dependent_widefield_imaging.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.11 交叉手相位校准与 RM synthesis](9_11_polarization_calibration_and_rm_synthesis.ipynb)
                * 下一节： [9.13 高级谱线分析](9_13_advanced_spectral_line_analysis.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.12 宽场方向相关成像：`w-term`、faceting 与 PB-aware mosaic

            `9.8` 已经把宽带/宽场成像的第一层逻辑建立起来了，但真实宽场处理往往还会继续遇到两类更棘手的方向相关问题：

            - 非共面基线带来的 `w-term`，会让离轴源在简单二维成像里出现失真、展宽甚至位置和峰值偏差；
            - 主波束随视场方向变化，多个指向或长时间覆盖的宽场数据如果不做 beam-aware 处理，就会在拼接结果里留下明显的通量偏差。

            这一节不试图复刻工业级 `w-projection`、`A-projection` 或 `AW-projection` 实现，而是用两个可以直接运行的最小实验，把现代宽场方向相关成像的判断链建立起来：

            - 为什么忽略 `w-term` 会让离轴源被严重压扁；
            - 为什么 faceting 能在局部恢复离轴源；
            - 为什么多指向拼接不能只做简单平均；
            - 为什么 `PB-aware mosaic` 的通量恢复往往伴随着边缘噪声放大。
            """
        ),
        md("***"),
        code(
            """
            def n_term(l, m):
                return np.sqrt(np.clip(1.0 - l**2 - m**2, 1.0e-12, None))


            def simulate_visibilities(u, v, w, sources, noise_std=0.0):
                vis = np.zeros(u.size, dtype=complex)
                for src in sources:
                    n_src = n_term(src["l"], src["m"])
                    phase = -2.0 * np.pi * (
                        u * src["l"] + v * src["m"] + w * (n_src - 1.0)
                    )
                    vis += (src["flux"] / n_src) * np.exp(1j * phase)
                if noise_std > 0.0:
                    vis += noise_std * (
                        RNG.normal(size=u.size) + 1j * RNG.normal(size=u.size)
                    )
                return vis


            def dirty_image_from_visibilities(
                u, v, w, vis, l_grid, m_grid, use_w_term=False
            ):
                ll, mm = np.meshgrid(l_grid, m_grid)
                image = np.zeros_like(ll, dtype=complex)
                nn = n_term(ll, mm)
                for uk, vk, wk, vis_k in zip(u, v, w, vis):
                    phase = uk * ll + vk * mm
                    if use_w_term:
                        phase = phase + wk * (nn - 1.0)
                    image += vis_k * np.exp(2j * np.pi * phase)
                image /= u.size
                return np.real(image)


            def phase_shift_visibilities(vis, u, v, w, l0, m0):
                n0 = n_term(l0, m0)
                shift = np.exp(2j * np.pi * (u * l0 + v * m0 + w * (n0 - 1.0)))
                return vis * shift


            def local_peak(image, l_grid, m_grid, l0, m0, radius):
                ll, mm = np.meshgrid(l_grid, m_grid)
                mask = ((ll - l0) ** 2 + (mm - m0) ** 2) <= radius**2
                return np.max(image[mask])


            def normalized_gaussian_map(x, y, x0, y0, sigma_x, sigma_y):
                profile = np.exp(
                    -0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
                )
                profile /= profile.max()
                return profile


            def primary_beam_gain_2d(x, y, x0, y0, fwhm):
                radius2 = (x - x0) ** 2 + (y - y0) ** 2
                return np.exp(-4.0 * np.log(2.0) * radius2 / (fwhm**2))


            def gaussian_kernel(npix, sigma_pix):
                coords = np.arange(npix) - npix // 2
                xx, yy = np.meshgrid(coords, coords)
                kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
                kernel /= kernel.sum()
                return kernel


            def fft_convolve_same(image, kernel):
                image_ft = np.fft.fft2(image)
                kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
                return np.real(np.fft.ifft2(image_ft * kernel_ft))


            # --- experiment 1: non-coplanar baselines and faceting ---
            widefield_sources = [
                {"name": "center", "flux": 1.00, "l": 0.000, "m": 0.000},
                {"name": "east_arc", "flux": 0.55, "l": 0.032, "m": -0.012},
                {"name": "northwest", "flux": 0.42, "l": -0.027, "m": 0.031},
            ]

            nvis = 420
            rho = RNG.uniform(30.0, 650.0, size=nvis)
            phi = RNG.uniform(0.0, 2.0 * np.pi, size=nvis)
            u_coords = rho * np.cos(phi)
            v_coords = rho * np.sin(phi)
            w_coords = RNG.uniform(-1200.0, 1200.0, size=nvis)
            vis_noise_std = 0.015

            widefield_vis = simulate_visibilities(
                u_coords, v_coords, w_coords, widefield_sources, noise_std=vis_noise_std
            )

            full_l_grid = np.linspace(-0.055, 0.055, 72)
            full_m_grid = np.linspace(-0.055, 0.055, 72)
            naive_dirty = dirty_image_from_visibilities(
                u_coords, v_coords, w_coords, widefield_vis, full_l_grid, full_m_grid, use_w_term=False
            )
            widefield_dirty = dirty_image_from_visibilities(
                u_coords, v_coords, w_coords, widefield_vis, full_l_grid, full_m_grid, use_w_term=True
            )
            wterm_error_map = naive_dirty - widefield_dirty

            source_east = widefield_sources[1]
            facet_half_width = 0.010
            facet_local_l = np.linspace(-facet_half_width, facet_half_width, 48)
            facet_local_m = np.linspace(-facet_half_width, facet_half_width, 48)
            shifted_vis = phase_shift_visibilities(
                widefield_vis, u_coords, v_coords, w_coords, source_east["l"], source_east["m"]
            )
            facet_dirty = dirty_image_from_visibilities(
                u_coords, v_coords, w_coords, shifted_vis, facet_local_l, facet_local_m, use_w_term=False
            )

            center_peak_naive = local_peak(naive_dirty, full_l_grid, full_m_grid, 0.0, 0.0, 0.004)
            center_peak_wide = local_peak(widefield_dirty, full_l_grid, full_m_grid, 0.0, 0.0, 0.004)
            east_peak_naive = local_peak(
                naive_dirty, full_l_grid, full_m_grid, source_east["l"], source_east["m"], 0.0045
            )
            east_peak_wide = local_peak(
                widefield_dirty, full_l_grid, full_m_grid, source_east["l"], source_east["m"], 0.0045
            )
            east_peak_facet = local_peak(facet_dirty, facet_local_l, facet_local_m, 0.0, 0.0, 0.003)

            extent_full_deg = [
                np.rad2deg(full_l_grid[0]),
                np.rad2deg(full_l_grid[-1]),
                np.rad2deg(full_m_grid[0]),
                np.rad2deg(full_m_grid[-1]),
            ]
            extent_facet_deg = [
                np.rad2deg(facet_local_l[0]),
                np.rad2deg(facet_local_l[-1]),
                np.rad2deg(facet_local_m[0]),
                np.rad2deg(facet_local_m[-1]),
            ]

            # --- experiment 2: beam-aware mosaic combination ---
            mosaic_npix = 96
            mosaic_coords_arcmin = np.linspace(-40.0, 40.0, mosaic_npix)
            x_arcmin, y_arcmin = np.meshgrid(mosaic_coords_arcmin, mosaic_coords_arcmin)

            mosaic_true_sky = (
                0.80 * normalized_gaussian_map(x_arcmin, y_arcmin, -18.0, 10.0, 7.0, 5.0)
                + 1.00 * normalized_gaussian_map(x_arcmin, y_arcmin, 0.0, 0.0, 6.0, 6.0)
                + 0.55 * normalized_gaussian_map(x_arcmin, y_arcmin, 20.0, -12.0, 8.0, 4.0)
                + 0.22 * normalized_gaussian_map(x_arcmin, y_arcmin, 28.0, 22.0, 4.0, 4.0)
            )

            mosaic_beam = gaussian_kernel(mosaic_npix, sigma_pix=1.3)
            pb_fwhm_arcmin = 36.0
            pb_1 = primary_beam_gain_2d(x_arcmin, y_arcmin, -12.0, 0.0, pb_fwhm_arcmin)
            pb_2 = primary_beam_gain_2d(x_arcmin, y_arcmin, 12.0, 0.0, pb_fwhm_arcmin)
            mosaic_noise_std = 8.0e-4

            pointing_1 = fft_convolve_same(mosaic_true_sky * pb_1, mosaic_beam)
            pointing_2 = fft_convolve_same(mosaic_true_sky * pb_2, mosaic_beam)
            pointing_1 += mosaic_noise_std * RNG.normal(size=(mosaic_npix, mosaic_npix))
            pointing_2 += mosaic_noise_std * RNG.normal(size=(mosaic_npix, mosaic_npix))

            naive_mosaic = 0.5 * (pointing_1 + pointing_2)
            pb_weight_sum = pb_1**2 + pb_2**2
            pb_weighted_mosaic = (
                pb_1 * pointing_1 + pb_2 * pointing_2
            ) / np.maximum(pb_weight_sum, 1.0e-6)

            mosaic_science_mask = (x_arcmin**2 + y_arcmin**2) <= 32.0**2
            outer_source_mask = ((x_arcmin - 28.0) ** 2 + (y_arcmin - 22.0) ** 2) <= 4.0**2
            mosaic_mean_abs_error_naive = np.mean(
                np.abs(naive_mosaic[mosaic_science_mask] - mosaic_true_sky[mosaic_science_mask])
            )
            mosaic_mean_abs_error_weighted = np.mean(
                np.abs(
                    pb_weighted_mosaic[mosaic_science_mask]
                    - mosaic_true_sky[mosaic_science_mask]
                )
            )
            outer_peak_true = np.max(mosaic_true_sky[outer_source_mask])
            outer_peak_naive = np.max(naive_mosaic[outer_source_mask])
            outer_peak_weighted = np.max(pb_weighted_mosaic[outer_source_mask])
            center_idx = np.argmin(np.abs(mosaic_coords_arcmin - 0.0))
            outer_x_idx = np.argmin(np.abs(mosaic_coords_arcmin - 28.0))
            outer_y_idx = np.argmin(np.abs(mosaic_coords_arcmin - 22.0))
            mosaic_noise_ratio = np.sqrt(
                pb_weight_sum[center_idx, center_idx] / pb_weight_sum[outer_y_idx, outer_x_idx]
            )
            """
        ),
        md(
            r"""
            ### 9.12.1 `w-term` 为什么会在宽场里压坏离轴源

            对非共面基线来说，可见度相位里除了熟悉的 $u l + v m$ 项，还会多出和 $w$ 有关的附加项。离轴角越大，这个项越不能忽略。

            下面先做一个最小实验：

            - 用含有三个点源的宽场天空生成 `uvw` 可见度；
            - 先按“普通二维成像”去做，完全忽略 `w-term`；
            - 再用带 `w-term` 的成像公式重建同一片天空。

            这样可以直接看到：中心源通常还凑合，但离轴源会在 naive 图像里被明显压暗和扭曲。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

            for ax, image, title in [
                (axes[0], naive_dirty, "Naive dirty image without w-term"),
                (axes[1], widefield_dirty, "Wide-field dirty image with w-term"),
                (axes[2], wterm_error_map, "Naive minus wide-field image"),
            ]:
                cmap = "coolwarm" if title == "Naive minus wide-field image" else "inferno"
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=extent_full_deg,
                    cmap=cmap,
                )
                ax.set_title(title)
                ax.set_xlabel("field angle l [deg]")
                ax.set_ylabel("field angle m [deg]")
                plt.colorbar(im, ax=ax, shrink=0.82)

            plt.tight_layout()

            print(f"中心源峰值：忽略 w-term ≈ {center_peak_naive:.3f}，宽场校正后 ≈ {center_peak_wide:.3f}")
            print(f"东侧离轴源峰值：忽略 w-term ≈ {east_peak_naive:.3f}，宽场校正后 ≈ {east_peak_wide:.3f}")
            print(
                f"若忽略 w-term，东侧离轴源只恢复了宽场校正峰值的 "
                f"{100.0 * east_peak_naive / east_peak_wide:.1f}%"
            )
            """
        ),
        md(
            r"""
            这一步最重要的判断是：**宽场里的问题并不总能靠更深的 CLEAN 自动补回来。** 如果相位模型本身就少了 `w-term`，离轴结构的保真度会在成像前就已经受损。
            """
        ),
        md(
            r"""
            ### 9.12.2 faceting 为什么能在局部把离轴源救回来

            faceting 的核心思想是：不要强迫整片天空都共享同一个相位中心，而是把感兴趣的宽场区域切成若干局部小块，在每个小块里重新定相和成像。

            下面把相位中心移到东侧离轴源附近，只看一个小 facet。这个做法当然不是完整宽场算法，但它很适合作为 `w-projection` 和 faceting 思想的训练入口。
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

            east_window = [
                np.rad2deg(source_east["l"] - 0.012),
                np.rad2deg(source_east["l"] + 0.012),
                np.rad2deg(source_east["m"] - 0.012),
                np.rad2deg(source_east["m"] + 0.012),
            ]

            axes[0].imshow(
                naive_dirty,
                origin="lower",
                extent=extent_full_deg,
                cmap="inferno",
            )
            axes[0].set_xlim(east_window[0], east_window[1])
            axes[0].set_ylim(east_window[2], east_window[3])
            axes[0].set_title("Naive full-field image around off-axis source")
            axes[0].set_xlabel("field angle l [deg]")
            axes[0].set_ylabel("field angle m [deg]")

            im = axes[1].imshow(
                facet_dirty,
                origin="lower",
                extent=extent_facet_deg,
                cmap="inferno",
            )
            axes[1].set_title("Facet image after local phase recentering")
            axes[1].set_xlabel("local facet offset l [deg]")
            axes[1].set_ylabel("local facet offset m [deg]")
            plt.colorbar(im, ax=axes[1], shrink=0.82)

            plt.tight_layout()

            print(f"东侧离轴源在 naive 整场图中的局部峰值 ≈ {east_peak_naive:.3f}")
            print(f"东侧离轴源在精确宽场图中的局部峰值 ≈ {east_peak_wide:.3f}")
            print(f"东侧离轴源在 facet 图中的局部峰值 ≈ {east_peak_facet:.3f}")
            print(
                f"facet 相比 naive 整场图，局部峰值提高了约 "
                f"{east_peak_facet / east_peak_naive:.1f} 倍"
            )
            """
        ),
        md(
            r"""
            faceting 想传达的不是“最好永远手工切 facet”，而是一个更根本的宽场事实：**相位校正在宽场里本来就是方向依赖的。** 现代软件里的 `w-projection`、`w-stacking` 和更复杂的宽场算法，本质上都是在更系统地处理同一个问题。
            """
        ),
        md(
            r"""
            ### 9.12.3 多指向拼接为什么不能只做简单平均

            `w-term` 处理的是宽场相位；主波束处理的是宽场增益。下面再做第二个实验：

            - 构造两个相邻 pointing 的观测图；
            - 先直接平均，得到 naive mosaic；
            - 再按主波束权重做一个简化版的 `PB-aware mosaic`。

            这样可以看到：简单平均会把边缘源压得很低，而 PB-aware 拼接可以把通量拉回来，但代价是边缘有效噪声上升。
            """
        ),
        code(
            """
            extent_arcmin = [
                mosaic_coords_arcmin[0],
                mosaic_coords_arcmin[-1],
                mosaic_coords_arcmin[0],
                mosaic_coords_arcmin[-1],
            ]

            fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

            for ax, image, title in [
                (axes[0], mosaic_true_sky, "True sky"),
                (axes[1], naive_mosaic, "Naive mosaic average"),
                (axes[2], pb_weighted_mosaic, "PB-aware weighted mosaic"),
            ]:
                im = ax.imshow(
                    image,
                    origin="lower",
                    extent=extent_arcmin,
                    cmap="inferno",
                )
                ax.set_title(title)
                ax.set_xlabel("RA offset [arcmin]")
                ax.set_ylabel("Dec offset [arcmin]")
                plt.colorbar(im, ax=ax, shrink=0.82)

            plt.tight_layout()

            print(f"科学区域内 naive mosaic 的平均绝对误差 ≈ {mosaic_mean_abs_error_naive:.3f}")
            print(
                f"科学区域内 PB-aware mosaic 的平均绝对误差 ≈ {mosaic_mean_abs_error_weighted:.3f}"
            )
            print(
                f"边缘源峰值：真实天空 ≈ {outer_peak_true:.3f}，"
                f"naive mosaic ≈ {outer_peak_naive:.3f}，"
                f"PB-aware mosaic ≈ {outer_peak_weighted:.3f}"
            )
            print(
                f"边缘源位置的有效噪声约为中心重叠区的 {mosaic_noise_ratio:.1f} 倍"
            )
            """
        ),
        md(
            r"""
            这里要建立的专业判断是：**主波束校正不是免费的。** 它能改善边缘通量恢复，但也会同步放大边缘噪声和建模误差。这正是为什么实际宽场数据处理中，mosaic 参数、PB cutoff 和科学分析半径必须一起设定。
            """
        ),
        md(
            r"""
            ### 9.12.4 与真实软件流程的对应

            若把这两个教学实验映射回真实成像工作流，最常见的关键词大致包括：

            - `w-projection`、`w-stacking`、faceting：处理非共面基线带来的 `w-term`；
            - `A-projection`：在成像核里处理主波束等方向相关增益；
            - `AW-projection`：把 `w-term` 和主波束方向相关效应一起放进成像算子；
            - `mosaic`、`joint mosaic`：多指向宽场拼接与联合成像；
            - 参数选择：`imsize`、`cell`、`gridder`、`wprojplanes`、`pb cutoff`、facet 数量和科学目标必须联动。

            这个 notebook 想传达的最终结论是：**宽场方向相关成像不是单一按钮，而是相位项、主波束项和拼接策略共同决定的工作流。** 如果科学目标在大视场边缘、或者动态范围很高，那么这些问题都不能交给默认值处理。
            """
        ),
    ],
    "9_13_advanced_spectral_line_analysis.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.12 宽场方向相关成像](9_12_direction_dependent_widefield_imaging.ipynb)
                * 下一节： [9.x 延伸阅读与后续实践方向](9_x_further_reading_and_workflow.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.13 高级谱线分析：`3D mask`、source finding 与简化运动学拟合

            `9.7` 已经把谱线处理从“最小演示”推进到了更完整的基础工作流，但真实的谱线科学分析通常不会停在 moment map 和全局谱线宽度上。到了更专业的训练层次，我们往往还需要继续回答三类问题：

            - 哪些体素可以被视为可信发射，应该如何构造一个更稳健的 `3D mask`；
            - 一个 line cube 中如果同时存在主盘、伴随云团或离散分量，怎样做基本的 source finding 和组件目录整理；
            - 在隔离主分量之后，怎样从 `PV diagram` 出发，做一个最小但有物理意义的运动学拟合。

            这一节继续采用完全自包含的合成谱线立方体，不尝试伪装成真实 CASA 或 pipeline 环境，而是把高级谱线工作流最重要的判断链拆开来练：

            1. 构造一个含旋转盘和伴随云团的 H I 风格数据立方体；
            2. 用平滑辅助的双阈值法建立 `3D mask`；
            3. 做连通域标记，生成一个最小组件目录；
            4. 只保留主盘分量，抽取 `PV diagram` 和 ridge；
            5. 用一个简化的 arctan 旋转曲线模型做入门级拟合。

            这不会替代真实工作里的 `SoFiA`、tilted-ring 或 3D forward modeling，但它能把“检测、分离、测量、建模”这条谱线分析链明确地建立起来。
            """
        ),
        md("***"),
        code(
            """
            def gaussian_map(x, y, x0, y0, sigma_x, sigma_y):
                profile = np.exp(
                    -0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
                )
                profile /= profile.max()
                return profile


            def gaussian_kernel(npix, sigma_pix):
                coords = np.arange(npix) - npix // 2
                xx, yy = np.meshgrid(coords, coords)
                kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pix**2))
                kernel /= kernel.sum()
                return kernel


            def fft_convolve_same(image, kernel):
                image_ft = np.fft.fft2(image)
                kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel), s=image.shape)
                return np.real(np.fft.ifft2(image_ft * kernel_ft))


            def robust_rms(data):
                median = np.median(data)
                return 1.4826 * np.median(np.abs(data - median))


            def spectral_smooth(array):
                array = np.asarray(array)
                if array.shape[0] < 3:
                    return array.copy()

                out = np.empty_like(array)
                out[0] = 0.75 * array[0] + 0.25 * array[1]
                out[-1] = 0.75 * array[-1] + 0.25 * array[-2]
                out[1:-1] = (
                    0.25 * array[:-2] + 0.50 * array[1:-1] + 0.25 * array[2:]
                )
                return out


            def spatial_smooth(cube, kernel):
                return np.stack([fft_convolve_same(channel, kernel) for channel in cube], axis=0)


            def dilate6(mask):
                expanded = mask.copy()
                expanded[1:] |= mask[:-1]
                expanded[:-1] |= mask[1:]
                expanded[:, 1:] |= mask[:, :-1]
                expanded[:, :-1] |= mask[:, 1:]
                expanded[:, :, 1:] |= mask[:, :, :-1]
                expanded[:, :, :-1] |= mask[:, :, 1:]
                return expanded


            def connected_components_3d(mask, min_voxels=40, min_chan=3):
                visited = np.zeros(mask.shape, dtype=bool)
                labels = np.zeros(mask.shape, dtype=int)
                neighbors = [
                    (dv, dy, dx)
                    for dv in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if (dv, dy, dx) != (0, 0, 0)
                ]
                shape = mask.shape
                next_label = 0

                for seed in np.argwhere(mask):
                    z, y, x = map(int, seed)
                    if visited[z, y, x]:
                        continue

                    stack = [(z, y, x)]
                    visited[z, y, x] = True
                    voxels = []
                    channels = set()

                    while stack:
                        cz, cy, cx = stack.pop()
                        voxels.append((cz, cy, cx))
                        channels.add(cz)

                        for dv, dy, dx in neighbors:
                            nz = cz + dv
                            ny = cy + dy
                            nx = cx + dx
                            if not (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]):
                                continue
                            if not mask[nz, ny, nx] or visited[nz, ny, nx]:
                                continue
                            visited[nz, ny, nx] = True
                            stack.append((nz, ny, nx))

                    if len(voxels) < min_voxels or len(channels) < min_chan:
                        continue

                    next_label += 1
                    voxel_array = np.array(voxels)
                    labels[
                        voxel_array[:, 0],
                        voxel_array[:, 1],
                        voxel_array[:, 2],
                    ] = next_label

                return labels, next_label


            def component_catalog(cube, labels, vel_kms, x_arcsec, y_arcsec, dv_kms):
                rows = []
                for label in range(1, int(labels.max()) + 1):
                    voxels = np.argwhere(labels == label)
                    if voxels.size == 0:
                        continue

                    zz = voxels[:, 0]
                    yy = voxels[:, 1]
                    xx = voxels[:, 2]
                    values = cube[zz, yy, xx]
                    weights = np.clip(values, 1.0e-6, None)
                    peak_index = np.argmax(values)

                    rows.append(
                        {
                            "label": int(label),
                            "voxel_count": int(voxels.shape[0]),
                            "chan_min": float(vel_kms[zz].min()),
                            "chan_max": float(vel_kms[zz].max()),
                            "x_centroid": float(np.average(x_arcsec[xx], weights=weights)),
                            "y_centroid": float(np.average(y_arcsec[yy], weights=weights)),
                            "peak_velocity": float(vel_kms[zz[peak_index]]),
                            "flux_proxy": float(values.sum() * dv_kms),
                        }
                    )

                rows.sort(key=lambda row: row["voxel_count"], reverse=True)
                return rows


            def format_catalog(rows):
                header = (
                    "label  voxels  v-range[km/s]    centroid[arcsec]      "
                    "peak[km/s]  flux proxy"
                )
                lines = [header, "-" * len(header)]
                for row in rows:
                    lines.append(
                        f"{row['label']:>5d}  "
                        f"{row['voxel_count']:>6d}  "
                        f"{row['chan_min']:>6.1f}..{row['chan_max']:<6.1f}  "
                        f"({row['x_centroid']:>6.1f}, {row['y_centroid']:>6.1f})  "
                        f"{row['peak_velocity']:>8.1f}  "
                        f"{row['flux_proxy']:>10.2f}"
                    )
                return "\\n".join(lines)


            def projected_arctan_velocity(radius_arcsec, v_sys, v_flat, r_turn, inclination_deg):
                sin_i = np.sin(np.deg2rad(inclination_deg))
                return (
                    v_sys
                    + np.sign(radius_arcsec)
                    * v_flat
                    * (2.0 / np.pi)
                    * np.arctan(np.abs(radius_arcsec) / r_turn)
                    * sin_i
                )


            def fit_arctan_curve(radius_arcsec, velocity_kms, v_sys, inclination_deg=62.0):
                best_v_flat = None
                best_r_turn = None
                best_score = np.inf

                for v_flat in np.linspace(40.0, 70.0, 121):
                    for r_turn in np.linspace(4.0, 12.0, 65):
                        model = projected_arctan_velocity(
                            radius_arcsec,
                            v_sys=v_sys,
                            v_flat=v_flat,
                            r_turn=r_turn,
                            inclination_deg=inclination_deg,
                        )
                        score = np.mean((velocity_kms - model) ** 2)
                        if score < best_score:
                            best_score = score
                            best_v_flat = float(v_flat)
                            best_r_turn = float(r_turn)

                return best_v_flat, best_r_turn, float(best_score)


            nx = ny = 52
            nchan = 41
            cell_arcsec = 1.5
            dv_kms = 4.0
            x_arcsec = (np.arange(nx) - (nx - 1) / 2.0) * cell_arcsec
            y_arcsec = (np.arange(ny) - (ny - 1) / 2.0) * cell_arcsec
            vel_kms = np.linspace(-80.0, 80.0, nchan)
            xx, yy = np.meshgrid(x_arcsec, y_arcsec)

            inclination_deg = 62.0
            v_sys_true = 6.0
            v_flat_true = 56.0
            r_turn_true = 8.0
            velocity_dispersion_kms = 7.5

            sigma_major_arcsec = 11.0
            sigma_minor_arcsec = 6.0
            rr = np.hypot(xx, yy)
            theta = np.arctan2(yy, xx)
            disk_map = np.exp(
                -0.5 * (
                    (xx / sigma_major_arcsec) ** 2
                    + (yy / sigma_minor_arcsec) ** 2
                )
            )
            v_rot = v_flat_true * (2.0 / np.pi) * np.arctan(rr / r_turn_true)
            v_los = (
                v_sys_true
                + v_rot
                * np.sin(np.deg2rad(inclination_deg))
                * np.cos(theta)
            )

            companion_xy_arcsec = (-26.0, 14.0)
            companion_velocity_kms = 62.0
            companion_width_kms = 3.5
            companion_peak = 0.34
            companion_map = gaussian_map(
                xx,
                yy,
                x0=companion_xy_arcsec[0],
                y0=companion_xy_arcsec[1],
                sigma_x=2.0,
                sigma_y=2.0,
            )

            cube = np.zeros((nchan, ny, nx))
            for chan_index, velocity in enumerate(vel_kms):
                disk_profile = np.exp(
                    -0.5 * ((velocity - v_los) / velocity_dispersion_kms) ** 2
                )
                companion_profile = np.exp(
                    -0.5 * ((velocity - companion_velocity_kms) / companion_width_kms) ** 2
                )
                cube[chan_index] = 0.58 * disk_map * disk_profile + companion_peak * companion_map * companion_profile

            beam_kernel = gaussian_kernel(npix=9, sigma_pix=1.0)
            cube = np.stack(
                [fft_convolve_same(channel_map, beam_kernel) for channel_map in cube],
                axis=0,
            )
            cube += 0.024 * RNG.normal(size=cube.shape)

            line_free_mask = np.abs(vel_kms) > 68.0
            cube_rms = robust_rms(cube[line_free_mask])
            extent_arcsec = [
                x_arcsec[0],
                x_arcsec[-1],
                y_arcsec[0],
                y_arcsec[-1],
            ]

            print(f"cube 维度 = {nchan} x {ny} x {nx}")
            print(f"速度范围 = {vel_kms[0]:.1f} 到 {vel_kms[-1]:.1f} km/s，通道宽度 = {dv_kms:.1f} km/s")
            print(
                f"输入主盘参数：v_sys = {v_sys_true:.1f} km/s, "
                f"v_flat = {v_flat_true:.1f} km/s, r_turn = {r_turn_true:.1f} arcsec"
            )
            print(
                f"伴随云团输入位置 ≈ ({companion_xy_arcsec[0]:.1f}, {companion_xy_arcsec[1]:.1f}) arcsec, "
                f"峰值速度 ≈ {companion_velocity_kms:.1f} km/s"
            )
            print(f"line-free 通道估计得到的 cube RMS ≈ {cube_rms:.4f}")
            """
        ),
        md(
            r"""
            ### 9.13.1 先看懂这个谱线立方体：主盘之外还有一个伴随云团

            这个合成 cube 刻意不是“只有一个干净旋转盘”的理想玩具模型，而是多加了一个偏高速度、偏离主盘中心的窄线云团。目的很明确：后面的 `3D mask` 和 source finding 不是为了炫技，而是为了把主盘和次级分量真正分开。

            下面先看三件事：

            - 主盘主导的 moment 0 空间分布；
            - 云团附近速度通道的空间结构；
            - 全立方体的积分谱线。
            """
        ),
        code(
            """
            preview_moment0 = np.sum(np.clip(cube, 0.0, None), axis=0) * dv_kms
            companion_chan = np.argmin(np.abs(vel_kms - companion_velocity_kms))
            integrated_spectrum = np.sum(cube, axis=(1, 2))

            fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))

            im0 = axes[0].imshow(
                preview_moment0,
                origin="lower",
                extent=extent_arcsec,
                cmap="inferno",
            )
            axes[0].set_title("Positive-clipped moment 0 preview")
            axes[0].set_xlabel("RA offset [arcsec]")
            axes[0].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im0, ax=axes[0], shrink=0.82)

            im1 = axes[1].imshow(
                cube[companion_chan],
                origin="lower",
                extent=extent_arcsec,
                cmap="coolwarm",
                vmin=-4.0 * cube_rms,
                vmax=8.0 * cube_rms,
            )
            axes[1].set_title(f"Channel map at {vel_kms[companion_chan]:.1f} km/s")
            axes[1].set_xlabel("RA offset [arcsec]")
            axes[1].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im1, ax=axes[1], shrink=0.82)

            axes[2].plot(vel_kms, integrated_spectrum, color="tab:blue", lw=1.8)
            axes[2].axvline(v_sys_true, color="gray", ls="--", lw=1.1, label="true system velocity")
            axes[2].axvline(companion_velocity_kms, color="tab:red", ls=":", lw=1.1, label="companion cloud")
            axes[2].set_xlabel("Velocity [km/s]")
            axes[2].set_ylabel("Integrated line flux [arb.]")
            axes[2].set_title("Integrated spectrum")
            axes[2].legend(loc="upper left")

            plt.tight_layout()

            print(f"积分谱线峰值约出现在 {vel_kms[np.argmax(integrated_spectrum)]:.1f} km/s")
            print(f"伴随云团通道 {vel_kms[companion_chan]:.1f} km/s 上的最大像素值 ≈ {cube[companion_chan].max():.3f}")
            """
        ),
        md(
            r"""
            ### 9.13.2 用平滑辅助的双阈值法构造 `3D mask`

            谱线 source finding 里最常见的思路之一是：

            - 先在平滑后的 cube 上寻找高显著性 seeds；
            - 再把这些 seeds 向较低阈值的邻近体素生长；
            - 最后做连通域分离，并丢掉明显太小、太窄的伪分量。

            这背后的专业判断是：**检测和测量通常不应使用完全相同的 mask。** 平滑后的 cube 更利于检测弱发射，但真正的测量仍应回到原始分辨率 cube 上完成。
            """
        ),
        code(
            """
            smoothed_cube = spatial_smooth(
                spectral_smooth(cube),
                gaussian_kernel(npix=7, sigma_pix=1.25),
            )
            smoothed_rms = robust_rms(smoothed_cube[line_free_mask])

            seed_threshold = 5.0 * smoothed_rms
            grow_threshold = 2.6 * smoothed_rms
            seed_mask = smoothed_cube > seed_threshold
            grow_mask = smoothed_cube > grow_threshold

            mask_3d = seed_mask.copy()
            for _ in range(2):
                mask_3d = grow_mask & dilate6(mask_3d)

            labels_3d, n_components = connected_components_3d(
                mask_3d,
                min_voxels=40,
                min_chan=3,
            )
            catalog_rows = component_catalog(
                cube=cube,
                labels=labels_3d,
                vel_kms=vel_kms,
                x_arcsec=x_arcsec,
                y_arcsec=y_arcsec,
                dv_kms=dv_kms,
            )

            all_mask_moment0 = np.sum(np.where(mask_3d, cube, 0.0), axis=0) * dv_kms
            component_footprint = np.max(labels_3d, axis=0)

            main_label = catalog_rows[0]["label"]
            companion_label = catalog_rows[1]["label"]
            main_mask_3d = labels_3d == main_label
            companion_mask_3d = labels_3d == companion_label
            main_moment0 = np.sum(np.where(main_mask_3d, cube, 0.0), axis=0) * dv_kms
            companion_moment0 = np.sum(np.where(companion_mask_3d, cube, 0.0), axis=0) * dv_kms

            fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))

            im0 = axes[0].imshow(
                all_mask_moment0,
                origin="lower",
                extent=extent_arcsec,
                cmap="inferno",
            )
            axes[0].set_title("Moment 0 from 3D mask")
            axes[0].set_xlabel("RA offset [arcsec]")
            axes[0].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im0, ax=axes[0], shrink=0.82)

            im1 = axes[1].imshow(
                component_footprint,
                origin="lower",
                extent=extent_arcsec,
                cmap="viridis",
                vmin=0.0,
                vmax=max(2, n_components),
            )
            axes[1].set_title("Connected-component footprint")
            axes[1].set_xlabel("RA offset [arcsec]")
            axes[1].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im1, ax=axes[1], shrink=0.82)

            im2 = axes[2].imshow(
                main_moment0 + companion_moment0,
                origin="lower",
                extent=extent_arcsec,
                cmap="magma",
            )
            axes[2].contour(
                companion_moment0,
                levels=[0.20 * companion_moment0.max()],
                colors="cyan",
                linewidths=1.2,
                origin="lower",
                extent=extent_arcsec,
            )
            axes[2].set_title("Main disk plus detached companion")
            axes[2].set_xlabel("RA offset [arcsec]")
            axes[2].set_ylabel("Dec offset [arcsec]")
            plt.colorbar(im2, ax=axes[2], shrink=0.82)

            plt.tight_layout()

            print(f"smoothed cube RMS ≈ {smoothed_rms:.4f}")
            print(f"seed threshold ≈ {seed_threshold:.4f}, grow threshold ≈ {grow_threshold:.4f}")
            print(f"最终保留下来的 3D 分量数 = {n_components}")
            print(format_catalog(catalog_rows))
            """
        ),
        md(
            r"""
            上面的目录已经足以支撑一个很基础但专业上正确的判断：

            - 最大分量是主盘；
            - 第二个分量是偏高速度、偏离盘中心的伴随云团；
            - 只有在把它们先分开之后，后面的 `PV` 分析和旋转曲线拟合才不会被混淆。

            这一步的意义并不是“得到一个漂亮的表格”，而是让后续的科学测量真正拥有对象级别的边界条件。
            """
        ),
        md(
            r"""
            ### 9.13.3 只保留主盘，抽取 `PV diagram` 与 ridge

            这里采用一个非常常见的入门做法：先用主盘分量的 mask 限定测量区域，再沿主轴附近做一个小宽度切片，把二维 `PV diagram` 画出来。

            需要特别强调的是：`PV diagram` 既不是 moment 1 的替代品，也不是自动给出旋转曲线的按钮。它更像是一个诊断平面，帮助你判断：

            - 主盘是否具有近似规则的旋转结构；
            - 中心区域是否受 beam smearing 显著影响；
            - 是否仍有次级分量混进主盘切片。
            """
        ),
        code(
            """
            main_spectrum = np.sum(np.where(main_mask_3d, cube, 0.0), axis=(1, 2))
            main_positive_spectrum = np.clip(main_spectrum, 0.0, None)
            v_sys_est = np.sum(vel_kms * main_positive_spectrum) / np.sum(main_positive_spectrum)

            disk_moment0 = np.sum(np.where(main_mask_3d, cube, 0.0), axis=0) * dv_kms
            major_axis_y_index = np.argmax(disk_moment0.sum(axis=1))
            major_axis_y_arcsec = y_arcsec[major_axis_y_index]
            pv_half_width_pix = 2
            pv_slice = slice(
                max(0, major_axis_y_index - pv_half_width_pix),
                min(ny, major_axis_y_index + pv_half_width_pix + 1),
            )
            pv_diagram = np.sum(
                np.where(main_mask_3d[:, pv_slice, :], cube[:, pv_slice, :], 0.0),
                axis=1,
            )
            pv_smoothed = spectral_smooth(pv_diagram)

            column_peak = np.max(pv_smoothed, axis=0)
            ridge_velocity = np.full(nx, np.nan)
            ridge_valid = (column_peak > 0.12 * column_peak.max()) & (np.abs(x_arcsec) > 3.0)

            for x_index in range(nx):
                profile = np.clip(pv_smoothed[:, x_index], 0.0, None)
                if profile.max() <= 0.0:
                    continue
                high_mask = profile > 0.55 * profile.max()
                ridge_velocity[x_index] = np.sum(vel_kms[high_mask] * profile[high_mask]) / np.sum(profile[high_mask])

            ridge_valid &= np.isfinite(ridge_velocity)
            pv_extent = [x_arcsec[0], x_arcsec[-1], vel_kms[0], vel_kms[-1]]

            fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))

            im = axes[0].imshow(
                pv_smoothed,
                origin="lower",
                extent=pv_extent,
                aspect="auto",
                cmap="inferno",
            )
            axes[0].scatter(
                x_arcsec[ridge_valid],
                ridge_velocity[ridge_valid],
                s=18,
                color="cyan",
                edgecolor="black",
                linewidth=0.35,
                label="PV ridge",
            )
            axes[0].axhline(v_sys_est, color="white", ls="--", lw=1.0, label="estimated systemic velocity")
            axes[0].set_xlabel("Major-axis offset [arcsec]")
            axes[0].set_ylabel("Velocity [km/s]")
            axes[0].set_title("Main-component PV diagram")
            axes[0].legend(loc="upper left")
            plt.colorbar(im, ax=axes[0], shrink=0.84)

            axes[1].plot(vel_kms, main_spectrum, color="tab:blue", lw=1.8)
            axes[1].axvline(v_sys_true, color="gray", ls="--", lw=1.0, label="true v_sys")
            axes[1].axvline(v_sys_est, color="tab:red", ls=":", lw=1.2, label="estimated v_sys")
            axes[1].set_xlabel("Velocity [km/s]")
            axes[1].set_ylabel("Main-component integrated flux [arb.]")
            axes[1].set_title("Spectrum of the isolated main disk")
            axes[1].legend(loc="upper left")

            plt.tight_layout()

            print(f"主轴切片中心约位于 y = {major_axis_y_arcsec:.1f} arcsec")
            print(f"主盘积分谱线估计得到的系统速度 ≈ {v_sys_est:.2f} km/s")
            print(f"用于拟合的 ridge 点数 = {ridge_valid.sum()}")
            """
        ),
        md(
            r"""
            从这里开始，就要有一个更像专业分析而不是课堂演示的意识：**我们看到的 ridge 不是“真实旋转曲线”，而是“被有限角分辨率、谱线宽度和切片方式共同卷积过的观测摘要”。**

            因此下一步的拟合只能被理解为“简化入口”，而不是最终科学结论。真正的高精度工作通常需要 tilted-ring、厚盘模型、beam-smearing 修正，甚至直接在 3D cube 上做 forward modeling。
            """
        ),
        md(
            r"""
            ### 9.13.4 做一个最小但专业上自洽的旋转曲线拟合

            下面用一个简化的 arctan 模型去拟合主盘的 `PV ridge`。它的优点是参数少、可解释性强，足够作为培训页的第一层入口：

            - `v_sys` 用主盘积分谱线估计值固定；
            - `v_flat` 描述外盘接近平坦时的特征速度；
            - `r_turn` 描述从固体体式上升过渡到平坦段的大致尺度。

            这一步最值得学习的不是“拟合成功了”，而是比较输入真值和恢复结果之间的差异，并认识到 beam smearing 会把简单 ridge 拟合系统性地拉向较低的外盘速度。
            """
        ),
        code(
            """
            fit_v_flat, fit_r_turn, fit_score = fit_arctan_curve(
                radius_arcsec=x_arcsec[ridge_valid],
                velocity_kms=ridge_velocity[ridge_valid],
                v_sys=v_sys_est,
                inclination_deg=inclination_deg,
            )
            fitted_curve = projected_arctan_velocity(
                x_arcsec,
                v_sys=v_sys_est,
                v_flat=fit_v_flat,
                r_turn=fit_r_turn,
                inclination_deg=inclination_deg,
            )
            true_curve = projected_arctan_velocity(
                x_arcsec,
                v_sys=v_sys_true,
                v_flat=v_flat_true,
                r_turn=r_turn_true,
                inclination_deg=inclination_deg,
            )

            fig, ax = plt.subplots(figsize=(8.6, 4.3))
            ax.scatter(
                x_arcsec[ridge_valid],
                ridge_velocity[ridge_valid],
                s=22,
                color="tab:blue",
                edgecolor="black",
                linewidth=0.35,
                label="Recovered PV ridge",
            )
            ax.plot(x_arcsec, true_curve, color="gray", lw=2.0, ls="--", label="Input projected curve")
            ax.plot(x_arcsec, fitted_curve, color="tab:red", lw=2.0, label="Arctan fit to PV ridge")
            ax.axhline(v_sys_est, color="tab:red", lw=1.0, ls=":")
            ax.set_xlabel("Major-axis offset [arcsec]")
            ax.set_ylabel("Velocity [km/s]")
            ax.set_title("Simplified kinematic fit")
            ax.legend(loc="upper left")
            plt.tight_layout()

            print(
                f"拟合结果：v_sys ≈ {v_sys_est:.2f} km/s, "
                f"v_flat ≈ {fit_v_flat:.2f} km/s, r_turn ≈ {fit_r_turn:.2f} arcsec"
            )
            print(
                f"输入真值：v_sys = {v_sys_true:.2f} km/s, "
                f"v_flat = {v_flat_true:.2f} km/s, r_turn = {r_turn_true:.2f} arcsec"
            )
            print(
                f"参数偏差：Δv_flat ≈ {fit_v_flat - v_flat_true:+.2f} km/s, "
                f"Δr_turn ≈ {fit_r_turn - r_turn_true:+.2f} arcsec"
            )
            print(f"ridge 拟合的均方根残差 ≈ {np.sqrt(fit_score):.2f} km/s")
            """
        ),
        md(
            r"""
            这个结果非常适合作为训练页里的“第一层专业判断”：

            - 主盘系统速度恢复得相当稳定；
            - 简化 arctan 模型已经能抓住主盘整体旋转结构；
            - `v_flat` 往往会比输入真值偏低，因为我们现在拟合的是已经被 beam smearing 和有限谱线宽度“压平”的 `PV ridge`，而不是未卷积的三维动力学模型。

            也正因为如此，真正的高级谱线工作流通常不会止步于这里，而会继续进入 tilted-ring、3D forward modeling、误差传播和模型比较。
            """
        ),
        md(
            r"""
            ### 9.13.5 与真实工作流的对应

            如果把这一页映射回更真实的软件环境，典型工作流大致会长成下面这样：

            - 先做 line cube imaging，并建立 line-free 通道和噪声模型；
            - 用平滑辅助检测或专门的 source finder 构造 `3D mask`；
            - 输出组件目录，确认哪些发射属于主盘、伴星系、潮汐尾或外流结构；
            - 沿主轴或物理上更合理的几何参数做 `PV diagram`；
            - 再进入 tilted-ring、3DBarolo / TiRiFiC 一类的运动学建模或更完整的 3D 前向拟合。

            这一页最想建立的专业习惯是：**高级谱线分析不是把 moment 0 和 moment 1 再画得更漂亮，而是把“检测边界、组件分离、诊断图、运动学模型”连成一条可重复的分析链。**
            """
        ),
    ],
    "9_x_further_reading_and_workflow.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)
                * 上一节： [9.13 高级谱线分析](9_13_advanced_spectral_line_analysis.ipynb)

            ***
            """
        ),
        md("导入标准模块:"),
        code(IMPORT_BLOCK),
        md(
            r"""
            ## 9.x 延伸阅读与后续实践方向

            当前这一版第 9 章已经先把连续谱主线和第一批谱线实践建立起来：

            - 数据检查与初步 QA；
            - 基础校准流程；
            - 连续谱成像；
            - 自校准；
            - 图像质量评估；
            - 平均与展宽的工程约束。
            - 基础谱线处理：line-free 选择、`uv-domain` 概念实验、channel map、平滑辅助 masking、PV 图、`W20/W50`、双分量近似与 H I 物理量入口；
            - 高级谱线分析：`3D mask`、source finding、组件目录、主盘/云团分离、`PV ridge` 与简化旋转曲线拟合；
            - 宽带与宽场高级成像：MFS 思路、频谱指数恢复、主波束校正与噪声放大权衡。
            - 偏振成像入门：Stokes `I/Q/U/V`、偏振强度 debias、leakage / `D-term` 影响与最小 `RM` 拟合。
            - 短间距与联合成像入口：missing flux、negative bowl、单碟互补与 feather 权衡。
            - 偏振高级入口：交叉手相位校准、Faraday 多分量判断与 `RM synthesis` 的分辨率约束。
            - 宽场方向相关入口：`w-term`、faceting 与 PB-aware mosaic 的核心取舍。

            但如果要进一步对齐现代国际训练体系，后续最值得继续扩展的方向仍然包括：

            - 更真实的谱线 3D 前向建模，例如 tilted-ring、厚盘、外流分量比较与不确定度传播；
            - 更真实的偏振校准链，例如 cross-hand delay、`D-term` 求解、`RM-CLEAN` 与 `QU-fitting`；
            - 更完整的短间距联合处理，例如 flux scale matching、joint deconvolution 与多阵列联合；
            - 更完整的宽场方向相关算法，例如更真实的 A-projection / AW-projection 与 mosaic 联合去卷积；
            - archive / pipeline 数据再分析。
            """
        ),
        md(
            r"""
            ### 推荐阅读

            - CASA guides 与各阵列官方 reduction guide  
              适合理解真实软件任务链和 QA 规范。
            - Thompson, Moran & Swenson  
              适合把实践操作重新连回干涉测量基本量。
            - synthesis imaging school / interferometry school 讲义  
              特别适合继续补足成像参数选择、自校准经验和 QA 习惯。
            - 阵列官方 pipeline 或 archive 文档  
              适合理解“science-ready”产品的边界与再处理空间。
            """
        ),
        md(
            r"""
            ### 下一步

            若继续扩展第 9 章，建议优先顺序仍然是：

            1. 继续加厚高级谱线建模与物理量估计；
            2. 更完整的联合成像与多阵列整合；
            3. 继续加深偏振校准链与 Faraday 建模；
            4. 继续加深宽场方向相关与 mosaic 联合成像；
            5. archive / pipeline 数据再分析。
            """
        ),
        md("***"),
    ],
    "9_3_Observing_smearing.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)

            ***
            """
        ),
        md(
            r"""
            # 兼容导航页：观测中的展宽

            旧版目录中，展宽内容位于 `9_3_Observing_smearing.ipynb`。为了保持兼容，这个入口继续保留。

            当前新版实践主线已经把这部分整理到：

            - [9.6 时间平均、频率平均与展宽](9_6_averaging_and_smearing.ipynb)

            建议直接阅读新版页面，因为那里已经加入可运行的数值实验，并把展宽和实际 averaging 参数选择联系了起来。
            """
        ),
    ],
    "pimaging.ipynb": [
        md(
            """
            ***

            * [总目录](../0_Introduction/0_introduction.ipynb)
            * [术语表](../0_Introduction/1_glossary.ipynb)
            * [9. 实践部分](9_1_visualisation-inspection.ipynb)

            ***
            """
        ),
        md(
            r"""
            # 兼容导航页：成像实践

            旧版目录中，成像实践位于 `pimaging.ipynb`。为了不破坏旧链接，这个入口继续保留。

            当前新版连续谱成像 notebook 已经统一为：

            - [9.3 连续谱基础成像](9_3_continuum_imaging.ipynb)

            新版页面已经从截图式演示改成可直接运行的成像最小原型，并与 dirty image、PSF、restored image 的工作流保持一致。
            """
        ),
    ],
}


def write_notebook(path: Path, cells):
    cells_with_ids = []
    for index, original_cell in enumerate(cells):
        cell = dict(original_cell)
        cell["id"] = f"{path.stem}-{index:03d}"
        cells_with_ids.append(cell)

    notebook = {
        "cells": cells_with_ids,
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
