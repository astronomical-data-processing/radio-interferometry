# 射电干涉孔径合成基础

本仓库是一套面向中文学习者的射电干涉基础与数据处理教程，采用 Jupyter Notebook 组织内容，覆盖从射电科学背景、数学基础、可见度空间、成像、去卷积、观测系统、校准，到实践工作流的完整主线。

项目起源于原始英文教材 [Fundamentals of Radio Interferometry](https://github.com/griffinfoster/fundamentals_of_interferometry)。当前中文仓库已经不再是早期的双语镜像，而是在原有结构基础上进行了系统中文化、重写、扩写与实践重构，目标是形成一套适合中文教学与培训使用的专业教程。

## 当前状态

- 第 1 至第 8 章主体内容已经完成系统中文重写与统一风格整理。
- 第 9 章已经重构为一条连续的实践工作流，不再是零散截图式演示。
- 当前版本已经具备一个完整可用的中文教材主体，后续重点主要是精修、加厚和继续提升训练深度，而不是补空白骨架。

如果只看一句话：当前仓库已经可以作为一套完整的中文射电干涉基础教程使用。

## 内容结构

- [0_Introduction/0_introduction.ipynb](0_Introduction/0_introduction.ipynb)：总目录与阅读入口。
- [0_Introduction/1_glossary.ipynb](0_Introduction/1_glossary.ipynb)：术语表。
- `1_Radio_Science`：射电科学与基础天体物理背景。
- `2_Mathematical_Groundwork`：傅里叶、采样、卷积、最小二乘等数学基础。
- `3_Positional_Astronomy`：位置天文学与坐标系统。
- `4_Visibility_Space`：基线、可见度、UV 覆盖与 van Cittert-Zernike 定理。
- `5_Imaging`：成像、权重、网格化、宽场效应。
- `6_Deconvolution`：去卷积、CLEAN、残差与图像质量。
- `7_Observing_Systems`：RIME、主波束、极化、传播效应、RFI 等观测系统问题。
- `8_Calibration`：1GC、2GC、3GC 与校准主线。
- `9_Practical`：现代实践工作流。

## 第 9 章实践部分

当前实践链已经形成比较完整的训练结构，包括：

- 数据检查与初步质量控制
- 基础校准流程
- 连续谱基础成像
- 自校准
- 图像质量评估与测量
- averaging 与 smearing
- 基础谱线处理
- 宽带与宽场成像
- 偏振成像
- 短间距与 feather
- 交叉手相位校准与 `RM synthesis`
- 宽场方向相关成像
- 高级谱线分析：`3D mask`、source finding、组件目录、`PV ridge` 与简化运动学拟合
- 端到端连续谱教学案例：从数据检查、校准、成像、自校准到通量测量与误差预算
- 连续谱源表生成：PyBDSF 风格的图像到目录流程
- 成像参数选择：cell size、image size、weighting、taper、mask 与 QA 闭环
- QA 与失败模式识别：从原始数据、校准解到图像伪影的因果诊断
- 从图像到科学量：通量、亮温、上限、相关噪声与误差预算
- 谱线物理量与运动学解释：速度约定、moment、柱密度、PV 图、beam smearing 与线宽修正
- 偏振与 `Faraday` 诊断：Stokes、偏振统计、校准误差、`RMSF`、退偏振与复杂 Faraday 结构

如果你只想从实践部分开始，建议直接阅读 [9_Practical/9_1_visualisation-inspection.ipynb](9_Practical/9_1_visualisation-inspection.ipynb)。

## 如何开始阅读

推荐顺序：

1. 从 [0_Introduction/0_introduction.ipynb](0_Introduction/0_introduction.ipynb) 进入全书目录。
2. 按章节顺序阅读第 1 到第 8 章理论部分。
3. 再进入 [9_Practical/9_1_visualisation-inspection.ipynb](9_Practical/9_1_visualisation-inspection.ipynb)，按实践链顺序继续。

如果只是局部查阅，也建议先看术语表和相关章节目录，避免不同章节之间的符号、术语和约定脱节。

## 运行方式

本仓库以 Notebook 为主，推荐使用 Python 3 和 Jupyter 环境。

最基础的打开方式通常是：

```bash
python -m pip install jupyter numpy matplotlib
jupyter lab
```

然后从仓库根目录打开对应的 `.ipynb` 文件即可。

需要说明的是：

- 当前重写后的很多实践 notebook，特别是第 9 章的大部分页面，已经尽量做到自包含，并优先使用 `numpy` 与 `matplotlib` 进行可运行演示。
- 部分历史页面、遗留示例或外部数据案例，仍可能需要额外依赖或数据文件。
- 当前的 [requirements.txt](requirements.txt) 已经整理为“当前基础依赖列表”，适合作为仓库的默认安装入口；但它仍不是严格锁定版本的可复现实验环境文件。
- 目前已确认 `ephem`、`healpy` 和 `aplpy` 都不再是当前仓库的活动依赖。

## 数据文件说明

当前仓库中的大量重写内容已经尽量减少对外部大数据文件的依赖，但部分历史内容或原始数据示例仍可能需要 `data/` 目录下的额外文件。

历史数据链接仍保留如下：

- FITS 图像数据：<https://www.dropbox.com/s/n3jyiajytwuldpu/fundamentals_fits.tar.gz?dl=0>
- KAT-7 仿真 measurement set：<https://www.dropbox.com/s/kb3p2mthei8dgl9/simulated_KAT-7_ms.tar.gz?dl=0>

若确实需要这些历史数据，请将其解压到仓库内对应的 `data/` 子目录中。后续如继续维护，建议优先采用相对路径和可配置路径变量，不要在 notebook、脚本或文档中写死个人机器路径。

## 维护与扩展

如果后续继续扩展本项目，建议先阅读以下文档：

- [Roadmap.md](Roadmap.md)：全书后续补强方向。
- [0_Introduction/editing_guide.ipynb](0_Introduction/editing_guide.ipynb)：编辑参考。

与当前仓库状态直接相关的维护约定包括：

- 中文正文为主，英文只保留必要术语、变量名、软件任务名和缩写。
- 优先保证 notebook 自包含、可执行、适合作为教学材料阅读。
- 实践页优先解释“这一步解决什么问题”，而不是简单堆命令。
- 程序、脚本、notebook 和文档中不要写死个人机器上的绝对路径，例如 `/home/username/...`。

第 8、9 章的部分重写工作曾使用生成脚本辅助完成：

- [tools/rebuild_chapter8_notebooks.py](tools/rebuild_chapter8_notebooks.py)
- [tools/rebuild_chapter9_notebooks.py](tools/rebuild_chapter9_notebooks.py)

当前教材化版本以仓库中的 `.ipynb` 和静态图为准。若继续使用或修改生成脚本，需要先确认脚本不会重新引入已移除的旧式代码单元、HTML toggle 或与当前导航不一致的内容。

## 风格与编辑入口

- [0_Introduction/0_introduction.ipynb](0_Introduction/0_introduction.ipynb)：总目录与结构入口。
- [0_Introduction/editing_guide.ipynb](0_Introduction/editing_guide.ipynb)：编辑参考。

## 致谢

本项目基于原始英文教材项目继续发展。感谢原始英文版本的作者与贡献者为射电干涉教学社区打下的重要基础：

- 原始英文仓库：<https://github.com/griffinfoster/fundamentals_of_interferometry>
- 原始课程网站：<https://ratt-ru.github.io/fundamentals_of_interferometry/>

中文版本在此基础上持续重写、整理和扩展，力图形成一套更适合中文学习者的系统教程。

## 许可证

许可证信息请见 [LICENSE](LICENSE) 和 [LICENSE.md](LICENSE.md)。
