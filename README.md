# 射电干涉教材系列

本仓库包含两册面向中文教学与科研训练的射电干涉教材，均采用 Jupyter Notebook 组织。第一册《射电干涉孔径合成基础》覆盖射电科学、数学、可见度、成像、观测系统和校准；第二册《射电干涉数据处理实践》覆盖数据检查、校准、成像、科学测量、质量控制和可复现课程项目。

项目起源于原始英文教材 [Fundamentals of Radio Interferometry](https://github.com/griffinfoster/fundamentals_of_interferometry)。当前中文仓库已经不再是早期的双语镜像，而是在原有结构基础上进行了系统中文化、重写、扩写与实践重构，目标是形成一套适合中文教学与培训使用的专业教程。

## 当前状态

- 第一册由原第 1 至第 8 章组成，主体内容已经完成系统中文重写与统一风格整理。
- 第二册由原第 9 章发展而来，已经建立独立入口、四条阅读路径和可测试的页面分类；现有 `9.x` 编号暂时保留以兼容外部链接。
- 当前版本已经具备两册完整可用的中文教材主体；这表示课程主线、练习和实验入口已经成立，不表示逐式科学复核、跨章去重和发布级校对从此结束。

如果只看一句话：当前仓库已经形成“理论基础册 + 数据处理实践册”的两册体系。

## 内容结构

- [0_Introduction/0_introduction.ipynb](0_Introduction/0_introduction.ipynb)：总目录与阅读入口。
- [0_Introduction/1_glossary.ipynb](0_Introduction/1_glossary.ipynb)：术语表。

### 第一册：《射电干涉孔径合成基础》

- `1_Radio_Science`：射电科学、辐射传输与机制、谱线、单碟/阵列尺度及定量问题集。
- `2_Mathematical_Groundwork`：傅里叶、采样、卷积、最小二乘、复数统计、正则化、一维 CLEAN、综合练习与 FFT 作业。
- `3_Positional_Astronomy`：位置天文学、坐标系统、时间标准、参考系与精密天体测量边界；另有[定量问题集](3_Positional_Astronomy/3_problem_set.ipynb)。
- `4_Visibility_Space`：基线、可见度、UV 覆盖、van Cittert-Zernike 定理、闭合量、缺短间距与 mosaicking；另有[100 分综合问题集](4_Visibility_Space/4_problem_set.ipynb)。
- `5_Imaging`：成像、权重、网格化、宽场效应与成像参数选择；另有[综合问题集](5_Imaging/5_problem_set.ipynb)。
- `6_Deconvolution`：去卷积、CLEAN、残差、图像质量、源搜索、正则化与目录选择函数；另有[100 分综合问题集](6_Deconvolution/6_problem_set.ipynb)。
- `7_Observing_Systems`：RIME、主波束、极化、传播效应、RFI、系统温度、SEFD、观测日志与 QA；另有[定量问题集](7_Observing_Systems/7_problem_set.ipynb)。
- `8_Calibration`：方向无关外场校准、方向无关自校准、方向相关校准、退化、模型不完备与解可信度；[校准习题](8_Calibration/8_problem_set.ipynb)包含可执行增益求解和留出诊断实验。

### 第二册：[《射电干涉数据处理实践》](9_Practical/9_0_introduction.ipynb)

- `9_Practical`：现代实践工作流、真实轻量样本包、[WSRT/PyBDSF 产品复盘](9_Practical/9_37_pybdsf_real_product_replay.ipynb)、[BIMA Measurement Set 校准复盘](9_Practical/9_38_bima_measurement_set_calibration_replay.ipynb)、[VLA 3C391 公开归档绝对校准实验](9_Practical/9_39_vla_3c391_archive_calibration.ipynb)与项目练习材料；另有[100 分综合实践问题集](9_Practical/9_problem_set.ipynb)。
- [book_manifest.yaml](9_Practical/book_manifest.yaml)：记录第二册每一页所属阅读路径、页面类型和教学层级，并由测试检查目录覆盖。

## 第二册：射电干涉数据处理实践

第二册已经形成比较完整的训练结构，包括：

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
- 短间距、mosaicking 与联合成像：最大可恢复尺度、missing flux、feather、joint deconvolution 与通量恢复验证
- 宽带宽场算法边界：`MT-MFS`、主波束谱指数偏差、`w-term`、`A/AW-projection` 与方向相关校准
- 观测设计与归档数据再分析：科学目标、频谱设置、校准节奏、`QA`、`weblog` 与 provenance
- 软件生态与可复现实践：CASA、WSClean、DP3/DDFacet、CARTA、PyBDSF、SoFiA、Astropy 生态与 provenance
- VLBI 实践入口：延迟模型、`fringe fitting`、相位参考、SEFD 校准、天体测量与紧致源成像
- 低频与高频特殊观测体制：电离层、RFI、宽场 DDE、对流层相干时间、opacity、`Tsys` 与快速相位校准
- Pipeline QA、`weblog` 与再处理决策：自动产品证据判读、归档数据重成像/重校准分支与 provenance
- 最小可复现处理项目：项目目录模板、配置文件、依赖图、数据/软件/处理身份、验证门与连续谱再处理案例
- 源表与检测产品验证：PyBDSF/SoFiA 检测产品、选择函数、完整性、可靠性、CARTA region 与发布样本分支
- 最小可运行处理工作流：数据清单、校验和、环境记录、执行入口、失败重跑、运行报告与归档连续谱重跑包
- 公开归档数据复盘：科学目标到检索条件、metadata/weblog 证据板、最小重跑、产品差异审计与限制声明
- 文件级示例仓库与发布包：清单契约、产品溯源图、验证门、失败样本与教学复盘循环
- 轻量样本数据集与练习包：科学不变量、平均/裁剪损失预算、练习分层、标准路径、盲测任务与失败对照
- 处理报告与复查量规：证据主线、报告结构、误差预算、复查循环、常见失败模式与结论降级
- 综合课程项目设计：科学问题、里程碑、能力评价、分层任务、项目库、低面亮度连续谱案例与可选 Measurement Set 纵向复现训练
- 真实轻量样本包与项目练习材料：可分发 `npy` 图像、局部噪声图、源表、区域文件、manifest、QA 摘要、报告模板、复查量规与分层任务
- VLA 3C391 公开归档实验：固定来源和 SHA-256、Perley-Butler 2017 绝对标度、延迟/带通/时间增益、mosaic 成像、机器可读 QA，以及三种 mask 与尺度受限方案的实测失败边界

若只学习数据处理实践，应从[第二册独立目录](9_Practical/9_0_introduction.ipynb)选择基础处理、科学专题、可复现工作流或课程项目路径。

## 如何开始阅读

推荐顺序：

1. 从 [0_Introduction/0_introduction.ipynb](0_Introduction/0_introduction.ipynb) 进入全书目录。
2. 按章节顺序阅读第一册第 1 到第 8 章。
3. 再进入[第二册目录](9_Practical/9_0_introduction.ipynb)，按课程目标选择阅读路径。

如果只是局部查阅，也建议先看术语表和相关章节目录，避免不同章节之间的符号、术语和约定脱节。

## 适用层次

- 本科高年级读者可以把第 1 至第 7 章作为主线，重点掌握辐射量、傅里叶分析、坐标系统、可见度、成像、去卷积和 RIME。
- 硕士阶段读者应继续完成第一册第 8 章校准习题，并在第二册中选择连续谱、谱线、偏振或宽场专题形成项目报告。
- 博士阶段或研究训练不能只阅读流程说明，还应使用真实 Measurement Set、校准表、weblog 或公开归档产品复现至少一条完整处理链。

第一册理论主线已经达到本科高年级教材所需的覆盖和数学深度；第 3、7 章已有综合定量问题集，第 8 章已有可执行校准实验。第二册 9.38 节把同一求解器接入真实 BIMA Measurement Set 列，9.39 节进一步用有独立归档身份和通量模型的 VLA 3C391 数据验证了绝对标度、完整校准转移、mosaic 重成像和最终 QA，并用四组受控成像实验确认现有方案仍只适合作为课程回归基线。9.39 是可选课程纵向训练，不是默认依赖 CASA 和大型数据的基础路径。详细判断见 [Roadmap.md](Roadmap.md)。

## 运行方式

本仓库以 Notebook 为主，要求 Python 3.10 或更高版本，并推荐使用隔离的虚拟环境。

最基础的打开方式是：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m jupyter lab
```

然后从仓库根目录打开对应的 `.ipynb` 文件即可。

需要说明的是：

- 第一册第 1 至第 6 章保留了较多可运行的数值演示；第 7 章提供系统与 RIME 定量问题集，第 8 章提供可执行复增益求解实验；第二册的 `practical_metrics.py`、现有 9.36 节及其轻量样本包提供可测试公式、文件核验、区域测量与误差预算路径。
- 部分历史页面、遗留示例或外部数据案例，仍可能需要额外依赖或数据文件。
- 当前的 [requirements.txt](requirements.txt) 已经整理为“当前基础依赖列表”，适合作为仓库的默认安装入口；但它仍不是严格锁定版本的可复现实验环境文件。
- 目前已确认 `ephem`、`healpy` 和 `aplpy` 都不再是当前仓库的活动依赖。
- `itrf2enu.py` 已经是纯 NumPy 实现；历史 Measurement Set 绘图脚本和 9.38 样本的重新提取需要可选的 `python-casacore`。9.38 的默认复盘使用已校验的 NumPy 提取包，不增加基础依赖。9.39 默认只读取 manifest 和参考指标，完整流水线另用 CASA 6.7 环境。WSClean、CASA、Pyxis/Tigger 等外部射电软件不属于基础 Python 依赖。

## 数据文件说明

默认教材路径不依赖外部大文件：当前 71 本含实际代码的 Notebook 均纳入 Python 3.10 和 3.13 基础环境回归。9.39 的 3C391 原始 MS 只在课程扩展运行时按固定 URL、长度和 SHA-256 下载，不进入 Git。旧 Högbom 和 Clark CLEAN 页面会优先读取用户提供的历史 FITS 图像；文件不存在时，自动使用固定随机种子的合成脏图和 PSF。

原项目的两个 Dropbox 归档地址已经失效，因此不再由 `Makefile` 自动下载。`data/scripts/` 下的 WSClean、Tigger 和 Measurement Set 脚本仍作为历史工具入口保留，运行它们需要用户自行准备对应数据并安装外部射电软件；这些工具不属于默认 Notebook 回归范围。

新增外部数据案例时，应记录公开来源、校验和、许可证和目标目录，并使用相对路径或可配置路径变量，不要写死个人机器路径。

## 验证

快速运行算法、链接、Notebook 结构和资产完整性测试：

```bash
make test
```

执行全部含代码的 Notebook：

```bash
make test-notebooks
```

Notebook 回归会把仓库复制到临时目录，并为每本 Notebook 启动独立 Python 内核，因此生成图和临时输出不会修改当前工作区。教材代码的运行期告警会按错误处理；GitHub Actions 会在 Python 3.10 和 3.13 下同时运行这两组检查。

## 维护与扩展

如果后续继续扩展本项目，建议先阅读以下文档：

- [Roadmap.md](Roadmap.md)：全书后续补强方向。
- [0_Introduction/editing_guide.ipynb](0_Introduction/editing_guide.ipynb)：编辑参考。

与当前仓库状态直接相关的维护约定包括：

- 中文正文为主，英文只保留必要术语、变量名、软件任务名和缩写。
- 优先保证 notebook 自包含、可执行、适合作为教学材料阅读。
- 实践页优先解释“这一步解决什么问题”，而不是简单堆命令。
- 程序、脚本、notebook 和文档中不要写死个人机器上的绝对路径，例如 `/home/username/...`。

## 风格与编辑入口

- [0_Introduction/0_introduction.ipynb](0_Introduction/0_introduction.ipynb)：总目录与结构入口。
- [0_Introduction/editing_guide.ipynb](0_Introduction/editing_guide.ipynb)：编辑参考。

## 致谢

本项目基于原始英文教材项目继续发展。感谢原始英文版本的作者与贡献者为射电干涉教学社区打下的重要基础：

- 原始英文仓库：<https://github.com/griffinfoster/fundamentals_of_interferometry>

中文版本在此基础上持续重写、整理和扩展，力图形成一套更适合中文教学与科研训练的系统教程。

## 许可证

本仓库维护的教材、代码和项目生成图示统一采用 [GNU General Public License version 2](LICENSE)（GPL-2.0-only）。项目继承自原始英文教材；翻译、大幅改写和扩充不会自动消除衍生关系，原作者的版权与适用的 GPLv2 条款继续保留。外部照片、论文图表、科学数据和其他第三方素材仍遵守各自条件，不因收录而自动改用 GPLv2。完整边界见 [LICENSING.md](LICENSING.md)，素材溯源见 [ASSET_PROVENANCE.md](ASSET_PROVENANCE.md)。
