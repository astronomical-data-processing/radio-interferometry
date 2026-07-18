# Third-Party Assets and Provenance

本仓库维护的教材、代码和由项目代码生成的图示按 GPLv2 发布。来自外部机构、论文、网站或个人的照片、截图、图表和其他素材不会因为进入本仓库而自动改用 GPLv2；它们仍受各自权利人和来源许可约束。

“图片来源”只构成署名线索，不等同于再分发许可证明。发布版本应为每个外部资产记录以下信息：

- 仓库内路径；
- 作者或权利机构；
- 原始来源 URL 或出版物；
- 明确的许可证、公共领域声明或书面许可；
- 获取日期和所做修改。

若无法确认再分发权限，应删除该资产，或用本项目自行生成、明确为公共领域或具有兼容开放许可证的替代图重绘。不要仅凭“用于教学”推定可以公开再分发。

## 上游教材

本项目继承自 [Fundamentals of Radio Interferometry](https://github.com/griffinfoster/fundamentals_of_interferometry)。本轮核对的上游版本为 commit `e590ab8adb32e917634d900e56b65955324ff477`，其根目录使用 GNU General Public License version 2 文本；本仓库的 [LICENSE](LICENSE) 保留该许可证原文。未发现上游项目级 “version 2 or later” 授权。仍与该版本逐字节相同的图示按上游许可和贡献历史保留；翻译、大幅改写和扩充的衍生文件也继续遵守适用的上游条款。第三方素材不因本项目采用 GPLv2 而自动改变许可证。完整范围见 [LICENSING.md](LICENSING.md)。

## 已完成的 P0 替换

第 1.9 节原有 6 幅来源未知图片和 1 幅仅有机构署名、未记录再分发许可的设施照片，已替换为本项目生成的物理示意图。生成入口为 [interferometry_history_figures.py](1_Radio_Science/figures/interferometry_history_figures.py)。替代图说明双缝路径差、恒星迈克尔逊合束、Hooker 顶部基线、现代光学长基线阵与延迟线、海崖反射、连接阵相关器和核心/长基线布局，不冒充历史照片或特定阵列的精确复原。

第 1.4、1.8、1.10、1.11 节原有的天体照片、论文图、望远镜照片和阵列设施照片也已删除，改用 [radio_science_schematics.py](1_Radio_Science/figures/radio_science_schematics.py) 生成的原创教学图。该脚本当前共生成 28 幅图，覆盖多波段源形态、H I 潮汐结构、旋转曲线、大气窗口、自由-自由辐射、单孔径工程约束、馈源光路、阵列布局、低频孔径阵、毫米阵、VLBI、冗余阵和圆柱阵；所有设施相关图注均明确其为概念图，不按特定设施尺寸或站位复原。

第 1.5 节原有 Orion A 光学底图与射电等值线合成图、第 1.6 节原有 Cygnus A 射电图没有随文件提供独立来源和许可说明，现已分别改用同一脚本生成的 H II 区自由-自由辐射概念图和双射电瓣概念图。替代图只表达光学消光、发射量、核心、喷流、热点和射电瓣之间的物理关系，不冒充真实观测产品。

第 1.6 节原有 Cygnus A 射电瓣/热点谱 PNG 已改为 Notebook 内的可执行数据表和绘图。六个频点的 `L` 射电瓣、`L1` 热点通量逐项来自 Steenbrugge, Heywood & Blundell (2010), MNRAS 401, 67-76, Table 1, DOI `10.1111/j.1365-2966.2009.15663.x`；代码只做未加权描述性幂律拟合，并明确说明低频曲率、通量协方差和跨数据集系统误差边界。

第 5.1、5.2 节原来作为 Fourier 变换输入的野鸭照片和 Boccioni 绘画也已删除，改用 [fourier_input_figures.py](5_Imaging/figures/fourier_input_figures.py) 生成的合成射电天线场景和旋涡星系。替代输入保留了轮廓、直线边缘、平滑背景、旋臂和小尺度结点等教学所需频率结构，同时不再依赖外部作品。

这批 P0 替换消除了上述章节中仅有署名、没有明确再分发许可的图片。后续仍应在新增或发现其他第三方资产时逐项审核，不能把本次清理理解为对未来素材的自动授权。

## 第三方科学数据

`9_Practical/sample_packages/pybdsf_abell2255_replay/` 包含 PyBDSF 官方仓库测试夹具的一个原样子集：WSRT Abell 2255 裁剪 FITS、频率平均 Stokes-I 图、RMS 图、高斯模型、高斯残差和源目录。来源为 `lofar-astron/PyBDSF` commit `c70103be3ae9ae9908286f144e6ce956acc0ce5c`；历史首次在 commit `7e407c27019b71dfeca4d5690ee700ae637deea5` 将它作为测试数据加入。上游仓库整体按 GPLv3 发布，但未发现针对该 FITS 的单独许可声明或原始归档编号。本仓库保留了上游许可证全文，并在样本包 manifest 中逐文件记录原路径和 SHA-256；FITS 和目录内容未修改，仅在仓库中使用了更清楚的本地文件名。

该数据只能支持产品复盘、FITS 契约和源目录 QA。它不是完整观测归档或 Measurement Set；裁剪输入头缺少 `BUNIT` 和可用的物理频率轴，因此不能用于重新校准、重新成像或独立通量标度核查。

`9_Practical/sample_packages/bima_ngc4826_ms_replay/` 包含 casacore 官方仓库的 BIMA NGC 4826 测试 Measurement Set，当前来源 commit 为 `ef9a25f41cd2c7edfe7a2d0eee549becb55a6403`，历史首次在 commit `54c10722cb1a35342cbe6b132063fbf6ff9a002d` 将该文件加入 field-selection 测试。上游仓库使用 GNU Library General Public License v2，但未发现该 MS 的单独许可声明或原始归档编号。本仓库保留上游 `COPYING` 全文和原样 `.ms.tgz`，manifest 记录源路径、表摘要和 SHA-256。

为使 Python 3.10+ 基础环境无需 CASA/casacore 也能完成教学复盘，`extract_visibility.py` 从原 MS 导出 3C273 校准场和一个 NGC 4826 field/DDID 的 `DATA`、`FLAG`、`UVW`、`WEIGHT`、天线和时间列，并另存全部七个目标 field、四个 64 通道 data description 的行级 `UVW`、元数据和 flag 比例。这些派生数组同样被校验和固定。校准场结束到目标场开始相隔约 989 s，因此本案例不声称可把相对增益直接同步转移到目标。该包支持表结构、flag、$uv$ 覆盖和相对标量增益实验，但因缺少独立通量模型、完整日志和原归档身份，不能建立绝对通量标度或声称完成可发布的重成像。

## 新增资产规则

新增外部资产时，应同时更新本文件或同目录清单。优先顺序为：

1. 本项目代码生成且可复现的图；
2. 明确公共领域的机构素材；
3. CC BY 或其他与项目分发方式兼容的开放许可素材；
4. 已取得书面许可的素材。

不得新增来源未知、仅有搜索结果地址、只有机构名称而无许可说明，或禁止再分发的资产。
