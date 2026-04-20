# Chapter 9 Handover

## 目的

本文档用于记录第 9 章 `9_Practical` 当前已经完成的重构工作、剩余的扩展方向，以及未来继续修改时应保持的工作约束。

如果后续需要切换到新会话，建议优先阅读：

1. `Roadmap.md`
2. `Chapter8_Handover.md`
3. `Chapter9_Handover.md`


## 当前定位

第 9 章的目标不是保留零散的 CASA 截图式实践页，而是把全书前面各章的理论内容，落实成一条现代射电干涉数据处理工作流。

当前这一版的重点已经从“零散实践页”推进到“两条基础主线”：

- 数据检查与初步质量控制；
- bandpass / gain / applycal 的基础逻辑；
- dirty image、PSF 与 restored image；
- 自校准的实际停止准则；
- 图像质量评估与基本测量；
- averaging 与 smearing 的工程约束。
- 一条可运行且更接近真实工作的谱线处理链：line-free 通道选择、`uv-domain` 基线拟合概念实验、channel map、平滑辅助 masking、PV 图、`W20/W50`、双分量近似与 H I 物理量入口。


## 本轮已完成内容

本轮已经系统性重写或新增以下 notebook：

- `9_Practical/9_1_visualisation-inspection.ipynb`
- `9_Practical/9_2_calibration_workflow.ipynb`
- `9_Practical/9_3_continuum_imaging.ipynb`
- `9_Practical/9_4_self_calibration.ipynb`
- `9_Practical/9_5_image_assessment_and_measurement.ipynb`
- `9_Practical/9_6_averaging_and_smearing.ipynb`
- `9_Practical/9_7_spectral_line_processing.ipynb`
- `9_Practical/9_x_further_reading_and_workflow.ipynb`

同时保留并改写了兼容入口：

- `9_Practical/9_3_Observing_smearing.ipynb`
- `9_Practical/pimaging.ipynb`

配套新增生成脚本：

- `tools/rebuild_chapter9_notebooks.py`

当前上述 notebook 已顺序执行验证，状态为可运行；`9.7` 中先前 moment1 计算触发的运行期 warning 也已消除。


## 已完成的结构升级

与旧版相比，第 9 章目前已经具备以下改进：

- `9.1` 不再只是 CASA 启动说明，而是加入了合成测量集、阵列布局、UV 覆盖、动态谱、简单 QA 与 flag 建议；
- `9.2` 明确区分了 bandpass、时间增益与 applycal 的角色，并给出目标场校准前后的残差对比；
- `9.3` 把原来的截图式成像页改成了可运行的 dirty image / PSF / restored image 最小原型；
- `9.4` 把自校准从抽象概念落到了 phase-only 与 amplitude+phase 的工作流判断；
- `9.5` 增加了背景噪声、动态范围、beam-aware 通量测量等基础 QA 动作；
- `9.6` 把 smearing 与 averaging 参数选择联系起来，不再只是静态公式说明；
- `9.7` 已从“最小谱线原型”加厚为更完整的训练页，加入了 line-free 选择对比、`uv-domain` 基线拟合概念实验、平滑辅助 mask、PV 图、`W20/W50`、双分量近似，以及柱密度和 H I 质量的入门换算；
- `9.x` 已改为下一步扩展方向与阅读引导页；
- 旧文件 `pimaging.ipynb` 和 `9_3_Observing_smearing.ipynb` 现在保留为兼容导航页，不再承载主线内容。


## 当前仍然存在的缺口

第 9 章目前已经完成了连续谱基础主线，但离路线图中的完整实践体系仍有明显扩展空间。

### 1. 谱线处理已经起步，但还需要加厚

当前 `9.7` 已经不再只是最小工作流，而是覆盖了 line-free 选择、基线模型、mask、PV、多分量近似和基础物理量换算的综合练习。但若要达到成熟训练教程的深度，后续仍建议继续补强：

- 更真实的 uv-domain continuum subtraction 场景，例如多基线、多场或频变连续谱；
- 更系统的 3D mask 与 source finding 接口；
- 更完整的线型拟合与多分量情形，例如联合拟合、模型比较或不确定度传播；
- PV 图和旋转曲线、外流或非圆周运动之间的联系；
- 与物理量估计的进一步连接，例如总气体质量、旋转质量和距离误差的系统影响。

### 2. 偏振实践尚未进入第 9 章

虽然第 7、8 章已经打下理论基础，但当前还缺：

- Stokes I/Q/U/V 成像
- 偏振强度与极化角测量
- leakage / D-term 的流程化说明
- RM 或 Faraday rotation 的入门实践接口

### 3. 宽带与宽场高级成像还未展开

当前 `9.3` 和 `9.6` 已建立入口，但还缺：

- MFS / MT-MFS
- primary beam correction
- wide-field 参数选择实验
- A-projection / AW-projection 的工作流说明

### 4. 图像测量还可以更完整

`9.5` 已有基础 QA 与 beam-aware flux，但后续还可继续补：

- 源尺寸估计
- 误差条估计
- source finding
- residual 结构的更系统诊断


## 建议的下一步顺序

若继续扩展第 9 章，建议按下面顺序推进：

1. 继续加厚现有谱线处理 notebook
2. 扩展宽带/宽场高级成像 notebook
3. 新增偏振处理实践
4. 增加数据合并与短间距案例
5. 最后再补 archive / pipeline / software ecosystem 相关内容


## 与全书路线图的关系

当前判断与 `Roadmap.md` 一致：

- 第 9 章现在已经从“零散实践页”升级成“连续谱主线 + 已加厚的第一批谱线实践”；
- 但它仍只是 P0 路线中的基础层；
- 后续最重要的仍然是把第 9 章继续扩成更厚的谱线、偏振、宽场、短间距等专题实践平台。


## 工作约束

后续继续扩写第 9 章时，仍应保持以下原则：

- 中文正文为主，英文只保留必要术语、软件任务名、变量名和缩写；
- notebook 尽量自包含、可运行，不依赖个人机器路径；
- 优先解释“这一步解决什么问题”，再展示命令或参数；
- 参数类内容优先采用对比实验写法，而不是命令堆砌；
- 若保留旧文件名，应优先作为兼容入口，不要让旧结构继续成为主线障碍；
- 继续避免在程序、脚本、notebook 和文档中写死个人机器绝对路径。
