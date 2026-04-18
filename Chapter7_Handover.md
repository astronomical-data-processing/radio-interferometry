# Chapter 7 Handover

## 目的

本文档用于记录第 7 章 `7_Observing_Systems` 的当前状态、差距判断、后续优先级、以及未来继续修改时必须遵守的工作约束。

如果后续需要切换到新会话，建议先阅读：

1. `Roadmap.md`
2. `Chapter7_Handover.md`

前者用于理解全书层面的总体方向，后者用于理解第 7 章当前已经做到哪里、下一步最值得做什么。


## 项目背景

当前项目目标不是把仓库整理成一套中文科普材料，而是把它打磨成一套面向中国学生的、专业、系统、可训练的射电干涉教程。

对第 7 章的核心要求是：

- 保持专业性，不降格为泛泛介绍；
- 在可能情况下增加广度与深度，而不是只做形式上的中文化；
- 尽量把概念、公式、工程解释和可运行示例放在一起；
- 让 notebook 本身成为训练材料，而不只是静态讲义；
- 尽量减少对外部隐式依赖的依赖，优先使用自包含示例；
- 全部内容避免写死个人机器的绝对路径。


## 已完成状态

当前第 7 章已经完成一轮系统性重写，主线从概念介绍推进到了“可执行的观测系统教程”。

已经重写并验证的 notebook：

- `7_Observing_Systems/7_0_introduction.ipynb`
- `7_Observing_Systems/7_1_jones_notation.ipynb`
- `7_Observing_Systems/7_2_rime.ipynb`
- `7_Observing_Systems/7_3_analogue.ipynb`
- `7_Observing_Systems/7_4_digital.ipynb`
- `7_Observing_Systems/7_5_primary_beam.ipynb`
- `7_Observing_Systems/7_6_polarization.ipynb`
- `7_Observing_Systems/7_7_propagation_effects.ipynb`
- `7_Observing_Systems/7_8_rfi.ipynb`
- `7_Observing_Systems/7_x_further_reading_and_references.ipynb`

其中，`7_0` 到 `7_8` 的主体 notebook 已按当前工作流做过运行验证，目标是保证：

- notebook 内的代码单元可以顺序执行；
- 不依赖旧版隐藏文件或个人环境；
- 示例尽量用 `numpy`、`matplotlib` 和 notebook 内生成的数据完成；
- 解释与数值输出相互一致。


## 第 7 章当前结构

当前目录主线如下：

- `7.1` Jones 符号
- `7.2` RIME
- `7.3` 模拟电子学
- `7.4` 数字相关器
- `7.5` 主波束
- `7.6` 极化与馈源
- `7.7` 传播效应
- `7.8` RFI
- `7.x` 延伸阅读

当前判断是：

- 章节结构本身已经合理；
- 当前主要问题已不再是“目录缺不缺大标题”；
- 当前真正的差距在于：每个标题下面的工程深度、误差传播、校准接口、真实系统案例还不够。


## 当前优势

和旧版相比，第 7 章已经具备以下明显进步：

- 用 Jones / RIME 作为全章统一语言，而不是各节彼此脱节；
- 从模拟链路到数字相关器形成了连续的系统响应主线；
- 主波束、极化、传播、RFI 已经不再只是概念说明，而开始转向可计算示例；
- 多个 notebook 已改为自包含示例，可直接作为训练材料；
- 延伸阅读页已经按主题组织，不再只是空白占位页。


## 与标准教材相比的主要缺口

当前第 7 章与标准教材、暑校课程相比，仍有以下明显不足。

### 1. 天线与口径工程细节不足

当前 `7.5` 已经覆盖主波束、频率伸缩和方向依赖增益，但还缺：

- 口径效率分解
- spillover
- blockage
- taper efficiency
- Ruze 公式
- 表面误差与高频效率
- 指向误差导致的增益损失
- beam squint / beam squash
- far-field 条件

这部分是把“主波束”从图像直觉提升到“性能预算”的关键。

### 2. 接收机与相关器工程深度仍不够

当前 `7.3` 与 `7.4` 已经覆盖 LNA、SEFD、ADC、FFT、PFB 动机、FX/XF、corner turn、MS 骨架，但还缺：

- noise figure 与 noise temperature 的换算
- SSB / DSB / 2SB 接收机
- LO 与时钟稳定度
- 采样抖动
- requantization efficiency
- time smearing
- bandwidth smearing
- PFB taps 与通道泄漏权衡
- 相位切换与相关器工程细节

### 3. VLBI 几乎尚未进入第 7 章主线

这是当前最明显的系统性缺口之一。还缺：

- hydrogen maser
- 时间频率基准
- 几何时延模型
- fringe fitting
- phase referencing
- 记录与相关流程
- e-VLBI
- astrometry / geodesy 接口

### 4. 极化部分还缺真实系统层面的复杂性

当前 `7.6` 已建立 `C-Jones`、`D-Jones`、基底变换、视差角旋转和泄漏示例，但还缺：

- off-axis instrumental polarization
- beam squint / squash 与极化的联系
- cross-hand delay / phase
- 宽带极化
- RM synthesis 的前置接口
- 极化保真度与误差传播

### 5. 传播效应到校准策略的桥接仍偏弱

当前 `7.7` 已经把 ISM / 电离层 / 对流层的物理量级讲清楚，但还缺：

- coherence time
- isoplanatic patch
- fast switching
- WVR
- GPS-TEC
- 低频电离层屏建模
- 传播误差如何转化为实际校准策略

### 6. RFI 章节仍偏基础

当前 `7.8` 已经具备动态频谱、自动标记和 fringe stopping 抑制示例，但还缺：

- AOFlagger / SumThreshold
- spectral kurtosis
- pre-correlation blanking
- reference cancellation
- spatial filtering / nulling
- flag 对权重、uv 覆盖和后续成像的传播影响

### 7. 全章仍缺一份系统误差总图

建议最终补一页“系统误差总表”，把下面这些量连在一起：

- Jones 项
- 物理来源
- 观测症状
- 常见频段
- 常用校准手段
- 若不校正会影响什么科学量


## 后续优先级

### P0：下一步最值得做

1. 强化 `7.5` 的天线工程部分

- 加入口径效率、Ruze、表面误差、指向误差、beam squint / squash
- 目标：把主波束从“图样”提升为“性能预算与系统误差”

2. 强化 `7.3` 与 `7.4` 的接收机和相关器工程部分

- 加入 noise figure、SSB/DSB/2SB、时钟稳定度、smearing、requantization
- 目标：让这一部分更接近标准教材和暑校的工程深度

3. 增加一个 VLBI 专题 notebook

- 建议放在 `7_Observing_Systems` 下，作为第 7 章扩展节
- 目标：补齐当前最明显的结构性缺口

4. 强化 `7.6` 的真实极化误差部分

- 加入 off-axis 极化误差、cross-hand delay/phase、宽带极化接口

### P1：紧随其后

1. 强化 `7.7` 中从传播效应到校准策略的桥梁
2. 强化 `7.8` 中现代 RFI 方法谱系
3. 在 `7.x` 或单独 notebook 中加入系统误差总表

### P2：有价值但不必抢先

1. 增加不同阵列的案例对比

- JVLA
- MeerKAT
- ALMA
- LOFAR
- VLBI

2. 为每个 notebook 增加训练型小任务

- 不是传统习题，而是“改参数看现象”的 notebook 任务


## 推荐实施顺序

建议按下面顺序继续：

1. `7.5` 天线与口径工程补强
2. `7.3` / `7.4` 接收机与数字相关器补强
3. 新增 VLBI 专题
4. `7.6` 极化系统误差补强
5. `7.7` 传播效应到校准策略补强
6. `7.8` 高级 RFI 方法补强
7. `7.x` 系统误差总图与阅读引导补强


## 未来继续修改时的要求

以下要求属于后续工作时应显式遵守的约束。

### 内容定位

- 面向中国学生，目标是优秀的专业训练教程
- 不写成泛泛科普
- 应尽可能同时具备广度、深度和工程感
- 要与国际标准教材和暑校内容相互对照

### 写作风格

- 中文正文为主
- 英文仅保留必要术语、缩写、软件任务名、公式变量
- 优先解释物理意义，再解释实现方式
- 尽量把“概念、公式、代码、工程解释、误差后果”串在一起

### notebook 设计原则

- 优先自包含、可运行示例
- 尽量避免依赖外部临时数据、隐藏文件、个人本地路径
- 若旧 notebook 结构混乱，可直接重写，不必小修小补
- 优先使用 `numpy`、`matplotlib` 这类基础依赖
- 若使用图片，应标明来源

### 路径与可移植性

- 禁止在文档、notebook、脚本中写死个人机器绝对路径
- 禁止出现类似 `/home/...` 这类个人路径
- 一律优先使用相对路径或可配置变量

### 质量要求

- 每次重写后尽量做 notebook 运行验证
- 若某节改写后仍依赖外部资源，应在正文中清楚说明
- 数值输出要与正文结论一致
- 图和实验不要只是“好看”，必须服务于理解


## 会话切换时建议的接手方式

如果未来换新会话继续工作，建议采用下面的最小接手流程：

1. 先读 `Roadmap.md`
2. 再读 `Chapter7_Handover.md`
3. 检查 `7_Observing_Systems` 当前哪些 notebook 已经改写
4. 从本文件的 “P0” 列表中选择下一项继续
5. 保持现有风格：专业中文、自包含 notebook、可执行示例、禁止绝对路径


## 一句话总结

第 7 章当前已经从“中文化过的旧讲义”升级为“结构合理的观测系统教程”；下一步最重要的工作，不是再加更多大标题，而是把每个现有标题继续推进到标准教材级别的工程深度和校准接口层面。
