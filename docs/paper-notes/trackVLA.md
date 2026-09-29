# TrackVLA: Embodied Visual Tracking in the Wild

[论文链接](https://arxiv.org/abs/2505.23189)

[论文仓库](https://github.com/ShaoanWang/TrackVLA)

## 解决的问题以及解决方案

现有的做追踪的方法：把追踪分为识别目标和运动规划两个阶段，分别交给不同的 model 进行训练。TrackVLA 将两个方面合到一起，学习二者之间的“协同关系”。

传统方法的问题：传统的解耦合设计可能会造成识别和规划两个阶段的累积误差，举个例子，当识别失败的时候，后续的规划肯定也是没有用的，同时这个错误的规划也会进一步造成后续识别的困难。（之前看过的 [LOVON](./lovon.md) 就是走的这个模式，在其中，YOLO 部分负责目标检测，之后检测结果喂给 L2MM）

TrackVLA 的方法：对于这两项任务采用相同的 token 编码以及 LLM 预测下一个 token，但是解码方式不同。识别任务使用语言模型解码出文本，规划任务使用 anchor-based diffusion head 解码出路径点轨迹，两者联合训练。

## 实验结论

可以以零样本的方式在真实世界当中部署，同时有 855K 个具身视觉追踪样本和 855K 个真实世界识别样本。

## 局限性

- 视野范围受限，因为当前方法仅依赖自我中心的观测，那么实际上可见的范围比较狭窄，同时视角单一
- 控制器目前只有比较宽泛的路径点控制，缺少局部运动的控制器

## 具体架构

### Observation Encoding

这一部分涉及到视觉特征提取当中的设计。需要注意的是，TrackVLA 的设计特点就是把视觉提取模块同时应用到 track 以及 VQA（video question answer）任务上，在前者上训练以学习追踪能力，在后者上训练以学习识别能力。因此我们先看 track 任务如何输入输出。

对于时间戳 $T$，需要处理截止到时间 $T$ 的视频帧。对于其中的每一帧，先输入 EVA-CLIP 得到视觉特征，视觉特征的维度是 $V \in \mathbb{R}^{N \times C}$。其中 $N$ 代表 patch 大小（通常为 256，也就是把输入的单帧图像按照 $16 \times 16$ 分割），$C$ 是 token 的维度。

为了进一步减少数据量，进一步使用 GirlPool 池化策略，对于当前最新的帧，得到 $V_{fine} \in \mathbb{R}^{64 \times C}$，对于之前的历史帧，则池化为 $V_{coarse} \in \mathbb{R}^{4 \times C}$。

同时使用滑动窗口避免序列的无限长，窗口大小设置为 $k$。也就是追踪最近的 $k$ 个历史帧以及当前帧。之后把得到的 token 序列，通过一个双层 MLP，投影到 LLM 的潜空间。

之后再看 VQA 部分，前面的处理相同，之后 VQA 会把自己的输入样本全部编码为 $V_{coarse}$，最后经过 MLP。

### LLM Forwarding

LLM 部分负责预测接下来的 token，之后将结果输入解码部分。首先把之前视觉编码器当中得到的视觉 token 以及相对应的指令的 token，拼接在一起形成一个 token 序列。在指令特征当中，有一个特殊的 \[Track] token，用来表示任务类型。经过 LLM 处理之后，我们把 VQA 相关任务的结果送入语言头，解码得到文字回答；把 track 相关任务的结果送入动作头，处理之后得到 waypoints。

### Anchor-based Diffusion Action Model

不同于一般的扩散模型，这里使用的模型首先从选定的轨迹加噪，之后进行去噪学习。先从训练样本当中，使用 K-Means 选出来 $M$ 条有代表性的轨迹，每一条轨迹由若干个 waypoints 组成，每个 waypoint 是一个三元组，代表 robot 地点以及朝向。接下来对每个轨迹加入高斯噪声。对于扩散模型，接受噪声路径集以及 LLM 输出的隐藏状态 token作为输入，输出去噪路径集以及每条路径对应的评分。

对于样本集当中的每一个路径，对其做一个标记，把和真实路径 $\tau_{gt}$ 最接近的路径标记为 $s_{nearset} = 1$，其余的全部标记成 0。

因此最后的 loss 函数为：

$$
\mathcal{L}_{\mathrm{track}}
=
\sum_{i=1}^{M}
\left[
s_i\operatorname{MSE}(\hat{\tau}_i,\tau_{\mathrm{gt}})
+
\lambda\operatorname{BCE}(\hat{s}_i,s_i)
\right]
$$

其中 BCE 表示二分类交叉熵，$\hat{\tau}_{i}$ 表示输出的去噪路径，$\hat{s}_{i}$ 代表输出的分数。

## 数据收集

总共有两种数据，visual-track data 以及 video question answering data。visual-track data 根据不同难度划分为三档，分别是粗粒度简单目标追踪，细粒度复杂目标追踪以及歧义追踪（比如：追踪你看到的第一个人）。video question answer 则包含人物识别以及开放世界问答两种数据。

## 实验分析

文章当中涉及到的指标：

| 指标 | 含义 | 趋势 |
|---|---|---|
| **EL** (Episode Length) | 一次跟踪平均持续多少步；Gym-UnrealCV 上限为 500 步 | 越高越好 |
| **SR** (Success Rate) | 成功的 episode 占比 | 越高越好 |
| **TR** (Tracking Rate) | 整个过程中，成功跟住目标的时间步占比 | 越高越好 |
| **CR** (Collision Rate) | 因与目标碰撞而结束的 episode 占比 | 越低越好 |
| **ACC** | 识别任务的正确率 | 越高越好 |
| **FPS** | 每秒处理帧数 | 越高越快 |



