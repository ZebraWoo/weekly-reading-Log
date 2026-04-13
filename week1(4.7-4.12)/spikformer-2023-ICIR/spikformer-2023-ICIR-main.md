# 0 abstract
## 0.1 SNN vs. the self-attention mechanism
1. snn: energy-efficient and event-driven paradigm
2. self-attention: capture feature dependencies
## 0.2  In this paper
1. leverage both: self-attention capability and biological properties
2. propose a novel: Spiking Self Attention(SSA)
3. a framework: Spiking Transformer
	 i. sparse visual feature
	 ii. using spike-form Query, Key and Value without softmax
	 iii. efficient and low energy consumption
	 iv. performance: outperform, both neuromorphic and static datasets


neuromorphic datasets?			
[(23 封私信 / 46 条消息) SNN系列文章21——常用神经形态数据集及其处理 - 知乎](https://zhuanlan.zhihu.com/p/623822251)

# 1 Introduction
## 1.1 vanilla self-attention
 1. 核心组件：Q、K、V 是什么？
	在 VSA 中，每一个输入（单词或特征）都会被转换成三个不同的向量：
	- **Query (查询项 - Q):** 就像你手里拿着的“借书条”，代表“我正在寻找什么样的信息”。
    
	- **Key (键值项 - K):** 就像书架上每本书的“标签/索引”，代表“我包含哪些信息”。
    
	- **Value (内容项 - V):** 就像书里的“实际内容”，代表“如果匹配成功，我要提供什么信息”。
    
 2. VSA 的运行步骤
	VSA 的目标是计算输入的序列中，哪些部分是相关的，并把相关的部分“融合”在一起。
	- **第一步：计算相似度（打分）**
		- 系统将你的查询项 $Q$ 与所有的键值项 $K$ 进行点积（Dot Product）计算。
		- 如果 $Q$ 和某个 $K$ 很接近，分数就高，表示这两者关系紧密。

$$Score = Q \cdot K^\top$$
		**第二步：缩放与归一化（Softmax）**
			为了防止分数过大导致计算不稳定，先除以一个缩放因子 $\sqrt{d_k}$。接着，使用 **Softmax** 函数将这些分数转化为 **0 到 1 之间的概率（权重）**，所有权重相加等于 1。

$$AttentionMap = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

> **注：** 这就是你之前提到的 SSA 试图丢弃的部分。在 VSA 中，Softmax 极其重要，因为它决定了模型该“专注”看哪一部分。

**第三步：加权求和**

最后，用计算出的权重去乘以对应的 **Value (V)**。

- 权重大的内容会被保留更多，权重小的则被忽略。
    

---

### 3. VSA 的数学表达

综合起来，VSA 的公式非常简洁：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

---

### 4. VSA 的优缺点

#### 优点：

1. **全局视野：** 无论两个词离得有多远，VSA 都能通过一步计算建立它们之间的联系（解决长距离依赖问题）。
    
2. **并行计算：** 与传统的 RNN（必须一个词一个词读）不同，VSA 可以同时处理整个句子，效率极高。
    

### 缺点：

1. **计算量大：** 计算量随序列长度的**平方**级增长（$N^2$）。如果句子长度翻倍，计算量会翻四倍。
    
2. **显存占用高：** 在处理超长文本（如整本书）时，Softmax 生成的注意力矩阵会非常吃内存。
## 1.2 spiking self attention(SSA)
1. Q,K,V are in spike form, only contains of 0/1
2. obstacles to the SSA: softmax
	i. natural non-negativeness
	ii. less fine-grained feature
3. be done by logical and operation and addition
	1. the number of operations in SSA is small
	2. decomposable:$QK^TV$=$Q(K^TV)$ ,$O(n^2)$->$O(n)$
## 1.3 Spiking Transformer
  1. directly trained Transformer in the SNNs
  2. 3-fold contributions:
	  i. using sparse-form Query, Key and Value
	  ii. implement self-attention and Transformer in SNNs'
## 1.4 Spiking Neural Networks
1. Leaky Integrate-and-Fire neuron
2. PLIF 
3. two ways to get deep SNN models: ANN-to-SNN conversion and direct training
	1. ANN-to-SNN conversion: replacing the ReLU activation layers with spiking neurons. 
	2. Direct Training: surrogate gradient is used for backpropagation


## 1.5 LIF原理
	按“充电 -> 放电 -> 复位”理解：

1. 充电（膜电位积分）

$H[t]=V[t−1]+$$\frac{1}{τ}$$​(X[t]−(V[t−1]−V_{reset}​))$

- `X[t]`：当前时刻输入电流（突触输入汇总）
- `V[t-1]`：上一时刻膜电位
- τ：时间常数，控制“变化快慢”
- 直观上：膜电位会被输入推高，但也会向 `Vreset` 漏回去（leaky）

2. 是否发放脉冲（阈值比较）

$S[t]=Θ(H[t]−V_{th})S[t]=Θ(H[t]−V_{th}​)$

- `H[t] >= Vth` 就发放，`S[t]=1`
- 否则不发放，`S[t]=0`
- `\Theta` 就是硬阈值函数（Heaviside）

3. 发放后更新膜电位（复位机制）

$V[t]=H[t](1−S[t])+V_{reset}S[t]$

- 若 $S[t]=0$（没放电）：$V[t]=H[t]$
- 若 $S[t]=1$（放电）：$V[t]=Vreset$
- 一行公式把“没放电保留、放电就复位”两种情况合并了

---

### 为什么这对 SNN 重要？

- 输出 `S[t]` 是 0/1 脉冲，天然稀疏、事件驱动。
- 和 ANN 连续实值激活不同，SNN靠时间维度累积信息（多个 `t`）。
- Spikformer 里后续注意力等模块都基于这种 spike 表示来设计。