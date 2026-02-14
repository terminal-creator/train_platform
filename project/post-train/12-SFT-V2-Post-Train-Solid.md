# Post-Train Solid: SFT-V2 —— 通往稳健后训练的终极指南

> **摘要**：SFT（监督微调）看似简单——“准备数据，运行 `Trainer`，结束”。但在追求“世界级”模型效果时，SFT 是一个极其精密的系统工程。本文（SFT-V2）旨在超越基础教程，深入探讨如何打造 **Solid（稳健）、Sharp（敏锐）且 Robust（鲁棒）** 的模型。我们将从数据美学、训练动力学、以及那些鲜为人知的工程细节三个维度展开。

---

## 1. 核心哲学：SFT 到底在做什么？

在进入工程细节前，必须统一认知。根据 **LIMA (Less Is More for Alignment)** 假设和业界共识：
*   **SFT 不是在教模型新知识**（Pre-training 已经做完了）。
*   **SFT 是在做“格式对齐”和“知识激活”**。它教会模型用特定的语气、结构和逻辑来提取预训练阶段的知识。

**Solid SFT 的目标**：在不破坏预训练分布（避免“脑损伤”）的前提下，最大化地激发模型的指令遵循能力。

---

## 2. Data-Centric AI：数据的“炼金术”

“Garbage In, Garbage Out” 是老生常谈，但在 V2 阶段，我们需要更高级的数据策略。

### 2.1 合成与进化 (Synthesis & Evolution)
依赖人工标注极其昂贵且难以扩展。**Solid** 的 SFT 高度依赖高质量合成数据。
*   **Evol-Instruct (WizardLM)**：不要只用原始 Prompt。通过 LLM 改写 Prompt，使其变得更复杂、更具体、增加约束条件。
    *   *深度化*：“请写代码...” -> “请写一个 Python 函数，处理...，时间复杂度为 O(n)...”
*   **Magpie / Self-Instruct**：利用强模型（如 Claude-3.5-Sonnet, GPT-4o）生成多样化的对话数据。
*   **Rejection Sampling (拒绝采样)**：让模型生成 N 个回答，用 Reward Model 或 LLM-as-a-Judge 选最好的一个作为 SFT 目标。这是提升 SFT 质量最有效的手段之一。

### 2.2 数据配比的黄金法则
不要把所有数据一股脑丢进去。
*   **通用能力 (General)**: ~20% (保持基础智商)
*   **数学/代码 (Math/Code)**: ~30% (提升逻辑推理，即使不是数学模型，这对 CoT 也有帮助)
*   **特定领域 (Domain)**: ~40% (你的业务目标)
*   **安全/拒答 (Safety)**: ~10% (防止模型变成疯子)

### 2.3 数据的“去毒”与清洗
*   **Perplexity 过滤**：用小模型计算 PPL，剔除异常高（乱码）或异常低（废话）的样本。
*   **De-duplication (去重)**：不仅是文本去重，而是 **语义去重**（MinHash LSH 或 Embedding Clustering）。太多相似数据会导致模型过拟合特定的句式。

---

## 3. Training Dynamics：训练动力学与黑魔法

代码跑通不代表训练正确。

### 3.1 序列打包 (Sequence Packing) 的深坑
为了提高 GPU 利用率，我们通常将多条短数据拼接到 `max_length`（如 4096）。
*   **常规做法**：直接拼接，使用 `eos_token` 分隔。
*   **Solid 做法**：
    1.  **Block Diagonal Attention Mask**：必须修改 Attention Mask，确保 Sample B 看不到 Sample A 的 token。否则模型会学习到错误的因果关系（Cross-contamination）。
    2.  **Position ID Reset**：Sample B 的第一个 token 的 Position ID 应该是 0，而不是接在 Sample A 后面。这对于 RoPE 位置编码至关重要。
    *   *注*：现代框架（如 Axolotl, LLaMA-Factory）通常支持，但务必检查配置是否开启。

### 3.2 损失函数的细节 (Loss Masking)
*   **User Prompt 不算 Loss**：这是常识，但必须确认。只计算 Assistant 回复部分的 Loss。
*   **加权 Loss (Weighted Loss)**：对于关键数据（如复杂的推理步骤），可以人为增加 Loss 权重；对于噪音较大的数据降低权重。

### 3.3 噪声嵌入 (NEFTune)
*   **原理**：在 Embedding 层加入均匀分布的噪声。
*   **效果**：虽然增加了训练 Loss，但能显著防止过拟合，提升模型在对话中的**鲁棒性**和**生成多样性**。
*   **配置**：通常 `noise_alpha` 设置为 5。

```python
# NEFTune 伪代码示意
embeddings = model.embed_tokens(input_ids)
if training:
    noise = torch.rand_like(embeddings) * 2 - 1 # [-1, 1]
    input_embeddings = embeddings + noise * (alpha / sqrt(seq_len * dim))
```

### 3.4 学习率与调度器
*   **Cosine Decay**：SFT 的标准配置。
*   **Min LR**：不要衰减到 0。设置为 Max LR 的 10% 左右，防止模型在训练后期“死”在一个局部最优解，失去泛化能力。
*   **Warmup**：必须有。前 3-5% 的步数用于预热，让优化器状态稳定。

---

## 4. 避免“对齐税” (Alignment Tax)

SFT 容易导致模型变“笨”（forgetting）。

### 4.1 LoRA vs Full Fine-tuning
*   **Full Fine-tuning**：上限最高，但容易破坏预训练知识。需要极小的学习率（如 1e-6, 2e-6）。
*   **LoRA**：更稳健。
    *   **Target Modules**：`all-linear` (q,k,v,o,gate,up,down) 是必须的。只微调 q,v 已经是过时的做法。
    *   **Rank & Alpha**：`r=64, alpha=128` 或 `r=128, alpha=256`。让 alpha/r = 2 有助于梯度流。

### 4.2 持续预训练回放 (Replay Buffer)
如果在特定领域 SFT 后发现通用能力下降（如不再会写 SQL），在 SFT 数据中混入 1-5% 的原始预训练数据（如通用网页文本、通用代码）可以有效缓解灾难性遗忘。

---

## 5. 评估：如何知道模型是 Solid 的？

Loss 曲线下降只代表模型“背”住了数据，不代表它学会了。

### 5.1 训练时评估 (Eval during Train)
*   不要只看 Training Loss。
*   准备一个 **Held-out Eval Set**（不参与训练的高质量数据集），在训练过程中每隔 N 步计算 Evaluation Loss。
*   **Golden Rule**：一旦 Eval Loss 开始上升（哪怕 Training Loss 还在降），立即停止（Early Stopping）。这是过拟合的明确信号。

### 5.2 综合能力测试
*   **IFEval (Instruction Following Evaluation)**：检测模型是否真的遵循了格式约束（如“字数限制”、“特定格式”）。
*   **GSM8K / HumanEval**：即使不微调数学/代码，也要测这两个指标。如果 SFT 后这两个指标大幅下降，说明模型“脑损伤”了。

---

## 6. SFT-V2 工程检查清单

1.  [ ] **数据**：是否进行了去重？是否混入了通用数据防止遗忘？
2.  [ ] **Tokenizer**：Padding Side 是否为 Right (训练)？BOS/EOS 是否正确添加？
3.  [ ] **Packing**：是否开启了 Attention Mask 隔离和 Position ID 重置？
4.  [ ] **参数**：Target Modules 是否包含了所有 Linear 层？
5.  [ ] **监控**：是否配置了 Eval Set？是否监控了 Grad Norm（梯度范数），防止梯度爆炸？

> **结语**：Post-Train Solid 不仅仅是跑通代码，它是一种对模型权重的敬畏。每一个 Token 的 Loss 都在雕刻模型的思维，工程师的责任是确保每一刀都刻得精准。
