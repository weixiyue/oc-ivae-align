# OC-iVAE：面向 N-CMAPSS DS03 的无监督健康指数构建模型设计

> Operating Condition-aware Identifiable VAE for Unsupervised Health Indicator Construction

---

## 1. 问题设定

- **数据集**：N-CMAPSS DS03，每个 flight cycle 为一个样本
- **任务**：完全无监督条件下，为每台发动机的每个 cycle 输出一个健康指数 HI ∈ [0, 1]
- **评估**：训练不使用任何标签；测试时可与 HI ground truth 对比，评估指标为 Monotonicity、Trendability、Prognosability
- **HI 方向**：1 = 健康，0 = 临近失效（单机趋势优先，不要求跨机绝对可比）

---

## 2. 数据预处理

### 2.1 下采样

原始采样率 1 Hz，下采样因子为 **10**（保留 0.1 Hz），每个 cycle 下采样后约 50～200 个时间步。

### 2.2 变长序列处理

不同 cycle 长度不等，采用 **P90 截断 + zero padding + attention mask** 策略：

- 计算训练集所有 cycle 下采样后长度的 P90 分位数，记为 `L_max`
- 超出 `L_max` 的 cycle 保留**末段**（末段含更多退化信息），截断首段
- 不足 `L_max` 的 cycle 在首部补零，并生成对应 boolean mask（`True` = 有效步，`False` = padding）

```python
# 伪代码示意
def pad_and_mask(cycle: np.ndarray, L_max: int):
    T, S = cycle.shape
    if T >= L_max:
        x = cycle[-L_max:]          # 保留末段
        mask = torch.ones(L_max, dtype=torch.bool)
    else:
        pad = np.zeros((L_max - T, S))
        x = np.concatenate([pad, cycle], axis=0)   # 首部补零
        mask = torch.zeros(L_max, dtype=torch.bool)
        mask[L_max - T:] = True
    return torch.tensor(x, dtype=torch.float32), mask
```

### 2.3 传感器选择与标准化

- 剔除方差接近零的常数传感器（标准差 < 1e-3 阈值）
- 对保留传感器按**训练集 unit 级均值和标准差**做 z-score 标准化（每个 unit 独立统计，避免跨 unit 的绝对量纲差异干扰编码器）

### 2.4 辅助变量 u 的构造

对每个 cycle，计算该 cycle 内所有时间步工况向量 W = [alt, XM, TRA, T2] 的均值和标准差：

$$u = [\underbrace{\bar{W}_{\text{alt}}, \bar{W}_{\text{XM}}, \bar{W}_{\text{TRA}}, \bar{W}_{\text{T2}}}_{\text{均值 4 维}},\ \underbrace{\sigma_{W_{\text{alt}}}, \sigma_{W_{\text{XM}}}, \sigma_{W_{\text{TRA}}}, \sigma_{W_{\text{T2}}}}_{\text{标准差 4 维}},\ \underbrace{\text{Fc}_{\text{onehot}}}_{\text{3 维}}] \in \mathbb{R}^{11}$$

此外，将 W 的时序（shape: `L_max × 4`，与传感器序列同步 padding）单独保存为 `W_seq`。

**实现说明**：`W_seq` 和 `u` 中的 W 统计量均使用全局 z-score 归一化（基于 dev 集所有 cycle 的 W 均值和标准差）。这是实现中为防止 FiLM 数值爆炸的必要修改；原始物理量（如 alt 约 3000m、T2 约 500K）会导致 FiLM 激活值溢出。

---

## 3. 模型结构

### 3.1 总览

```
输入 x ∈ R^{L_max × S}，mask ∈ {0,1}^{L_max}
辅助变量 u ∈ R^{11}
工况时序 W_seq ∈ R^{L_max × 4}
                    │
           ┌────────▼────────┐
           │  Shared TCN     │   前 (N-1) 层 Dilated Causal Conv
           │  Backbone       │   输出 h ∈ R^{L_max × C_hidden}
           └────────┬────────┘
                    │
        ┌───────────┴──────────────┐
        │                          │
┌───────▼────────┐        ┌────────▼────────┐
│  z_d Branch    │        │  z_c Branch     │
│  (1 层 TCN,    │        │  (1 层 TCN,     │
│   无条件化)    │        │   FiLM by u)    │
└───────┬────────┘        └────────┬────────┘
        │                          │
   Global AvgPool             Global AvgPool
        │                          │
  Linear → (μ_d, log σ_d²)   Linear → (μ_c, log σ_c²)
        │                          │
        └──────────┬───────────────┘
                   │
          ┌────────▼────────┐
          │  Prior Network  │   u → MLP → (μ_d(u), log σ_d²(u))
          │  (双头 MLP)     │           → (μ_c(u), log σ_c²(u))
          └────────┬────────┘
                   │
          重参数化采样 z_d, z_c
                   │
          ┌────────▼────────┐
          │   TCN Decoder   │  输入: [z_d; z_c] repeat → L_max
          │                 │  可选: 首层拼接 W_summary
          └────────┬────────┘
                   │
               重建 x̂ ∈ R^{L_max × S}
                   │
        ┌──────────┴──────────┐
        │                     │
   HI 路径               重建损失
z_d → w⊤z_d → HI_raw
```

### 3.2 Shared TCN Backbone

| 参数 | 取值 |
|------|------|
| 输入通道数 | S（保留的传感器数，DS03 通常 14～18） |
| 隐藏通道数 C_hidden | 64 |
| 层数 N | 6 |
| 膨胀因子序列 | [1, 2, 4, 8, 16, 32] |
| 卷积核大小 | 3 |
| 激活函数 | GELU |
| 归一化 | LayerNorm（channel-wise，兼容变长+mask） |
| Dropout | 0.1 |
| Causal padding | 是（保证不泄露未来信息） |

有效感受野 = $(k-1) \times \sum d_i + 1 = 2 \times (1+2+4+8+16+32) + 1 = 127$ 步，覆盖 `L_max` 的绝大部分范围。

每一层结构（残差块）：

```
输入 h_in
  │
  ├─ Conv1d(dilation=d) → LayerNorm → GELU → Dropout
  ├─ Conv1d(dilation=d) → LayerNorm → GELU → Dropout
  │
  └─ 残差连接（如输入输出通道不同则用 1×1 Conv 投影）
输出 h_out
```

**Mask 处理**：在每个 LayerNorm 之前，将 padding 位置（`mask=False`）的激活值强制置零，防止 padding 噪声传播。

### 3.3 z_d Branch（退化分支）

```
输入: h ∈ R^{L_max × C_hidden}（来自 Backbone）
  │
  ├─ 1 层 TCN Block（dilation=1，无 FiLM）
  │
  ├─ Masked Global Average Pooling
  │    avg = sum(h * mask, dim=T) / sum(mask, dim=T)
  │    输出 ∈ R^{C_hidden}
  │
  └─ Linear(C_hidden → 2×d_z_d)  →  μ_d ∈ R^{d_z_d},  log σ_d² ∈ R^{d_z_d}
```

- `d_z_d = 2`（退化子空间维度）
- **不施加任何工况条件化**，保证 z_d 只从传感器信号中提取信息

### 3.4 z_c Branch（工况分支）

```
输入: h ∈ R^{L_max × C_hidden}（来自 Backbone）
      u ∈ R^{11}（归一化 W 均值 4 维 + 归一化 W 标准差 4 维 + Fc one-hot 3 维）
  │
  ├─ 1 层 TCN Block（dilation=1）
  │    h_branch = TCNResBlock(h, mask)
  │
  ├─ FiLM Generator: MLP(u → R^{2 × C_hidden})
  │    输出 γ ∈ R^{C_hidden}，β ∈ R^{C_hidden}
  │
  ├─ TCNResBlock 之后施加 FiLM（不是在 LayerNorm 内部）
  │    h_film = γ.unsqueeze(1) ⊙ h_branch + β.unsqueeze(1)
  │
  ├─ Masked Global Average Pooling
  │
  └─ Linear(C_hidden → 2×d_z_c)  →  μ_c ∈ R^{d_z_c},  log σ_c² ∈ R^{d_z_c}
```

- `d_z_c = 8`（工况子空间维度，需覆盖 DS03 中飞行条件的主要变化方向）
- FiLM Generator 使用 2 层 MLP（`11 → 64 → 2×C_hidden`），输入为辅助变量 `u`，即归一化 W 统计量和 Fc one-hot

### 3.5 Prior Network（双头 MLP）

接受辅助变量 $u \in \mathbb{R}^{11}$，分别输出 z_d 和 z_c 的条件先验参数：

```python
class PriorNetwork(nn.Module):
    def __init__(self, u_dim=11, d_zd=2, d_zc=8, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(u_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU()
        )
        # z_d 先验头：小权重初始化，允许微弱工况调制
        self.head_d = nn.Linear(hidden, 2 * d_zd)
        nn.init.normal_(self.head_d.weight, std=0.01)

        # z_c 先验头：正常初始化，强工况调制
        self.head_c = nn.Linear(hidden, 2 * d_zc)

    def forward(self, u):
        h = self.shared(u)
        d_out = self.head_d(h)
        c_out = self.head_c(h)
        mu_d_prior, log_var_d_prior = d_out.chunk(2, dim=-1)
        mu_c_prior, log_var_c_prior = c_out.chunk(2, dim=-1)
        return mu_d_prior, log_var_d_prior, mu_c_prior, log_var_c_prior
```

**非对称初始化的理由**：z_d 的先验头小权重初始化，使得先验均值 $\mu_d(u) \approx 0$，方差 $\sigma_d^2(u) \approx 1$——即初始时 z_d 的先验接近标准正态，不被工况强力牵引。z_d 的实际退化信息需从观测 x 中提取，而非由 u 决定。

### 3.6 重参数化采样

$$z_d = \mu_d + \sigma_d \odot \epsilon_d, \quad \epsilon_d \sim \mathcal{N}(0, I)$$
$$z_c = \mu_c + \sigma_c \odot \epsilon_c, \quad \epsilon_c \sim \mathcal{N}(0, I)$$

### 3.7 TCN Decoder

```
输入: z = [z_d; z_c] ∈ R^{d_zd + d_zc}
  │
  ├─ 可选：拼接 W_summary → z_cond = [z; W_summary] ∈ R^{d_zd + d_zc + 8}
  │
  ├─ Linear(d_zd + d_zc [+ 8] → C_hidden)
  │
  ├─ Repeat / Unsqueeze → R^{1 × C_hidden} → Expand → R^{L_max × C_hidden}
  │   （将 z 广播为序列，作为 decoder 初始特征图）
  │
  ├─ 4 层 TCN Block（dilation=[1,2,4,8]，非 causal，双向感受野）
  │   （decoder 重建时可使用双向上下文，不存在信息泄露问题）
  │
  └─ Linear(C_hidden → S)  →  x̂ ∈ R^{L_max × S}
```

- Decoder 使用**非 causal** TCN（双向感受野），因为重建任务不需要自回归约束
- 层数少于 encoder（4 层 vs 6 层），避免 decoder 过强导致 latent 信息量不足（posterior collapse 的常见来源）

### 3.8 HI 导出头

```python
class HIHead(nn.Module):
    def __init__(self, d_zd=2):
        super().__init__()
        # 可学习线性投影，无偏置（偏置会影响归一化后的范围）
        self.proj = nn.Linear(d_zd, 1, bias=False)

    def forward(self, z_d):
        return self.proj(z_d).squeeze(-1)   # → scalar HI_raw per sample
```

投影向量 $w \in \mathbb{R}^2$ 的方向在训练中由单调性损失自动学习，无需手动指定。

---

## 4. 损失函数

### 4.1 重建损失

仅在有效时间步上计算（mask 覆盖 padding 位置）：

$$\mathcal{L}_{\text{recon}} = \frac{1}{B} \sum_{b=1}^{B} \frac{\sum_{t=1}^{L} \text{mask}_t \cdot \|x_t^{(b)} - \hat{x}_t^{(b)}\|_2^2}{\sum_{t=1}^{L} \text{mask}_t}$$

### 4.2 iVAE ELBO（KL 散度）

KL 散度对比的是 **posterior** $q_\phi(z|x,u)$ 和 **conditional prior** $p_\theta(z|u)$（而非标准正态）：

$$\mathcal{L}_{\text{KL}_d} = \text{KL}\left(\mathcal{N}(\mu_d, \sigma_d^2) \| \mathcal{N}(\mu_d^{\text{prior}}(u), \sigma_d^{2,\text{prior}}(u))\right)$$

$$\mathcal{L}_{\text{KL}_c} = \text{KL}\left(\mathcal{N}(\mu_c, \sigma_c^2) \| \mathcal{N}(\mu_c^{\text{prior}}(u), \sigma_c^{2,\text{prior}}(u))\right)$$

两个 Gaussian KL 均有解析解：

$$\text{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \frac{1}{2}\left[\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{\sigma_2^2} - 1\right]$$

权重非对称：$\beta_d = 0.5$，$\beta_c = 1.0$（z_d 的 KL 惩罚更小，允许 z_d 保留更多来自 x 的退化信息）。

### 4.3 HSIC 解耦损失

使用 mini-batch HSIC 估计，RBF kernel，带宽用中位数启发式（median heuristic）：

```python
def batch_hsic(z_d, z_c, sigma=None):
    """
    z_d: (B, d_zd), z_c: (B, d_zc)
    返回标量 HSIC 估计值
    """
    B = z_d.shape[0]
    # RBF kernel
    def rbf_kernel(X, sigma):
        dists = torch.cdist(X, X, p=2).pow(2)
        if sigma is None:
            sigma = dists.median().detach() + 1e-6
        return torch.exp(-dists / (2 * sigma**2))

    Kd = rbf_kernel(z_d, sigma)
    Kc = rbf_kernel(z_c, sigma)

    # 中心化
    H = torch.eye(B, device=z_d.device) - 1.0/B
    Kd_c = H @ Kd @ H
    Kc_c = H @ Kc @ H

    return (Kd_c * Kc_c).sum() / (B - 1)**2
```

### 4.4 单调性损失（Pairwise Ranking）

对 batch 内同一 unit 的有效 cycle 对 $(i, j)$（$cycle_j - cycle_i \ge \Delta t$，要求 HI 单调递减）：

$$\mathcal{L}_{\text{mono}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \max\left(0,\ \text{HI}_j - \text{HI}_i + \epsilon\right)$$

- $\mathcal{P}$：batch 内同 unit 的有效排序对集合，满足 $cycle_j - cycle_i \ge \Delta t$
- 当前实现没有 gap 上界过滤，也没有平方项
- $\Delta t = 3$：最小 cycle 间隔，可通过配置 `ranking_delta_t` 调整
- $\epsilon = 0.01$：margin，确保相邻 cycle 的 HI 有一定间隔而非仅大于零

```python
def monotonicity_loss(hi, unit_ids, cycle_ids, delta_t=3, margin=0.01):
    """
    hi: (B,) 标量 HI
    unit_ids: (B,) unit 编号
    cycle_ids: (B,) cycle 编号
    """
    unit_eq = unit_ids[:, None] == unit_ids[None, :]
    gap = cycle_ids[None, :] - cycle_ids[:, None]
    valid = unit_eq & (gap >= delta_t)
    later_minus_earlier = hi[None, :] - hi[:, None]
    return relu(later_minus_earlier + margin)[valid].mean()
```

### 4.5 Spearman 代理损失（辅助）

使用软排序近似（soft-rank），温度参数 $\tau = 0.1$，对每个 unit 独立计算 HI 下降趋势与 cycle progression 的近似 Spearman 相关：

1. 对 `-hi` 计算 soft rank；这里显式取负号，使 HI 随 cycle 递减时 rank 与 cycle rank 同向增加。
2. 对 `cycle_id` 计算 ascending soft rank。
3. 损失为 `1 - pearson_corr(rank(-hi), rank(cycle_id))`。

这等价于最大化 HI 下降过程与 cycle progression 之间的 Spearman 相关。

```python
def soft_spearman_loss(hi, cycle_ids, unit_ids, tau=0.1):
    """
    对每个 unit 计算 rank(-HI) 与 rank(cycle_id) 的近似 Spearman 相关
    """
    loss = torch.tensor(0.0, device=hi.device)
    count = 0
    for uid in unit_ids.unique():
        mask = unit_ids == uid
        hi_u = hi[mask]
        cyc_u = cycle_ids[mask].float()
        if hi_u.shape[0] < 4:
            continue
        # Soft rank：对每个元素计算其在序列中的软排名
        def soft_rank(x, tau):
            diff = x.unsqueeze(0) - x.unsqueeze(1)   # (N, N)
            return torch.sigmoid(diff / tau).sum(dim=1)  # 近似排名

        rank_hi = soft_rank(-hi_u, tau)
        rank_cyc = soft_rank(cyc_u, tau)
        # Pearson on ranks ≈ Spearman
        rho = pearson_corr(rank_hi, rank_cyc)
        loss = loss + (1 - rho)
        count += 1
    return loss / max(count, 1)

def pearson_corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)
```

### 4.6 总损失

$$\boxed{\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta_d \cdot \mathcal{L}_{\text{KL}_d} + \beta_c \cdot \mathcal{L}_{\text{KL}_c} + \lambda_{\text{HSIC}} \cdot \mathcal{L}_{\text{HSIC}} + \lambda_1 \cdot \mathcal{L}_{\text{mono}} + \lambda_2 \cdot \mathcal{L}_{\text{spearman}}}$$

| 超参数 | 初始推荐值 | 说明 |
|--------|-----------|------|
| $\beta_d$ | 0.5 | z_d 的 KL 权重，偏小以保留退化信息 |
| $\beta_c$ | 1.0 | z_c 的 KL 权重，偏大以强化工况解耦 |
| $\lambda_{\text{HSIC}}$ | 1.0 | 解耦约束权重 |
| $\lambda_1$ | 2.0 | Pairwise Ranking Loss 权重 |
| $\lambda_2$ | 0.5 | Spearman 代理损失权重（辅助） |
| $\epsilon$ | 0.01 | Ranking Loss margin |

---

## 5. 训练策略

### 5.1 Batch 构造（Unit-Stratified Sampling）

```python
class UnitStratifiedSampler(Sampler):
    """
    每个 batch 由 K 个 unit 各贡献 M 个 cycle 组成
    保证 pairwise ranking loss 有足够的 within-unit pair
    """
    def __init__(self, unit_ids, K=8, M=12):
        self.unit_ids = unit_ids
        self.K = K
        self.M = M
        self.units = np.unique(unit_ids)

    def __iter__(self):
        units_shuffled = np.random.permutation(self.units)
        for i in range(0, len(units_shuffled), self.K):
            batch_units = units_shuffled[i:i+self.K]
            batch_indices = []
            for uid in batch_units:
                idx = np.where(self.unit_ids == uid)[0]
                chosen = np.random.choice(idx, size=min(self.M, len(idx)), replace=False)
                batch_indices.extend(chosen.tolist())
            yield batch_indices
```

- Batch size = $K \times M$，实际约 96～192
- 每个 epoch 重新随机分配 unit 组合，保证全局工况分布覆盖

### 5.2 三阶段训练

| 阶段 | Epoch 范围 | 开启的损失项 | 说明 |
|------|-----------|-------------|------|
| **热身期** | 1 ～ 10 | $\mathcal{L}_{\text{recon}}$ only | encoder/decoder 先学会重建，避免冷启动时 KL 和排序损失干扰表示学习 |
| **过渡期** | 11 ～ 25 | + KL（线性 annealing） + $\mathcal{L}_{\text{HSIC}}$ | $\beta$ 从 0 线性增加到目标值，防止 posterior collapse；同步引入解耦约束 |
| **全损失期** | 26 ～ end | 全部损失项 | 联合优化，单调性损失完整引入 |

KL annealing 实现：

```python
def kl_weight(epoch, warmup_start=10, warmup_end=25, beta_target=1.0):
    if epoch < warmup_start:
        return 0.0
    elif epoch < warmup_end:
        return beta_target * (epoch - warmup_start) / (warmup_end - warmup_start)
    else:
        return beta_target
```

### 5.3 优化器与学习率

| 参数 | 取值 |
|------|------|
| 优化器 | AdamW |
| 学习率 | 3e-4 |
| Weight decay | 1e-4 |
| LR Scheduler | CosineAnnealingLR（T_max = total epochs） |
| Gradient clipping | max_norm = 1.0 |
| Total epochs | 100 |
| Early stopping | patience = 15，监控验证集 $\mathcal{L}_{\text{recon}}$ |

---

## 6. 推理与 HI 后处理

### 6.1 推理流程

推理时使用 posterior 均值（不采样），即 $z_d = \mu_d$，$z_c = \mu_c$，减少随机性：

```python
@torch.no_grad()
def get_hi(model, dataloader):
    model.eval()
    results = {}   # {unit_id: {cycle_id: HI_raw}}
    for x, mask, u, W_seq, unit_ids, cycle_ids in dataloader:
        mu_d, _, mu_c, _ = model.encode(x, mask, W_seq)   # 取均值
        hi_raw = model.hi_head(mu_d).cpu().numpy()
        for b in range(len(unit_ids)):
            uid, cid = unit_ids[b].item(), cycle_ids[b].item()
            results.setdefault(uid, {})[cid] = hi_raw[b]
    return results
```

### 6.2 Per-unit Min-Max 归一化

```python
def normalize_hi(hi_dict):
    """hi_dict: {unit_id: {cycle_id: HI_raw}}"""
    hi_norm = {}
    for uid, cycles in hi_dict.items():
        cids = sorted(cycles.keys())
        vals = np.array([cycles[c] for c in cids])
        v_min, v_max = vals.min(), vals.max()
        if v_max - v_min < 1e-8:
            norm_vals = np.ones_like(vals)
        else:
            norm_vals = 1.0 - (vals - v_min) / (v_max - v_min)
            # 1.0 - (...) 使得方向为：1=健康, 0=临近失效
        hi_norm[uid] = {c: v for c, v in zip(cids, norm_vals)}
    return hi_norm
```

注意：归一化**仅在测试时**按 unit 独立执行，训练时不做。

### 6.3 EWMA 平滑

```python
def ewma_smooth(hi_norm_dict, alpha=0.3):
    """alpha: 平滑因子，越小越平滑；建议范围 [0.2, 0.5]"""
    hi_smooth = {}
    for uid, cycles in hi_norm_dict.items():
        cids = sorted(cycles.keys())
        vals = [cycles[c] for c in cids]
        smoothed = [vals[0]]
        for v in vals[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        hi_smooth[uid] = {c: v for c, v in zip(cids, smoothed)}
    return hi_smooth
```

---

## 7. 评估指标

使用 PHM 领域标准三指标，均在 per-unit 归一化和 EWMA 平滑后的 HI 上计算：

**Monotonicity**（越接近 1 越好）：

$$M_k = \frac{|\{t: HI_{t+1}^{(k)} < HI_t^{(k)}\}|}{n_k - 1} - \frac{|\{t: HI_{t+1}^{(k)} > HI_t^{(k)}\}|}{n_k - 1}$$

$$M = \frac{1}{N_u} \sum_{k=1}^{N_u} M_k$$

取值范围为 $[-1, 1]$；完美单调递减为 1.0。

**Trendability**（越接近 1 越好）：

$$T_k = |\text{Pearson}(HI^{(k)}, \text{arange}(n_k))|$$

$$T = \frac{1}{N_u} \sum_{k=1}^{N_u} T_k$$

即 HI 序列与时间索引的绝对 Pearson 相关，不是 Spearman 相关；取值范围为 $[0, 1]$。

**Prognosability**（越接近 1 越好）：

对每个 unit 收集末端值 $eol_k = HI_k[-1]$：

$$P = \exp\left(-\frac{\text{std}(\{eol_k\}_{k=1}^{N_u})}{\text{mean}(\{eol_k\}_{k=1}^{N_u}) + 10^{-8}}\right)$$

**MAE**（有 ground truth HI 时计算）：

$$MAE = \text{mean}(|HI_{\text{pred}} - HI_{\text{gt}}|)$$

若某个 split 没有 ground truth HI，则 MAE 记为不可用。

---

## 8. 超参数汇总

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| 下采样因子 | 10 | 预处理 |
| L_max | P90 of cycle lengths | 变长处理 |
| S | 14～18（DS03 有效传感器数） | 输入维度 |
| C_hidden | 64 | TCN 隐藏通道数 |
| Backbone 层数 N | 6 | 膨胀序列 [1,2,4,8,16,32] |
| Decoder 层数 | 4 | 膨胀序列 [1,2,4,8] |
| d_z_d | 2 | 退化子空间维度 |
| d_z_c | 8 | 工况子空间维度 |
| u_dim | 11 | 辅助变量维度 |
| K（stratified batch） | 7 | 每 batch 的 unit 数；DS03 只有 9 个 unit，val 占 2 个后训练集剩 7 个 |
| M（stratified batch） | 12 | 每 unit 在 batch 内的 cycle 数 |
| $\beta_d$ | 0.5 | z_d KL 权重 |
| $\beta_c$ | 1.0 | z_c KL 权重 |
| $\lambda_{\text{HSIC}}$ | 1.0 | HSIC 解耦权重 |
| $\lambda_1$ | 2.0 | Pairwise Ranking 权重 |
| $\lambda_2$ | 0.5 | Spearman 代理权重 |
| Ranking margin $\epsilon$ | 0.01 | 单调性 margin |
| Ranking $\Delta t$ | 3 | 排序对步长 |
| Spearman $\tau$ | 0.1 | 软排序温度 |
| EWMA $\alpha$ | 0.3 | 后处理平滑因子 |
| 学习率 | 3e-4 | AdamW |
| Weight decay | 1e-4 | AdamW |
| Total epochs | 100 | 训练总轮数 |
| Early stopping | 可选项，默认关闭 | val set 只有 2 个 unit，val_recon 波动大，建议训练满 total_epochs |

---

## 9. 代码目录结构建议

```
oc_ivae/
├── data/
│   ├── dataset.py          # N-CMAPSS DS03 Dataset，输出 (x, mask, u, W_seq, unit_id, cycle_id)
│   ├── sampler.py          # UnitStratifiedSampler
│   └── preprocess.py       # 下采样、截断、z-score、u 构造
├── model/
│   ├── backbone.py         # Shared TCN Backbone（含 CausalDilatedConv、残差块、masked layernorm）
│   ├── branches.py         # z_d Branch、z_c Branch（含 FiLM Generator）
│   ├── prior.py            # PriorNetwork（双头 MLP）
│   ├── decoder.py          # TCN Decoder
│   ├── hi_head.py          # HIHead（可学习线性投影）
│   └── oc_ivae.py          # 主模型，组装以上模块，forward 返回所有中间量
├── loss/
│   ├── recon.py            # masked MSE
│   ├── kl.py               # Gaussian KL（posterior vs conditional prior）
│   ├── hsic.py             # mini-batch HSIC
│   ├── monotonicity.py     # Pairwise Ranking Loss
│   ├── spearman.py         # Soft Spearman Loss
│   └── total.py            # 损失汇总，支持阶段性权重调度
├── train/
│   ├── trainer.py          # 三阶段训练循环，KL annealing，early stopping
│   └── scheduler.py        # kl_weight 调度函数
├── eval/
│   ├── inference.py        # 推理，返回 HI_raw
│   ├── postprocess.py      # per-unit min-max 归一化 + EWMA 平滑
│   └── metrics.py          # Monotonicity、Trendability、Prognosability
└── configs/
    └── ds03_default.yaml   # 超参数配置文件
```

---

*文档版本：v1.0 | 基于 N-CMAPSS DS03 | PyTorch 实现*
