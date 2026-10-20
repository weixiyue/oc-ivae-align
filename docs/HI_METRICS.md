# 健康指数评价指标

本文对应当前 [utils/metrics.py](../utils/metrics.py) 和 [evaluate.py](../evaluate.py) 的实现。
公共评价输出六项指标；独立线性探测的回归指标另列，不混入公共六指标。

## 计算口径

每条记录对应一个 `(unit, cycle)`，预测、参考 HI 和 RUL 按键一对一关联。
Residual 短窗口分数先按 flight 取均值，再参与评价，不按窗口数重复计权。
dev 和 test 分别计算指标，公共评价中的 dev 包含网络训练和验证设备。

设设备数为 $U$，设备 $u$ 有 $n_u$ 个按 cycle 排序的 flight，总数 $N=\sum_u n_u$。
$h_{u,t}$ 为最终 HI，$g_{u,t}$ 为参考 HI，$r_{u,t}$ 为 RUL，$c_{u,t}$ 为实际 cycle。
最终 HI 采用全部 dev 原始分数拟合的一个全局 Min–Max：

$$
h=\frac{s-\min(s_{dev})}{\max(s_{dev})-\min(s_{dev})}.
$$

test 仅复用 dev 参数，因此可以超出 $[0,1]$；不逐设备缩放、不平滑、不裁剪。
上式适用于非退化范围；dev 分数恒定时沿用 sklearn 的零范围处理：dev 映射为 0，
test 映射为 $s-\text{dev 常数}$，并在元数据标记退化输出。
公共分数方向约定为健康随退化下降。

## 六项公共指标

| 输出键 | 名称与含义 | 通常期望 | 标签依赖 |
| --- | --- | --- | --- |
| `mae` | 平均绝对误差，衡量与参考 HI 的数值偏差 | 越小越好 | 参考 HI |
| `mape` | 平均绝对百分比误差，衡量相对偏差 | 越小越好 | 非零参考 HI |
| `monotonicity` | 单调性，衡量相邻变化方向是否一致 | 越接近 1 越好，需排除常数输出 | 无 |
| `trendability` | 趋势性，衡量 HI 与 cycle 的秩相关及跨设备方向一致性 | 越接近 1 越好 | cycle |
| `fc` | 失效收敛性，衡量末端 HI 相对整体下降幅度的离散程度 | 正常下降时越接近 1 越好 | 无 |
| `mi` | HI 与 RUL 的互信息变换，衡量统计依赖 | 越接近 1，依赖越强 | RUL |

### MAE 与 MAPE

$$
\mathrm{MAE}=\frac{1}{N}\sum_{u,t}|h_{u,t}-g_{u,t}|.
$$

令 $V=\{(u,t):g_{u,t}\ne0\}$，则

$$
\mathrm{MAPE}=\frac{100}{|V|}\sum_{(u,t)\in V}
\left|\frac{h_{u,t}-g_{u,t}}{g_{u,t}}\right|.
$$

两者均合并全部 cycle 计算，每个 cycle 等权，不是逐设备误差的简单平均。
MAPE 单位为百分比，参考 HI 接近零时会放大误差；恰为零的参考值被排除，并记录有效数量。
缺少参考 HI 时两者为 `null`；MAPE 无非零参考值时也为 `null`。

### Monotonicity：单调性

令 $\Delta h_{u,t}=h_{u,t+1}-h_{u,t}$：

$$
M_u=\frac{\left|\#\{\Delta h_{u,t}\le0\}-\#\{\Delta h_{u,t}>0\}\right|}{n_u-1},
\qquad M=\frac{1}{U}\sum_u M_u.
$$

取值 $[0,1]$。持续上升或下降都可得 1，因此不能单独判断健康方向是否正确。
当前实现将零差分计入非正方向，**恒定 HI 也可能得 1**，代码会记录退化提示。
不足两个 cycle 的设备无法计算；不会静默排除该设备后取平均。

### Trendability：趋势性

$$
\rho_u=\operatorname{Spearman}(h_u,c_u)
=\operatorname{Corr}(\operatorname{rank}(h_u),\operatorname{rank}(c_u)),
\qquad T=\left|\frac{1}{U}\sum_u\rho_u\right|.
$$

Spearman 使用平均秩处理并列值。总体取值 $[0,1]$，是**带符号相关先平均、再取绝对值**，
不是相关绝对值的平均；设备间相反趋势会抵消。逐设备结果保留符号，下降趋势通常为负。
常数 HI 等造成的 NaN 相关按当前协议转为 0，并记录原因。

### FC：失效收敛性

令各设备首末 HI 为 $a_u=h_{u,1}$、$b_u=h_{u,n_u}$：

$$
D=\frac{1}{U}\sum_u(a_u-b_u),\qquad
\sigma_b=\sqrt{\frac{1}{U}\sum_u(b_u-\bar b)^2},\qquad
\mathrm{FC}=\exp\left(-\frac{\sigma_b}{D}\right).
$$

末端标准差采用总体标准差（`ddof=0`）。当 $D>0$ 时，末端越集中、相对下降幅度越大，FC 越接近 1。
这是跨设备指标，不提供逐设备 FC。若 $D\le0$，会记录诊断；公式可能大于 1 或非有限，
不能将这种结果解释为良好退化表征。当前 FC 不应与其他公式的 Prognosability 混用。

### MI：互信息

每个设备先估计 HI 与 RUL 的互信息：

$$
I_u=I(H_u;R_u)=\iint p(h,r)\log\frac{p(h,r)}{p(h)p(r)}\,dh\,dr,
\qquad \mathrm{MI}=\frac{1}{U}\sum_u\left(1-e^{-\widehat I_u}\right).
$$

实际使用 sklearn `mutual_info_regression` 的连续变量近邻估计，默认 3 个近邻，
公共入口固定 `random_state=42`。输出 `mi` 是逐设备变换后的平均，
原始估计另存于 `per_unit.<unit>.mi_raw`；不是对全体样本估计一次互信息。
理论变换范围为 $[0,1)$。高 MI 表示依赖较强，不保证方向正确或与参考 HI 数值接近。
RUL 仅用于评价，不因此进入无监督训练目标。

## 输出与异常诊断

结果写入 `metrics.json` 的 `dev`、`test`，包含总体指标、`per_unit`、样本数及 `issues`。
单调性、趋势性和 MI 按设备等权汇总；MAE/MAPE 按合并 cycle 汇总；FC 使用设备首末值。
非有限指标保存为 JSON `null` 并说明原因，不静默丢弃失败设备。
参考 HI 无 cycle 列时沿用公共逐设备行序推定规则，该语义行序无法独立核实的限制记录在元数据中。

## 独立线性探测：R²、MAE、RMSE

实现见 [utils/linear_probe.py](../utils/linear_probe.py)，入口为
`python -m experiments.evaluate_linear_probe`。
网络冻结，分别以完整健康表征或最终标量 HI 为输入，仅在训练设备拟合
`StandardScaler → LinearRegression`，验证设备不参与拟合。
令 $\hat g_i$ 为线性回归预测、$g_i$ 为参考 HI：

$$
R^2=1-\frac{\sum_i(g_i-\hat g_i)^2}{\sum_i(g_i-\bar g)^2},\qquad
\mathrm{MAE}=\frac{1}{N}\sum_i|g_i-\hat g_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i(g_i-\hat g_i)^2}.
$$

测试主指标合并全部测试 cycle，每 cycle 等权。R² 越高越好，最优为 1，也可为负；
MAE/RMSE 越低越好，RMSE 对大误差更敏感。目标恒定或样本不足两个时，R² 为 `null` 并记录原因。
不对回归预测施加缩放、Sigmoid 或裁剪。

这些是**使用参考标签拟合后的表征评价**，与未经探测的最终 HI 指标分开报告；
结果写入独立 `probe_metrics.json`，不覆盖公共六指标的 `metrics.json`。
