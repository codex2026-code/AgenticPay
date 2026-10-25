# 最小信念追踪与维护对照实验

目标：检验**显式回传本方上一轮信念，是否改变保留价格估计、跨轮行为和协商结果**。
复用 AgenticPay 的 BuyerAgent、SellerAgent、ConversationMemory 和 Task1BasicPriceNegotiation。
仅研究单买家、单卖家、单商品；不训练模型，不引入 judge，不扩展 benchmark。

基线源码：`7c3244d9e91fab5197884f87bc68c8414b3e4a14`。
已执行的检查见 [VALIDATION.md](VALIDATION.md)。**尚未运行真实 LLM 实验；mock 数值不构成研究结果。**

## 两组和最小运行预算

| 设置 | context | maintained |
|---|---|---|
| 公开对话 | 完整保留 | 完整保留 |
| 当前角色的真实私有价格 | 仅本方可见 | 仅本方可见 |
| 每轮结构化 mental_model | 记录至评估轨迹 | 记录至评估轨迹 |
| 本方上一轮信念 | 不回传，始终为 null | 仅回传紧邻上一轮的有效 JSON |
| 对手 mental_model、真实底价、评估误差 | 不可见 | 不可见 |
| 提示模板、模型、采样参数、轮数、协议 | 相同 | 相同 |

`context` 保留原始“公开历史重新推断”的记忆机制，但将自由文本 mental_model
改为共同 JSON schema；**不是逐字不变的原始提示 baseline**。默认不启用 tracker 的
Agent 仍走原来的代码路径。两组首轮 prompt 相同，随后唯一的模板输入变化是
`<previous_belief>` 内的值。两组均要求根据新证据修正估计，不强制区间收缩。

双方在一局中同时采用所选模式，因此 live 对比衡量**联合启用显式状态的总效果**，
不能分离买方与卖方的单独贡献。显式状态增加输入 token；本实验不主张等 token 效率优势。

默认三个固定场景，统一公开标价 120，商品描述相同；场景 ID 和可成交性不进入 prompt：

| 评估用场景 | 买方最高价 B | 卖方最低价 S | 用途 |
|---|---:|---:|---|
| wide_overlap | 92 | 48 | 宽成交区间 |
| narrow_overlap | 76 | 70 | 窄成交区间 |
| no_overlap | 61 | 68 | 检查不合理成交；不成交并非失败 |

默认 1 个 seed、两种模式、最多 6 轮：6 局，最多 72 次 live 模型调用。
默认另取 2 个固定前缀探针对，每对 2 次调用，合计最多 76 次。
`--seeds 11 29` 为 12 局，最多 144 + 4 次调用。先跑一个 seed，再决定是否增加。
相同 seed 在匹配的组别/角色/轮次使用相同请求 seed；两个角色、两组、各局使用独立实例。
模式运行顺序按 case/seed 确定性打乱。不同 seed 是生成重复，不是新增独立场景。

## 运行

从仓库根目录执行。Python >= 3.10。这里是文本实验，无需安装完整 requirements.txt
中的 torch、transformers、vLLM 或 SGLang；已运行的本地模型服务可直接复用。

```bash
python3 -m venv .venv-belief
source .venv-belief/bin/activate
python -m pip install -r experiments/belief_tracking/requirements.txt pytest

# 不连接模型、不消耗推理额度：先检查流程
python -m pytest -q tests/test_belief_tracking.py
python -m experiments.belief_tracking.run \
  --backend mock --seeds 11 29 \
  --out experiments/belief_tracking/runs/smoke
```

连接已有的 OpenAI-compatible Chat Completions 服务，例如学校服务器上的 Qwen 服务。
`MODEL_ID` 必须使用该服务实际提供的模型名称；本代码不下载或启动大模型：

```bash
MODEL_ID='your-served-model-id'
python -m experiments.belief_tracking.run \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --model "$MODEL_ID" \
  --seeds 11 --max-rounds 6 --temperature 0 --max-tokens 1024 \
  --probe-count 2 \
  --out experiments/belief_tracking/runs/model_seed11
```

如果服务要求认证，在运行前设置 `OPENAI_API_KEY`；不要将密钥写入仓库。
`openai` 是接口类型，不限制模型厂商。程序仅发送单次独立 Chat Completions 请求，
不使用共享 conversation/thread ID。不支持 seed 的服务需显式加 `--omit-seed`，
不会自动删参数后重试。API 错误会保留失败记录；不会换模型、重试到成功或伪造回答。

复现建议：使用固定 checkpoint/服务版本，记录 tokenizer/chat template 和服务启动参数，
尽量选用不依赖隐藏长推理输出的文本配置。接口不支持 temperature/max_tokens 时需先适配接口。
temperature=0 和相同 seed 仍不保证远端推理逐位一致；记录中保存返回模型、fingerprint、
finish_reason 和 usage（若服务提供）。模板/源码哈希、Python 和依赖版本保存在 manifest。

输出路径必须是新目录，避免混入旧实验。真实运行格式失败时先检查 raw_response、
finish_reason 和 error，再决定修改提示或增加输出预算；如改设置，两组一起重跑。
有输出格式/API 错误时进程退出码为 1，仍保存已有轨迹与报告。

重新计算指标不需要调用模型：

```bash
python -m experiments.belief_tracking.evaluate \
  experiments/belief_tracking/runs/model_seed11
```

## 每轮记录及状态更新

```json
{
  "price_low": 60.0,
  "price_high": 80.0,
  "confidence": 0.6,
  "opponent_strategy": "holding_firm",
  "own_strategy": "small concession",
  "evidence": "The opponent repeated the same public offer."
}
```

买方估计的是卖方最低接受价，卖方估计的是买方最高接受价。
`confidence` 明确定义为“真实保留价落在自己所报区间内”的概率，取值 [0,1]。
策略标签为 `unknown/anchoring/conceding/holding_firm/value_argument/pressure/other`。
这里只采集简短策略和依据摘要，不要求长篇推理过程。

`maintained` 在解析成功后原样保存这一 JSON，下一轮仅注入当前角色自己的状态；
`context` 只留下评估轨迹，不将轨迹作为下一轮输入。
无效 JSON、倒置区间、非法置信度、NaN/Infinity 等都记为缺失，不调用另一个 LLM 修复，
不猜测区间。若公开消息合法，协商继续；紧邻上一轮信念无效时下轮状态为 null，
不会悄悄复用更早的估计。reset 会清除上一局的信念和轨迹。

## 隔离边界和协商协议

- 每个 Agent 只初始化本方价格；共同 product_info/environment_info 无私有价格。
- 传入 Agent 的 history 仅包含 role/content/round；current_state 只保留公开报价、轮次、状态。
  原始 environment 对象、reward、info、oracle、双方合并轨迹都不会进入模型输入。
- 只有合法 `<message>` 会传给对方并进入 ConversationMemory。
  缺块、未闭合标签、嵌套私有块、额外输出、多重或缺失角色报价会终止该局，
  **没有把 raw response 当作 message 的兜底路径**。
- 真实标签仅在离线 evaluate 中与已记录估计相连；评价结果不反馈到 Agent。
  信念模块和模型 backend 不接收 case 或对手真实价格。
- `runs/` 默认不入 Git；运行目录权限 0700。完整 prompt/response、oracle 和评分文件
  是研究者私有评估资料，不得作为某一方 Agent 的检索目录或上下文。

这是**模型输入层面的隔离**，不是同机进程的操作系统沙箱。
Agent 在 `<message>` 中主动透露自己的底价仍可能发生，需检查公开文本；不能把报出一个
恰等于底价的合法 offer 一律判为泄漏。针对管线泄漏的确定性检查是：固定公开历史、
本方私有信息和本方上一轮信念，仅改对手真实底价，本方完整 prompt 必须逐字不变。
测试对两个角色和后续轮次都执行此检查，并验证双方私有策略状态不交叉。

每轮买方先回复，卖方看到本轮买方消息后回复，再调用一次 env.step。
沿用 Task1 的报价交叉成交：seller offer <= buyer offer 即成交，成交价取 seller offer；
`MAKE_DEAL` 不是此环境的决定条件。本实验 tolerance=0，避免“差价接近”形成假成交。
上游轮次检查存在零基下标偏移，runner 将 `max_rounds` 传成 N−1，使 CLI 的 N 确实是 N 轮。
不直接修订上游所有环境。私有价格不用于修复报价；越过本方底价或无合理成交区间的
协议成交会由评估器单独标出，不能算有效成功。

## 指标

令本轮区间为 [L_t,U_t]，中点 m_t=(L_t+U_t)/2，真实对手保留价为 r，公开标价为 P。
指标按角色分开计算，先对一局求均值，再跨局求均值，避免长对话权重更大。
所有缺失值保留为 null/NA，并报告有效信念数、有效相邻轮数和格式失败率。

| 维度 | 指标 | 含义与解释 |
|---|---|---|
| 价格误差 | MAE = mean(abs(m_t−r))；NMAE = MAE/P；末轮 NMAE | 误差越小越好；末轮无效则缺失 |
| 区间 | coverage = mean(1[L_t<=r<=U_t])；width_norm = mean((U_t−L_t)/P) | 宽区间容易覆盖，必须一起看 |
| 置信度 | mean((confidence_t−coverage_t)^2) | 自选区间事件的 Brier；不据小样本宣称整体校准改善 |
| 跨轮一致性 | mean(abs(m_t−m_(t−1))/P)；区间 IoU；策略标签切换率 | 仅相邻有效轮；低漂移可能是错误固化，不设“越小越好”目标 |
| 协商 | 协议成交、有效成交、无效成交、成交价、轮数 | 有效成交需 S<=p<=B；不可成交场景分开解读 |
| 效用 | 买方 B−p，卖方 p−S；无有效成交时双方实现效用记 0 | 不把无效成交的名义负收益当作有效交易 |
| 效率/分配 | 正剩余场景成交率；4(B−p)(p−S)/(B−S)^2 | 后者为归一化 Nash，失败为 0；B<=S 时 NA |
| 协议质量 | own_limit_violation、格式失败率、episode_error | 避免选择性丢弃失败后得出改进结论 |

策略标签无客观策略真值，**不计算“策略判断准确率”**。缺失信念不会被补成 0 误差；
summary 同时记录每项有效局数。配对差值按相同 case+seed 计算 maintained−context。
不同协商长度/终止时刻会影响逐轮平均，须同时看末轮误差、完整轨迹和冻结前缀探针。

## 固定公开对话的状态消融探针

默认从 context 轨迹中选首个可用买方前缀、首个可用卖方前缀，各自冻结完整 prompt。
对每个前缀重新调用两次：一次上一轮信念为 null，一次放入该角色在原 context 轨迹中
紧邻上一轮产生的有效信念。两次使用相同模型、采样设置、请求 seed；除 prior 块外逐字相同。
探针结果不推进环境，也不改写 live 历史；输出估计变化、误差差值、报价变化、策略变化。
若没有有效的第二轮前缀，则探针数为 0，不伪造样本。`--probe-count 0` 可以关闭。

这检查“提供上一轮估计是否影响同一输入条件下的输出”。它不是贝叶斯更新正确性的证明，
也不能单凭输出不变就断言模型没有读取状态。服务端非确定性同样可能影响差值。

判断下一步的标准：首先确保格式/越界失败没有上升；再检查误差改善是否伴随合理宽度、
置信度和成交表现，而非仅仅漂移降低。若误差反而上升且区间不再修正，应优先怀疑
上一轮信念造成锚定。只有少数场景时先逐条审阅，不做显著性或普适性结论。

## 文件与输出

| 文件 | 作用 |
|---|---|
| agenticpay/memory/belief_tracking.py | JSON schema、严格公开消息解析、本方上一轮状态 |
| agenticpay/agents/{buyer,seller}_agent.py | 可选 tracker 接入；未启用时保留原行为 |
| experiments/belief_tracking/run.py | 成对协商、公开投影、固定前缀探针、运行清单 |
| experiments/belief_tracking/models.py | 独立无会话接口；仅流程测试用 MockModel |
| experiments/belief_tracking/evaluate.py | 离线真值比较、逐局指标和报告 |
| tests/test_belief_tracking.py | 隔离、重置、错误处理、协议、手算指标、可复现测试 |

每次运行生成：`manifest.json`（版本/设置）、`oracle.json`（评估标签）、
`turns.jsonl`（完整私有轨迹）、`episodes.jsonl`（公开历史与终止信息）、
`probes.jsonl`（冻结前缀重放）、`scored_turns.jsonl`（逐轮真值/误差）、
`episode_metrics.csv`（逐局结果）、`summary.json`（均值和配对差值）、`report.md`（中文报告）。
MockModel 刻意不使用 previous_belief；两组结果相同只证明这个假模型没有被实现意外偏置，
**不能说明真实 LLM 的显式信念维护有效或无效**。
