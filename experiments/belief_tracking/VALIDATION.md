# 验证记录：2026-10-25

**真实 LLM 实验：未运行。当前执行环境没有配置模型端点或 API key。**
本次验证只证明实现的流程、隔离约束和指标计算可以工作，不支持“显式信念维护有效/无效”的结论。

执行环境：Python 3.12.14（运行生成的 manifest.json 同时记录依赖版本）；
AgenticPay 基线 `7c3244d9e91fab5197884f87bc68c8414b3e4a14`。

执行命令：

```bash
python -m pytest -q tests/test_belief_tracking.py
python -m experiments.belief_tracking.run \
  --backend mock --seeds 11 29 \
  --out experiments/belief_tracking/runs/smoke
```

结果：**33 passed**。其中端到端测试还在两个独立临时目录中重复运行上述同等配置，
验证完整 summary 相同；不是仅验证同一次运行的字段存在。

| 检查 | 已验证事实 |
|---|---|
| 隐藏价格隔离 | 固定本方信息/公开历史，仅替换对手真实底价，买方和卖方在后续轮次的完整 prompt 均逐字不变 |
| 状态隔离 | maintained 仅注入本方上一轮信念；context 始终 null；双方私有策略哨兵不跨角色 |
| 公共通道 | 私有块不进入历史；坏标签、嵌套块、缺失/重复/错误角色报价停止该局，没有 raw-output fallback |
| 缺失处理 | 非法区间、置信度、NaN/Infinity、未知字段被记录为缺失，不修复；无效上一轮不复用旧状态 |
| 生命周期 | reset 清空状态；买方先、卖方能看到本轮买方消息；N=1/N=6 时都严格限制轮数 |
| 指标 | 手算 MAE、归一化误差、Brier、IoU、效用、Nash；区分无效成交与不可成交场景 |
| API 接线 | 通过本机 HTTP fixture 验证单次请求、温度、token 上限、seed 和返回元数据；fixture 不是 LLM |
| 可复现流程 | 3 场景 × 2 seeds × 2 模式 = 12 局；112 次 live 假模型调用；另有 2 对/4 次冻结前缀调用 |
| 输入消融 | 同一探针对除 previous_belief 块外 prompt 完全一致，请求 seed 相同；不影响 live 轨迹 |

MockModel 故意忽略上一轮信念，因此两组 mock 输出相同。这是测试夹具的设计，
不能用来判断真实 Agent 是否维护信念。完整 smoke 轨迹位于被 Git 忽略的 runs 目录；
不将假模型性能表作为研究结果提交。

未验证：真实模型的格式遵循率、信念估计水平、状态回传是否造成锚定、协商效果、
远端服务的确定性，以及模型生成的公开消息是否主动披露本方底价。
运行后按 README 的误差/宽度/一致性/协商多指标联合检查，再决定是否值得扩大样本。
