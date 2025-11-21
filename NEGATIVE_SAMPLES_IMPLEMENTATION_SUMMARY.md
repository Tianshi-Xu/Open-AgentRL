# 负样本功能实现总结

## 概述

本实现为 GRPO agentic 强化学习训练添加了保存失败 tool call 轨迹作为负样本的功能。当 tool call 出现错误触发回滚时，系统可以在修复前保存失败轨迹，用于训练模型避免类似错误。

## 核心特性

### 1. 失败轨迹收集
- 当 tool call 失败触发回滚时，在尝试修复之前保存完整的失败轨迹
- 每个负样本包含完整的状态信息：token IDs、mask、log probabilities、错误信息等

### 2. 每组样本数量限制
- 通过 `max_negative_samples_per_group` 参数限制每组最多保存的负样本数
- 防止负样本过多导致训练批次失衡

### 3. 与 GRPO 无缝集成
- 负样本与父样本共享相同的 UID，因此属于同一个组
- GRPO 计算优势时会考虑组内所有样本（包括负样本）
- 负样本的低奖励会自然产生负优势

## 实现细节

### 修改的文件

1. **verl/verl/experimental/agent_loop/tool_agent_loop.py**
   - 在 `RollbackManager` 中添加负样本保存配置
   - 在 `AgentData` 中添加负样本存储字段
   - 实现 `_create_negative_sample()` 方法创建负样本
   - 在 `_handle_rollback()` 中添加负样本保存逻辑
   - 在 `run()` 方法中将负样本添加到输出

2. **verl/verl/workers/config/rollout.py**
   - 添加 `save_negative_samples` 配置参数
   - 添加 `max_negative_samples_per_group` 配置参数

3. **verl/verl/trainer/config/rollout/rollout.yaml**
   - 添加负样本相关配置项

### 数据流程

```
Tool Call Error
    ↓
Rollback Detection
    ↓
Save Failed Trajectory (if enabled and within limit)
    ↓
AgentData.negative_samples[]
    ↓
AgentLoopOutput.extra_fields["negative_samples"]
    ↓
DataProto.non_tensor_batch["negative_samples"]
    ↓
Trainer Processing (expand batch, assign rewards)
    ↓
GRPO Advantage Computation (grouped by UID)
```

### 负样本结构

```python
{
    "prompt_ids": list[int],           # 完整的 prompt token IDs
    "response_ids": list[int],         # response token IDs
    "response_mask": list[int],        # response mask
    "response_logprobs": list[float],  # log probabilities (如果有)
    "error_messages": list[str],       # 触发回滚的错误消息
    "error_types": list[str],          # 错误类型（如 "ImportError"）
    "tool_position": str,              # 错误发生位置（如 "turn_1"）
    "assistant_turns": int,            # assistant turns 数量
    "user_turns": int,                 # user turns 数量
    "tool_calls": list[dict]           # tool call 信息
}
```

## 配置说明

### YAML 配置

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      # 启用 tool call 回滚机制
      enable_tool_rollback: true
      
      # 每个位置最多重试次数
      max_tool_retries: 3
      
      # 保存失败轨迹作为负样本
      save_negative_samples: true
      
      # 每组最多保存的负样本数量
      max_negative_samples_per_group: 1
      
      # 触发回滚的错误模式
      rollback_on_errors:
        - "ImportError"
        - "ModuleNotFoundError"
        - "SyntaxError"
        - "IndentationError"
        - "NameError"
```

### 命令行参数

```bash
python -m recipe.demystify.custom_main_ppo \
    actor_rollout_ref.rollout.multi_turn.enable_tool_rollback=True \
    actor_rollout_ref.rollout.multi_turn.max_tool_retries=3 \
    actor_rollout_ref.rollout.multi_turn.save_negative_samples=True \
    actor_rollout_ref.rollout.multi_turn.max_negative_samples_per_group=1 \
    # ... 其他参数
```

## Trainer 集成

### 步骤 1: 提取负样本

```python
if "negative_samples" in batch.non_tensor_batch:
    negative_samples = batch.non_tensor_batch["negative_samples"]
    # negative_samples 是一个数组，每个元素对应一个原始样本的负样本列表
```

### 步骤 2: 扩展批次

```python
def expand_batch_with_negatives(batch: DataProto, config) -> DataProto:
    """将负样本作为独立训练样本添加到批次中"""
    
    # 1. 提取负样本
    negative_samples_array = batch.non_tensor_batch["negative_samples"]
    
    # 2. 转换为 tensor 格式
    # 3. 分配与父样本相同的 UID（用于 GRPO 分组）
    # 4. 添加到原始批次
    
    return expanded_batch
```

### 步骤 3: 分配负奖励

```python
def assign_negative_rewards(batch: DataProto, negative_reward: float = -0.5):
    """为负样本分配负奖励"""
    
    if "is_negative_sample" in batch.non_tensor_batch:
        is_negative = batch.non_tensor_batch["is_negative_sample"]
        for i, is_neg in enumerate(is_negative):
            if is_neg:
                # 在最后一个有效 token 位置设置负奖励
                response_length = batch.batch["response_mask"][i].sum().item()
                if response_length > 0:
                    batch.batch["token_level_rewards"][i, response_length - 1] = negative_reward
```

### 步骤 4: GRPO 优势计算

GRPO 会自动处理：
- 负样本与父样本在同一组（共享 UID）
- 计算组内平均奖励和标准差
- 负样本的低奖励会产生负优势：`advantage = (reward - group_mean) / group_std`

## 使用示例

### 配置示例

```yaml
algorithm:
  adv_estimator: grpo

data:
  train_batch_size: 64

actor_rollout_ref:
  rollout:
    n: 8  # 每个 prompt 生成 8 个样本
    multi_turn:
      enable_tool_rollback: true
      max_tool_retries: 3
      save_negative_samples: true
      max_negative_samples_per_group: 1  # 每组最多 1 个负样本

# 负样本的奖励值（应该低于正常失败）
trainer:
  negative_sample_reward: -0.5
```

### 训练流程

1. **Rollout 阶段**：
   - 对每个 prompt 生成 n 个样本（如 n=8）
   - 当 tool call 出错时，保存失败轨迹（如果启用且未超限）
   - 回滚并重试，最终得到成功的轨迹

2. **批次处理**：
   - 提取负样本并转换为训练样本
   - 分配相同的 UID 用于分组
   - 添加负奖励标记

3. **GRPO 训练**：
   - 每组可能有 8 个成功样本 + 1 个失败样本 = 9 个样本
   - GRPO 在这 9 个样本上计算优势
   - 失败样本获得负优势，成功样本根据表现获得正/负优势

## 优势

### 1. 从错误中学习
- 模型可以学习避免导致错误的 tool call 模式
- 提供了明确的负反馈信号

### 2. 平衡训练
- 限制每组负样本数量，保持正负样本平衡
- 不会因为过多负样本而导致训练偏向

### 3. 完整上下文
- 每个负样本包含完整的轨迹信息
- 模型可以理解错误发生的上下文

### 4. 无缝集成
- 通过 extra_fields 机制与现有流程集成
- 不需要修改核心 GRPO 算法

### 5. 可配置性
- 可以通过配置启用/禁用
- 可以灵活控制负样本数量

## 注意事项

1. **负样本收集时机**：负样本在回滚机制尝试修复错误**之前**收集，保留了原始的失败状态

2. **成功重试保留**：回滚机制成功修复后的轨迹会保留在主轨迹中

3. **正负样本平衡**：通过 `max_negative_samples_per_group` 限制，避免负样本过多

4. **log probabilities 正确性**：负样本保存了生成时的 log probabilities，确保可以正确计算策略梯度

5. **UID 分组**：负样本与父样本共享 UID，确保在 GRPO 中属于同一组进行对比

## 测试

项目包含完整的单元测试（`test_negative_samples.py`），覆盖：
- RollbackManager 初始化
- AgentData 负样本字段
- 负样本创建
- 数量限制
- 输出集成
- 错误模式匹配
- 功能开关

所有测试通过，验证了实现的正确性。

## 未来扩展

可能的增强方向：

1. **动态负奖励**：根据错误严重程度动态调整负奖励值
2. **错误类型权重**：为不同类型的错误分配不同的重要性
3. **负样本采样**：当负样本过多时进行智能采样
4. **负样本分析**：提供工具分析常见错误模式
5. **多样性采样**：确保负样本覆盖不同类型的错误

## 文件清单

- `verl/verl/experimental/agent_loop/tool_agent_loop.py` - 核心实现
- `verl/verl/workers/config/rollout.py` - 配置数据类
- `verl/verl/trainer/config/rollout/rollout.yaml` - 配置文件
- `test_negative_samples.py` - 单元测试
- `NEGATIVE_SAMPLES_FEATURE.md` - 功能文档（英文）
- `example_negative_samples_integration.py` - 集成示例
- `NEGATIVE_SAMPLES_IMPLEMENTATION_SUMMARY.md` - 本文档
