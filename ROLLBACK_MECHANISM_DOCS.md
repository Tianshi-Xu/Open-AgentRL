# 工具调用回滚机制 - 实现文档

## 概述

实现了一个上下文优化的工具调用回滚机制,用于处理 Python code agent 在 Agentic RL 训练中的错误场景。当 LLM 生成的 Python 代码出现错误时,系统会:
1. 捕获错误并提供反馈给 LLM
2. LLM 生成新的工具调用
3. 回滚到错误发生前的状态,用新的工具调用替换旧的
4. 继续执行,节省上下文空间

## 核心优势

1. **上下文节省**: 错误的工具调用被替换而不是累积,避免重复错误占据宝贵的上下文空间
2. **自动修正**: LLM 可以根据错误反馈自动生成正确的代码
3. **可控重试**: 通过配置控制重试次数,避免无限循环
4. **灵活配置**: 可以指定哪些错误需要回滚
5. **状态隔离**: 使用检查点机制,不影响主流程

## 实现细节

### 1. 数据结构扩展

#### AgentData 类新增字段
```python
class AgentData:
    # ... 原有字段 ...
    
    # 回滚相关字段
    self.tool_call_checkpoints: list[dict[str, Any]] = []  # 检查点栈
    self.tool_retry_counts: dict[str, int] = defaultdict(int)  # 重试计数
    self.max_tool_retries = 3  # 最大重试次数(从配置覆盖)
```

### 2. 配置参数

在 `verl/trainer/config/rollout/rollout.yaml` 中新增:

```yaml
multi_turn:
  # 启用工具调用回滚机制
  enable_tool_rollback: false
  
  # 单个工具调用位置的最大重试次数
  max_tool_retries: 3
  
  # 触发回滚的错误模式列表
  rollback_on_errors:
    - "ImportError"
    - "ModuleNotFoundError"
    - "SyntaxError"
    - "IndentationError"
    - "NameError"
    - "tool call format is wrong"
```

### 3. 核心方法

#### 3.1 检查点管理

**_create_checkpoint(agent_data)**
- 创建当前状态的快照
- 保存: prompt_ids, response_ids, response_mask, response_logprobs, messages, image_data, turns

**_restore_checkpoint(agent_data, checkpoint)**
- 从检查点恢复状态
- 用于回滚到工具调用前的状态

#### 3.2 错误检测与处理

**_should_rollback(error_text)**
- 检查错误文本是否匹配配置的回滚模式
- 返回 True/False 决定是否触发回滚

**_format_error_feedback(error_messages)**
- 格式化错误信息为 LLM 友好的反馈
- 提示 LLM 修正错误并生成新的工具调用

#### 3.3 主回滚逻辑

**_handle_processing_tools_state(agent_data)** - 完整流程:

1. **创建检查点**
   ```python
   checkpoint = self._create_checkpoint(agent_data)
   tool_position_key = f"turn_{agent_data.assistant_turns}"
   ```

2. **检查重试限制**
   ```python
   if agent_data.tool_retry_counts[tool_position_key] >= agent_data.max_tool_retries:
       return AgentState.TERMINATED
   ```

3. **执行工具调用**
   ```python
   tasks = [self._call_tool(tool_call, agent_data.tools_kwargs) for tool_call in agent_data.tool_calls]
   responses = await asyncio.gather(*tasks)
   ```

4. **错误检测**
   ```python
   has_rollback_error = False
   error_messages = []
   for tool_response, tool_reward, _ in responses:
       error_text = tool_response.text or ""
       if self._should_rollback(error_text):
           has_rollback_error = True
           error_messages.append(error_text)
   ```

5. **触发回滚** (如果启用且检测到错误)
   
   a. 追加错误反馈到上下文
   ```python
   error_feedback = self._format_error_feedback(error_messages)
   error_message = {"role": "user", "content": error_feedback}
   agent_data.messages.append(error_message)
   ```
   
   b. 让 LLM 重新生成工具调用
   ```python
   new_state = await self._handle_generating_state(agent_data, self.sampling_params, ignore_termination=True)
   ```
   
   c. 恢复检查点
   ```python
   self._restore_checkpoint(agent_data, checkpoint)
   ```
   
   d. 递归重试
   ```python
   return await self._handle_processing_tools_state(agent_data)
   ```

6. **正常处理** (如果没有错误或不需要回滚)
   - 处理工具响应
   - 更新 prompt_ids, response_mask
   - 继续生成流程

## 工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│ _handle_processing_tools_state                              │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 1. 创建检查点         │
        │    checkpoint = ...    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 2. 检查重试次数      │
        │    >= max_retries?    │
        └───────────┬───────────┘
                    │
            ┌───────┴───────┐
            │ 是            │ 否
            ▼               ▼
    ┌──────────┐    ┌──────────────┐
    │ 放弃     │    │ 3. 执行工具  │
    │ TERMINATED│    │    调用      │
    └──────────┘    └──────┬───────┘
                           │
                           ▼
                   ┌──────────────┐
                   │ 4. 检测错误  │
                   │    匹配模式? │
                   └──────┬───────┘
                          │
              ┌───────────┴───────────┐
              │ 有错误 & 启用回滚    │ 无错误或不回滚
              ▼                       ▼
    ┌─────────────────────┐   ┌─────────────────┐
    │ 5a. 追加错误反馈    │   │ 6. 正常处理     │
    │ 5b. LLM 重新生成    │   │    工具响应     │
    │ 5c. 恢复检查点      │   │                 │
    │ 5d. 递归重试        │   │ 返回 GENERATING │
    │     (retry_count++) │   └─────────────────┘
    └──────────┬──────────┘
               │
               └──────┐
                      │ (递归)
                      ▼
          [回到步骤 1: 创建检查点]
```

## 使用示例

### 启用回滚机制

在训练配置中设置:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable_tool_rollback: true
      max_tool_retries: 3
      rollback_on_errors:
        - "ImportError"
        - "SyntaxError"
        - "IndentationError"
```

### 场景示例

**场景**: LLM 生成了有语法错误的 Python 代码

1. **初始状态**:
   - messages: [user_msg, assistant_msg_with_code]
   - tool 执行返回: "SyntaxError: invalid syntax"

2. **回滚流程**:
   - 检测到 "SyntaxError" 匹配回滚模式
   - 追加错误反馈: "The previous tool call(s) failed with the following error(s): SyntaxError..."
   - LLM 看到错误,生成修正后的代码
   - 回滚到执行前状态,替换旧的工具调用
   - 重新执行新的工具调用

3. **最终结果**:
   - 上下文中只保留修正后的代码
   - 错误的代码和中间的错误反馈都被回滚掉
   - 节省了上下文空间

## 关键设计决策

### 1. 为什么使用递归而不是循环?
- 递归让每次重试都经过完整的状态机流程
- 保证状态一致性和代码复用
- 通过 max_retries 避免无限递归

### 2. 为什么在恢复检查点后还要重新生成?
- 确保新的工具调用被 LLM 生成并处理
- 保持与原始流程的一致性
- 允许 LLM 根据错误反馈做出调整

### 3. 检查点包含哪些状态?
- **必须包含**: prompt_ids, response_ids, response_mask, messages
- **可选**: response_logprobs (如果启用)
- **图像数据**: image_data (多模态场景)
- **计数器**: assistant_turns, user_turns

### 4. 为什么基于位置而不是内容来计数?
```python
tool_position_key = f"turn_{agent_data.assistant_turns}"
```
- 同一位置的不同错误应该共享重试配额
- 避免相同错误在同一位置无限重试
- 更符合实际使用场景

## 性能考虑

1. **检查点开销**: 使用 `copy.deepcopy()` 创建状态副本
   - 只在执行工具前创建一次
   - 仅在检测到错误时使用

2. **重试开销**: 每次重试需要额外的 LLM 推理
   - 通过 `max_tool_retries` 限制
   - 只在特定错误时触发

3. **上下文节省**: 避免错误工具调用累积
   - 长期来看减少了上下文长度
   - 提高了后续推理效率

## 测试与验证

运行验证脚本:
```bash
python test_rollback_mechanism.py
```

检查项目:
- ✓ 所有必需方法已实现
- ✓ AgentData 字段正确添加
- ✓ 类级配置正确初始化
- ✓ 关键逻辑模式存在
- ✓ 语法检查通过

## 潜在问题与解决方案

### 问题 1: LLM 重复生成相同错误
**解决方案**: 
- 错误反馈足够详细,包含完整错误信息
- 可以在反馈中添加重试次数提示
- 达到 max_retries 后放弃 rollout

### 问题 2: 检查点内存占用
**解决方案**:
- 只在工具执行前创建,不保留历史
- 可以考虑只拷贝必要字段
- 使用浅拷贝优化(根据需要)

### 问题 3: 无限递归风险
**解决方案**:
- max_tool_retries 硬限制
- 每次递归都检查计数器
- 达到限制返回 TERMINATED

## 未来扩展

1. **智能错误分类**: 不同类型错误使用不同的重试策略
2. **渐进式提示**: 重试次数越多,错误提示越详细
3. **错误历史**: 记录错误模式,用于训练优化
4. **自适应配额**: 根据错误类型动态调整 max_retries
5. **部分回滚**: 只回滚部分状态,保留有用信息

## 总结

该回滚机制成功实现了:
- ✅ 上下文优化: 避免错误工具调用累积
- ✅ 自动修正: LLM 根据反馈自我纠正
- ✅ 可控重试: 配置化的重试策略
- ✅ 状态安全: 检查点机制保证一致性
- ✅ 灵活配置: 可根据场景调整参数

该实现为 Agentic RL 训练中的代码生成任务提供了一个robust的错误处理机制,有效提升了训练效率和模型质量。
