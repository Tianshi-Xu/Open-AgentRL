# 工具调用回滚机制 - 快速使用指南

## 功能简介

当 LLM 生成的工具调用(如 Python 代码)出现错误时,自动回滚并重试,避免错误工具调用占据上下文空间。

## 启用方法

在你的训练配置文件中设置:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      # 启用回滚机制
      enable_tool_rollback: true
      
      # 每个位置最多重试 3 次
      max_tool_retries: 3
      
      # 这些错误会触发回滚
      rollback_on_errors:
        - "ImportError"
        - "ModuleNotFoundError" 
        - "SyntaxError"
        - "IndentationError"
        - "NameError"
        - "tool call format is wrong"
```

## 工作原理

```
正常流程(无错误):
用户问题 → LLM生成代码 → 执行成功 → 继续

回滚流程(有错误):
用户问题 → LLM生成错误代码 → 执行失败 
         ↓
         检测到错误(如 SyntaxError)
         ↓
         回滚到执行前状态
         ↓
         追加错误反馈 → LLM重新生成 → 替换旧代码
         ↓
         执行新代码 → 成功则继续,失败则继续重试(最多3次)
```

## 核心优势

1. **节省上下文**: 错误代码不会累积,被新代码替换
2. **自动修正**: LLM 看到错误反馈,自动生成正确代码
3. **可控重试**: 最多重试 3 次,避免无限循环
4. **灵活配置**: 可自定义哪些错误需要回滚

## 配置参数说明

- `enable_tool_rollback`: 是否启用回滚机制(默认 false)
- `max_tool_retries`: 单个位置最大重试次数(默认 3)
- `rollback_on_errors`: 触发回滚的错误模式列表(支持部分匹配)

## 示例场景

### 场景 1: Import 错误

```python
# LLM 第1次生成(错误)
import nonexistent_module  # ImportError

# 系统检测到 ImportError,回滚并追加反馈
# "Error: ModuleNotFoundError: No module named 'nonexistent_module'"

# LLM 第2次生成(修正)
import os  # 正确的导入

# 上下文中只保留修正后的代码,节省空间
```

### 场景 2: 语法错误

```python
# LLM 第1次生成(错误)
def calculate(x)
    return x * 2  # SyntaxError: invalid syntax

# 系统检测到 SyntaxError,回滚

# LLM 第2次生成(修正)  
def calculate(x):
    return x * 2

# 错误代码被替换,不占用上下文
```

## 自定义错误模式

如果你的场景需要捕获其他错误,可以扩展配置:

```yaml
rollback_on_errors:
  - "ImportError"
  - "SyntaxError"
  - "TypeError"           # 新增
  - "ValueError"          # 新增
  - "custom error message" # 自定义错误消息
```

## 性能影响

- **正常情况**: 无性能影响(不触发回滚)
- **错误情况**: 每次重试需要额外的 LLM 推理
- **上下文优化**: 长期来看减少上下文长度,提高效率

## 调试建议

1. **查看日志**: 启用 DEBUG 级别查看回滚详情
   ```python
   logger.setLevel("DEBUG")
   ```

2. **调整重试次数**: 根据任务复杂度调整 max_tool_retries

3. **优化错误反馈**: 如果 LLM 持续失败,考虑修改错误消息格式

## 注意事项

1. 回滚机制默认**关闭**,需要显式启用
2. 达到 max_retries 后会放弃本次 rollout
3. 只有匹配 rollback_on_errors 的错误才会触发回滚
4. 其他错误会按原流程处理(不回滚)

## 验证安装

运行测试脚本验证实现:
```bash
python test_rollback_mechanism.py
```

应该看到:
```
✓ All checks passed! Implementation is complete.
```

## 更多信息

详细技术文档: `ROLLBACK_MECHANISM_DOCS.md`
