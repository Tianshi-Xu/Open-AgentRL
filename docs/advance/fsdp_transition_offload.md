# FSDP 过渡阶段卸载策略说明

本文档面向所有基于 VERL 的 FSDP 训练/推理混合工作流，介绍 `offload_at_transition_only` 新增配置项的设计背景、实现细节与使用方法。该改动已经合入 `verl/verl/workers/fsdp_workers.py` 与 `verl/verl/workers/config/engine.py`，对 Actor 端的 FSDP 行为具有关键影响，请相关同学务必阅读。

## 1. 背景问题

在混合引擎（Hybrid Engine）模式下，训练阶段需要模型权重、梯度与优化器状态常驻 GPU，以保证 PPO 等算法的吞吐；而推理/rollout 阶段又需要释放显存，以便 SGLang 或其他生成引擎加载同一模型。此前仅能通过：

- 将 `param_offload` / `optimizer_offload` 设为 `True`，在 **每个训练 step** 后自动卸载，导致 GPU 利用率显著下降；
- 将二者设为 `False`，让模型全程留在 GPU，训练虽快，但切换到 rollout 时无法释放显存，触发 `resume_memory_occupation` 连续失败。

为了在二者之间取得平衡，我们引入了“仅在模式切换时卸载”的策略。

## 2. 新增配置项

在 `FSDPEngineConfig` 中新增字段：

```yaml
actor:
  fsdp_config:
    offload_at_transition_only: true
```

- **默认值**：`False`，保持原逻辑。
- **互斥关系**：如果同时启用了 `param_offload` 或 `optimizer_offload`，则该选项会被忽略，并在日志中给出警告；原因是原有布尔开关会在每个 step 自动卸载。

## 3. 核心行为变更

引擎内部新增了两个状态位：

- `_actor_model_offloaded`
- `_actor_optimizer_offloaded`

通过这两个标记判断模型或优化器当前是否在 CPU，从而实现惰性加载/卸载：

1. **训练阶段 (`update_actor`)**
   - 若检测到模型/优化器在 CPU（包括因为过渡策略被显式卸载的情况），则在训练前加载回 GPU；
   - 训练结束后仅在 `param_offload=True` 或 `optimizer_offload=True` 时继续执行逐步卸载。

2. **切换到 rollout (`rollout_mode`)**
   - 必要时先把模型拉回 GPU 汇总权重后，再根据策略决定是否卸载；
   - 当 `offload_at_transition_only=True` 时，会在权重同步后手动调用 `offload_fsdp_model_to_cpu`，并同步卸载优化器状态，确保推理阶段有充足显存。

3. **回到训练 (`trainer_mode`)**
   - 若检测到模型/优化器被过渡策略卸载，则在恢复训练模式前重新加载。

4. **Checkpoint 保存/加载 (`save_checkpoint`, `load_checkpoint`)**
   - 在持久化前临时加载所需权重与优化器状态；
   - 完成后根据进入函数前的状态决定是否再次卸载，既保证功能正确，也避免不必要的数据迁移。

5. **计算 LogProb (`compute_log_prob`)**
   - 调用前确保模型处于 GPU；
   - 完成后仅在原始 `param_offload=True` 的情况下执行卸载，保持历史行为兼容。

## 4. 配置与使用示例

`shell` 脚本中可以直接传参：

```bash
python3 -m recipe.demystify.custom_main_ppo \
  ... \
  actor_rollout_ref.actor.fsdp_config.offload_policy=$offload \
  actor_rollout_ref.actor.fsdp_config.offload_at_transition_only=True
```

当 `offload=False`、`param_offload=False`、`optimizer_offload=False` 时：

- 训练阶段：模型与优化器始终留在 GPU；
- 进入 rollout：在同步权重后统一卸载到 CPU；
- 回到训练：再次加载回 GPU。

## 5. 性能与注意事项

- 该策略能显著提升训练阶段的 GPU 利用率，并避免 rollout 侧的内存不足报错。
- 切换阶段仍存在一次性卸载/加载的成本，建议在真机上通过一两个 epoch 的小规模实验评估总耗时与显存占用。
- 若后续希望支持“仅卸载模型，不卸载优化器”或更细粒度的策略，可以在现有状态位与分支逻辑基础上扩展。

## 6. 建议的验证步骤

1. `offload=False`、`offload_at_transition_only=True` 运行 PPO + rollout：
   - 观察训练循环中 GPU 显存是否稳定；
   - 关注 rollout 阶段显存是否充分释放，`resume_memory_occupation` 是否消失。
2. 切换为 `param_offload=True` 验证兼容性：
   - 日志中应看到警告提示过渡策略被忽略；
   - 行为与旧版本保持一致。

## 7. 迁移指南

- 已有使用 `offload` 布尔开关的脚本无需改动；
- 希望启用新策略的脚本只需在 Actor 的 FSDP 配置中增加一行；
- 自定义扩展或二次开发时，若在其他模块引用 `_actor_model_offloaded` / `_actor_optimizer_offloaded`，请确保遵循“加载后及时复位、卸载后同步打标”的约定。

如对实现细节仍有疑问，建议直接阅读 `verl/verl/workers/fsdp_workers.py` 中与该选项相关的代码段，并结合自身业务场景进行针对性测试。
