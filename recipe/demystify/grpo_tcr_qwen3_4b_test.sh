set -x
export SANDBOX_STATS_LOG_EVERY=80
export VERL_TOOL_PARSER_ENABLE_REPAIR=0
export VERL_TOOL_PARSER_ENABLE_FEEDBACK=0
# ================= data/model/tool =================
open_agent_rl=dataset/Open-AgentRL-30K/Open-AgentRL-30K.parquet
aime_2024=dataset/Open-AgentRL-Eval/aime2024/aime_2024_problems.parquet
aime_2025=dataset/Open-AgentRL-Eval/aime2025/aime_2025_problems.parquet
model_path=pretrained_model/Qwen3-4B-RA-SFT

train_files="['$open_agent_rl']"
test_files="['$aime_2025', '$aime_2024']"

# tool
tool_config_path=recipe/demystify/sandbox_fusion_tool_config.yaml

# wandb
project_name=demystify-agentic-rl
experiment_name=GRPO-TCR-Qwen3-4B
default_local_dir=output/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

# remove KL divergence ✓
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# clip higher ✓
clip_ratio_low=0.2
clip_ratio_high=0.28

# loss agg ✓
loss_agg_mode="token-mean"

# Dymaic Sampleing, we do not utilize dynamic sampling here since it is too expensive for agentic rl x
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

#Overlong Reward Shaping ✓
reward_manager=dapo
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0


max_turns=16
max_prompt_length=2560
max_response_length=20480
actor_lr=4e-6

train_batch_size=16
ppo_mini_batch_size=256
n_resp_per_prompt=16
n_resp_per_prompt_val=2

# ================= perfomance =================
infer_dp=1
infer_tp=1 # vllm
train_sp=1 # train
offload=False

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

# ================= save rollouts =================
ROLLOUT_SAVE_PATH="${default_local_dir}/rollout"
VAL_SAVE_PATH="${default_local_dir}/validation"

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# Create validation save directory
if [ ! -d "$VAL_SAVE_PATH" ]; then
    mkdir -p $VAL_SAVE_PATH
fi

    python3 -m recipe.demystify.custom_main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/demystify/reward.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/demystify/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@ \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=$offload \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.prefill_attention_backend=fa3 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.decode_attention_backend=flashinfer \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    # +actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype=fp8_e5m2 \
    # +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    # actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \