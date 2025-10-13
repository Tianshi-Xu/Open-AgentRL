## Demystifying Reinforcement Learning in Agentic Reasoning

<p align="center">
  <a href="https://arxiv.org/abs/2509.06949">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="Paper on arXiv"
    />
  <a href="https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K">
    <img 
        src="https://img.shields.io/badge/Datasets-Hugging%20Face%20Data-orange?logo=huggingface&logoColor=yellow" 
        alt="Datasets on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/Gen-Verse/DemyAgent-4B">
    <img 
        src="https://img.shields.io/badge/DemyAgent%204B-Hugging%20Face%20Model-FFCC00?logo=huggingface&logoColor=yellow" 
        alt="DemyAgent-4B on Hugging Face"
    />
  </a>
</p>


> [**Demystifying Reinforcement Learning in Agentic Reasoning**](arxiv link)
>  [Zhaochen Yu](https://zhaochenyu0201.github.io/), [Ling Yang](https://yangling0818.github.io/), [Jiaru Zou](https://jiaruzouu.github.io/), [Shuicheng Yan](https://www.comp.nus.edu.sg/cs/people/yansc/), [Mengdi Wang](https://mwang.princeton.edu/), <br>**National University of Singapore, University of Illinois at Urbana-Champaign, Princeton University**<br>

## Introduction

<table class="center">     <tr>     <td width=100% style="border: none"><img src="figs/overview.png" style="width:100%"></td>     </tr>     <tr>     <td width="100%" style="border: none; text-align: center; word-wrap: break-word">An overview of our research on agentic RL. </td>   </tr> </table>

In this work, we systematically investigate three dimensions of agentic RL: **data, algorithms, and reasoning modes**. Our findings reveal: (1) real end-to-end trajectories and high-diversity datasets significantly outperform synthetic alternatives; (2) exploration-friendly techniques like reward clipping and entropy maintenance boost training efficiency; (3) deliberative reasoning with selective tool calls surpasses frequent invocation or verbose self-reasoning. We contribute high-quality SFT and RL datasets, demonstrating that **simple recipes enable even 4B models to outperform 32B models** on challenging benchmarks including AIME2024/2025, GPQA-Diamond, and LiveCodeBench-v6.

## 🚩 New Updates

- **[2025.10]** We fully open-source our work, including:
  - Training code for both SFT and RL stages
  - High-quality SFT dataset (3K samples) and RL dataset (30K samples)
  - Model checkpoints: SFT models (Qwen2.5-7B-RA-SFT, Qwen3-4B-RA-SFT) and RL-trained model (DemyAgent-4B)
  - Evaluation Scripts for our models

## 📦 Dataset

- [🤗 3K Agentic SFT Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K)
- [🤗 30K Agentic RL Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K)

## 🤖 Model Zoo

| **Model**         | **Download**                                                 |
| ----------------- | ------------------------------------------------------------ |
| Qwen2.5-7B-RA-SFT | [🤗 HuggingFace](https://huggingface.co/Gen-Verse/Qwen2.5-7B-RA-SFT) |
| Qwen3-4B-RA-SFT   | [🤗 HuggingFace](https://huggingface.co/Gen-Verse/Qwen3-4B-RA-SFT) |
| DemyAgent-4B      | [🤗 HuggingFace](https://huggingface.co/Gen-Verse/DemyAgent-4B) |

## 🚀 Get Started

```bash
git clone https://github.com/Gen-Verse/Open-AgentRL.git
conda create -n OpenAgentRL python=3.11 
conda activate OpenAgentRL
cd Open-AgentRL
bash scripts/install_vllm_sglang_mcore.sh
pip install -e .[vllm]
```

## 🔧 Training

### Cold-Start SFT

Before you start SFT, make sure you have downloaded the [3K Agentic SFT Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-SFT-3K) and the corresponding base models like [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Configure [qwen3_4b_sft.sh](recipe/demystify/qwen3_4b_sft.sh) and [qwen2_7b_sft.sh](recipe/demystify/qwen2_7b_sft.sh), and set the absolute paths to your model and the `.parquet` data files.

- **TRAIN_DATA**: Path to the `.parquet` file of the SFT dataset
- **EVAL_DATA**: Path to the evaluation data (can be set to the same as **TRAIN_DATA**)
- **MODEL_PATH**: Path to your base models like [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) or [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **SAVE_PATH**: Directory to save the SFT model checkpoints

After all configurations are set, simply run the code below to finetune Qwen3-4B-Instruct-2507:

```bash
bash recipe/demystify/qwen3_4b_sft.sh
```

After obtaining the SFT models, utilize the following command to merge the model:

```bash
python3 -m verl.model_merger merge --backend fsdp --local_dir xxx/global_step_465 --target_dir xxx/global_step_465/huggingface
```

### Agentic RL

After obtaining the SFT models (you can also directly use our provided checkpoints [Qwen2.5-7B-RA-SFT](https://huggingface.co/Gen-Verse/Qwen2.5-7B-RA-SFT) and [Qwen3-4B-RA-SFT](https://huggingface.co/Gen-Verse/Qwen3-4B-RA-SFT)), you can start Agentic RL with our GRPO-TCR recipe.

First, download our [30K Agentic RL Data](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-30K) and the [evaluation datasets](https://huggingface.co/datasets/Gen-Verse/Open-AgentRL-Eval).

Then, configure the [SandboxFusion](https://github.com/bytedance/SandboxFusion) environment for code execution.

There are two ways to create a sandbox:

1. **Local Deployment**: Deploy SandboxFusion locally by referring to [the SandboxFusion deployment documentation](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment)
2. **Cloud Service**: Use Volcano Engine Cloud FaaS service by referring to [Volcano Engine Code Sandbox](https://www.volcengine.com/docs/6662/1539235)

Using either method, obtain an API endpoint (something like `https://<ip-address-or-domain-name>/run_code`), and configure it in **`recipe/demystify/sandbox_fusion_tool_config.yaml`** and **the function check_correctness in`verl/utils/reward_score/livecodebench/code_math.py`**.

Next, configure the Agentic RL scripts [grpo_tcr_qwen2_7b.sh](recipe/demystify/grpo_tcr_qwen2_7b.sh) and [grpo_tcr_qwen3_4b.sh](recipe/demystify/grpo_tcr_qwen3_4b.sh):

- **open_agent_rl**: Path to the `.parquet` file of the agentic RL dataset
- **model_path**: Path to the SFT models
- **aime2024/aime2025**: Benchmark datasets evaluated every 10 training steps. Set the absolute paths to the `.parquet` files of the benchmarks. You can also add more benchmarks like GPQA-Diamond in **test_files**
- **default_local_dir**: Directory to save your RL checkpoints

**Training Resources**: We conducted our training on one $8\times \text{Tesla-A100}$ node with a batch size of 64.

After finishing the configurations, run the code below to conduct Agentic RL with the GRPO-TCR recipe:

```bash
bash recipe/demystify/grpo_tcr_qwen3_4b.sh
```

You can observe the training dynamics and evaluation results in Weights & Biases (wandb).

## 📊 Evaluation

If you have already trained a model, you can refer to the following process for agentic reasoning capability evaluation. Alternatively, you can download our checkpoint from [🤗 DemyAgent-4B](https://huggingface.co/Gen-Verse/DemyAgent-4B) for direct testing.

### AIME2024/2025 and GPQA-Diamond

Configure the scripts [eval_qwen2_7b_aime_gpqa.sh](recipe/demystify/eval/eval_qwen2_7b_aime_gpqa.sh) and [eval_qwen3_4b_aime_gpqa.sh](recipe/demystify/eval/eval_qwen3_4b_aime_gpqa.sh). The configuration process is similar to the training setup—set the paths to your models and `.parquet` files of the benchmarks.

Simply run the code below to evaluate performance on AIME2024/2025 and GPQA-Diamond:

```bash
bash recipe/demystify/eval/eval_qwen3_4b_aime_gpqa.sh
```

You can observe the average@32/pass@32/maj@32 metrics from your wandb project.

### LiveCodeBench-v6

First, run inference for the benchmark:

```bash
bash recipe/demystify/eval/eval_qwen3_4b_livecodebench.sh
```

Specifically, we save the validation rollout paths in **VAL_SAVE_PATH**. After obtaining the validation rollouts, refer to the official evaluation process for local results in [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).

## 📈 Results

We provide the evaluation results of the agentic reasoning abilities of our models on challenging benchmarks including AIME2024/AIME2025, GPQA-Diamond, and LiveCodeBench-v6.

|                            | **MATH**     |              | **Science**      | **Code**             |
| -------------------------- | ------------ | ------------ | ---------------- | -------------------- |
| **Method**                 | **AIME2024** | **AIME2025** | **GPQA-Diamond** | **LiveCodeBench-v6** |
| *Self-Contained Reasoning* |              |              |                  |                      |
| Qwen2.5-7B-Instruct        | 16.7         | 10.0         | 31.3             | 15.2                 |
| Qwen3-4B-Instruct-2507     | 63.3         | 47.4         | 52.0             | **35.1**             |
| Qwen2.5-72B-Instruct       | 18.9         | 15.0         | 49.0             | -                    |
| DeepSeek-V3                | 39.2         | 28.8         | 59.1             | 16.1                 |
| DeepSeek-R1-Distill-32B    | 70.0         | 46.7         | 59.6             | -                    |
| DeepSeek-R1-Zero (671B)    | 71.0         | 53.5         | 59.6             | -                    |
| *Agentic Reasoning*        |              |              |                  |                      |
| Qwen2.5-7B-Instruct        | 4.8          | 5.6          | 25.5             | 12.2                 |
| Qwen3-4B-Instruct-2507     | 17.9         | 16.3         | 44.3             | 23.0                 |
| ToRL-7B                    | 43.3         | 30.0         | -                | -                    |
| ReTool-32B                 | 72.5         | 54.3         | -                | -                    |
| Tool-Star-3B               | 20.0         | 16.7         | -                | -                    |
| ARPO-7B                    | 30.0         | 30.0         | 53.0             | 18.3                 |
| rStar2-Agent-14B           | **80.6**     | <u>69.8</u>  | **60.9**         | -                    |
| **DemyAgent-4B (Ours)**    | <u>72.6</u>  | **70.0**     | <u>58.5</u>      | <u>26.8</u>          |

As demonstrated in the table above, despite having only 4B parameters, **DemyAgent-4B** matches or even outperforms much larger models (14B/32B) across challenging benchmarks. Notably, **DemyAgent-4B achieves state-of-the-art agentic reasoning performance**, surpassing [ReTool-32B](https://arxiv.org/pdf/2504.11536) and [rStar2-Agent-14B](https://arxiv.org/pdf/2508.20722), and even outperforming long-CoT models like DeepSeek-R1-Zero on AIME2025, which further validates the insights of our research.

## 📝 Citation

```bibtex
@article{yu2025demystify,
  author    = {Zhaochen Yu and Ling Yang and Jiaru Zou and Shuicheng Yan and Mengdi Wang},
  title     = {Demystifying Reinforcement Learning in Agentic Reasoning},
  year      = {2025},
  journal   = {arXiv preprint},
}
```

## 🙏 Acknowledgements

This work aims to explore more efficient paradigms for Agentic RL. Our implementation builds upon the excellent codebases of [VeRL](https://github.com/volcengine/verl) and [ReTool](https://github.com/ReTool-RL/ReTool). We sincerely thank these projects for their valuable insights and high-quality implementations, which have greatly facilitated our research.







