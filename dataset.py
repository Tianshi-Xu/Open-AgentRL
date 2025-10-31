from datasets import load_dataset
import wandb
# 加载数据集
dataset = load_dataset("Gen-Verse/Open-AgentRL-SFT-3K") # 替换为具体数据集名称
dataset.save_to_disk("dataset/Open-AgentRL-SFT-3K")

wandb.login(key=)