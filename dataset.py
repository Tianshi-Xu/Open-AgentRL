import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import sys

def series_to_item(ls):
    """从pandas Series/numpy array中提取实际值（完全参考verl的实现）"""
    import numpy
    import pandas
    
    while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
        ls = ls[0]
    return ls

def convert_nested_value_to_list_recursive(data_item):
    """递归转换嵌套值为list（参考verl的实现）"""
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        return data_item

# 加载tokenizer（使用与训练相同的模型路径）
MODEL_PATH = "/home/v-tianshixu/pretrained_model/Qwen3-4B-Instruct-2507"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"✅ 成功加载tokenizer")
except Exception as e:
    print(f"❌ 加载tokenizer失败: {e}")
    sys.exit(1)

# 加载数据（完全按照verl的方式）
try:
    df = pd.read_parquet("dataset/Open-AgentRL-SFT-3K/full_sft_3k_shuffled_v4.parquet")
    print(f"✅ 成功加载数据，总样本数: {len(df)}")
    print(f"列名: {list(df.columns)}")
except Exception as e:
    print(f"❌ 加载数据失败: {e}")
    sys.exit(1)

# 检查必需的列
messages_key = "messages"
if messages_key not in df.columns:
    print(f"❌ 数据中缺少'{messages_key}'列")
    print(f"可用的列: {list(df.columns)}")
    sys.exit(1)

# 按照verl的方式提取messages（关键！）
print("正在提取messages（按照verl的方式）...")
messages_list = df[messages_key].apply(series_to_item).tolist()
# 处理numpy array的情况：转换为list
for i, msg in enumerate(messages_list):
    if isinstance(msg, np.ndarray):
        messages_list[i] = msg.tolist()
print(f"✅ 成功提取 {len(messages_list)} 个样本的messages")

# 提取tools（如果存在）
tools_list = None
tools_key = "tools"
if tools_key in df.columns:
    tools_list = df[tools_key].apply(convert_nested_value_to_list_recursive).tolist()
    print(f"✅ 成功提取tools")
else:
    print(f"⚠️  数据中没有'{tools_key}'列，将使用None")

print("\n正在tokenize...")

# 真实tokenize每个样本
lengths = []
errors = []

for idx in tqdm(range(len(messages_list)), desc="处理样本"):
    try:
        messages = messages_list[idx]
        
        # 最终确保messages是list（处理numpy array等）
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        if not isinstance(messages, list):
            raise ValueError(f"Messages should be a list, got {type(messages)}, value: {messages}")
        
        # 验证messages格式
        if len(messages) == 0:
            raise ValueError("Messages list is empty")
        if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages):
            raise ValueError(f"Invalid message format: {messages[0] if messages else None}")
        
        # 获取tools（如果存在）
        tools = None
        if tools_list is not None:
            tools = tools_list[idx]
        
        # 使用apply_chat_template真实tokenize（与训练时完全一致）
        result = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )
        
        # 处理返回值（可能是tensor或list）
        if isinstance(result, torch.Tensor):
            tokens = result
        elif isinstance(result, list):
            tokens = torch.tensor(result)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected return type from apply_chat_template: {type(result)}")
        
        # 获取真实长度
        if tokens.dim() == 1:
            seq_length = tokens.shape[0]
        elif tokens.dim() == 2:
            seq_length = tokens.shape[1]
        else:
            raise ValueError(f"Unexpected token tensor shape: {tokens.shape}")
        
        lengths.append(seq_length)
        
    except Exception as e:
        errors.append((idx, str(e)))
        if len(errors) <= 5:  # 只打印前5个错误
            print(f"\n❌ 错误样本 {idx}: {e}")
            import traceback
            traceback.print_exc()

# 统计结果
if lengths:
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    print("\n" + "="*50)
    print("序列长度统计（真实token数）:")
    print("="*50)
    print(f"样本总数: {len(lengths)}")
    print(f"平均长度: {lengths.float().mean().item():.0f} tokens")
    print(f"中位数: {lengths.median().item():.0f} tokens")
    print(f"最小值: {lengths.min().item():.0f} tokens")
    print(f"最大值: {lengths.max().item():.0f} tokens")
    print(f"标准差: {lengths.float().std().item():.0f} tokens")
    
    # 统计超过特定长度的样本数量
    print("\n超长样本统计:")
    count_16k = (lengths > 16384).sum().item()
    pct_16k = count_16k / len(lengths) * 100
    print(f"  >16384 tokens: {count_16k} 样本 ({pct_16k:.2f}%)")
    
    count_32k = (lengths > 32768).sum().item()
    pct_32k = count_32k / len(lengths) * 100
    print(f"  >32768 tokens: {count_32k} 样本 ({pct_32k:.2f}%)")
    
    if count_32k > 0:
        max_length_val = lengths.max().item()
        print(f"  最长样本: {max_length_val:.0f} tokens")
    
    print("\n分位数:")
    for p in [50, 75, 90, 95, 99]:
        val = torch.quantile(lengths.float(), p/100).item()
        print(f"  {p}%分位: {val:.0f} tokens")
    
    print("\n长度分布:")
    # 使用手动计算来包含>32K的样本
    bins_list = [0, 1024, 2048, 4096, 8192, 16384, 32768, float('inf')]
    bin_labels = ["<1K", "1K-2K", "2K-4K", "4K-8K", "8K-16K", "16K-32K", ">32K"]
    
    # 手动计算每个区间的数量
    hist_counts = []
    for i in range(len(bins_list) - 1):
        left = bins_list[i]
        right = bins_list[i + 1]
        if right == float('inf'):
            count = ((lengths >= left).sum()).item()
        else:
            count = ((lengths >= left) & (lengths < right)).sum().item()
        hist_counts.append(count)
    
    for label, count in zip(bin_labels, hist_counts):
        pct = count / len(lengths) * 100
        print(f"  {label:>8}: {count:>5} 样本 ({pct:>5.1f}%)")
    
    # 建议的max_length
    p95 = torch.quantile(lengths.float(), 0.95).item()
    p99 = torch.quantile(lengths.float(), 0.99).item()
    print("\n建议的max_length设置:")
    print(f"  覆盖95%样本: {p95:.0f} tokens")
    print(f"  覆盖99%样本: {p99:.0f} tokens")
    print(f"  当前设置: 32768 tokens")
    if p95 > 32768:
        print(f"  ⚠️  警告: 95%分位超过当前max_length，可能丢失数据！")
    elif p95 > 16384:
        print(f"  💡 建议: 可考虑使用 {int(p95*1.1):.0f} tokens")
    else:
        print(f"  ✅ 当前设置充足")
        
    # 显存估算（更准确的估算）
    print("\n显存需求估算（单GPU, 4B模型, batch_size=1）:")
    avg_length = lengths.float().mean().item()
    print(f"  序列长度: {avg_length:.0f} tokens (平均)")
    print(f"  模型参数: 8 GB (bf16)")
    print(f"  梯度: 8 GB (bf16)")
    print(f"  优化器: 32 GB (AdamW fp32)")
    # 激活值估算：hidden_size=2560, num_layers=36, checkpointing后约保存5%的激活
    # 每个checkpoint: batch * seq_len * hidden_size * 2 bytes (bf16)
    # 估算同时有多个checkpoint在内存
    est_activation_per_checkpoint = 1 * avg_length * 2560 * 2 / (1024**3)  # GB per checkpoint
    est_active_checkpoints = 2  # 估算同时有2-3个checkpoint
    est_activation = est_activation_per_checkpoint * est_active_checkpoints * 0.05  # checkpointing节省
    est_activation = max(est_activation, 1.0)  # 至少1GB
    print(f"  激活值: ~{est_activation:.1f} GB (gradient checkpointing后估算)")
    print(f"  总计: ~{48 + est_activation:.1f} GB")

if errors:
    print(f"\n⚠️  处理失败的样本数: {len(errors)}")
    if len(errors) > 5:
        print(f"   (仅显示前5个错误)")

print("\n" + "="*50)