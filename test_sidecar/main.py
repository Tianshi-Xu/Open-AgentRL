# 文件名: occupy_gpus_active.py

import torch
import time
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GPU Active Placeholder Script")
    parser.add_argument(
        "--size", 
        type=int, 
        default=512, 
        help="矩阵乘法的大小 (N x N)。"
             "调小这个值以降低功耗/负载，调大以增加负载。"
             "通常 256 或 512 足够触发利用率监控。"
    )
    return parser.parse_args()

def main():
    args = get_args()
    
    if not torch.cuda.is_available():
        print("错误: CUDA (GPU) 不可用。退出。")
        return

    try:
        # 1. 从 torchrun 获取GPU编号
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        num_gpus = torch.cuda.device_count()

        if local_rank >= num_gpus:
            print(f"错误: 进程Rank {local_rank} 超出了可用的GPU数量 ({num_gpus}).")
            return
            
        # 2. 绑定到指定GPU
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        gpu_name = torch.cuda.get_device_name(local_rank)
        
        matrix_size = args.size
        
        print(f"[进程 Rank {local_rank}] 准备就绪, 目标 GPU: {local_rank} ({gpu_name})")
        print(f"[进程 Rank {local_rank}] 模式: 持续计算 (矩阵大小: {matrix_size}x{matrix_size})")

        # 3. 创建用于计算的张量
        # 我们创建两个张量，然后循环计算它们的乘积
        try:
            A = torch.randn(matrix_size, matrix_size, device=device)
            B = torch.randn(matrix_size, matrix_size, device=device)
            print(f"[进程 Rank {local_rank}] 已在 GPU {local_rank} 上创建张量。")
        
        except Exception as e:
            print(f"[进程 Rank {local_rank}] 创建张量失败 (可能是显存不足?): {e}")
            return

        # 4. 核心：无限循环计算
        print(f"[进程 Rank {local_rank}] ...开始持续的轻量级计算... (按 Ctrl+C 终止)")
        while True:
            # 执行一个轻量级的计算操作 (矩阵乘法)
            # 这将确保 GPU-Util > 0%
            C = A @ B
            
            # (可选) 如果你发现这个循环占用了100%的CPU，
            # 可以在这里加一个极短的CPU睡眠，
            # 但通常没必要，因为 torchrun 会处理。
            # time.sleep(0.001) 

    except KeyboardInterrupt:
        print(f"\n[进程 Rank {local_rank}] 收到终止信号，释放 GPU {local_rank}。")
    except Exception as e:
        print(f"\n[进程 Rank {local_rank}] 发生意外错误: {e}")

if __name__ == "__main__":
    main()