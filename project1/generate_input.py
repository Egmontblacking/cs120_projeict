#!/usr/bin/env python3
"""
生成测试用的 input.txt 文件
每帧包含 100 bits，可以生成任意数量的帧
"""
import numpy as np

# 配置
FRAME_DATA_BITS = 100  # 每帧100位（包含8位ID + 92位数据）
NUM_FRAMES = 100  # 生成100帧

# 设置随机种子以保证可重复
np.random.seed(1)

# 生成随机数据
total_bits = NUM_FRAMES * FRAME_DATA_BITS
bits = np.random.randint(0, 2, total_bits, dtype=np.uint8)

# 保存到文件
with open("input.txt", "w") as f:
    f.write("".join(map(str, bits)))

print(f"✓ Generated input.txt")
print(f"  Total bits: {total_bits}")
print(f"  Frames: {NUM_FRAMES}")
print(f"  Bits per frame: {FRAME_DATA_BITS}")
print(f"  First 50 bits: {''.join(map(str, bits[:50]))}")
