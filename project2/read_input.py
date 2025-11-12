import random
import os

key = 42  # 示例 key，范围在 0-99 之间
count_bytes = 6250  # 生成文件的字节数
random.seed(key)
data = random.randbytes(count_bytes)

print(f"文件大小: {len(data)} 字节")
print(data[:20].hex())


INPUT = "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/project2/INPUT.bin"
with open(INPUT, "rb") as f:
    data = f.read()
    print(f"文件大小: {len(data)} 字节")
    print(f"前 20 字节: {data[:20]}")
    print(data[:20].hex())
