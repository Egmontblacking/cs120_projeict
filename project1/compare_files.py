INPUT_PATH = "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/project1/INPUT.txt"
OUTPUT_PATH = "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/project1/OUTPUT.txt"

# 读取文件
with open(INPUT_PATH, "r") as f:
    input_data = f.read().strip()

with open(OUTPUT_PATH, "r") as f:
    output_data = f.read().strip()

print(f"INPUT.txt 长度: {len(input_data)} 位")
print(f"OUTPUT.txt 长度: {len(output_data)} 位")

# 比较前 100 位
compare_length = min(len(input_data), len(output_data))
input_bits = input_data[:compare_length]
output_bits = output_data[:compare_length]

# 计算差异
differences = sum(1 for i in range(compare_length) if input_bits[i] != output_bits[i])
diff_positions = [i for i in range(compare_length) if input_bits[i] != output_bits[i]]
print(f"前 {compare_length} 位中有 {differences} 位不同。")
# print(f"不同的位置: {diff_positions}")
