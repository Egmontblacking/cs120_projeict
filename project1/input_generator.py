from random import randint

OUTPUT_PATH = "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/project1/INPUT.txt"

# 生成 100 帧，每帧 100 位
num_frames = 100
bits_per_frame = 100

with open(OUTPUT_PATH, "w") as f:
    for frame_id in range(1, num_frames + 1):
        # 前 8 位是帧序号（ID），转换为二进制
        id_binary = format(frame_id, "08b")  # 8 位二进制，例如 1 -> "00000001"

        # 后 92 位是随机数据
        random_bits = "".join([str(randint(0, 1)) for _ in range(92)])

        # 组合成 100 位
        frame_data = id_binary + random_bits

        # 写入文件
        f.write(frame_data)

        # 打印前几帧的信息用于验证
        if frame_id <= 5:
            print(
                f"Frame {frame_id}: ID={id_binary} ({frame_id}), Total={len(frame_data)} bits"
            )

print(f"\n生成完成！总共 {num_frames} 帧，每帧 {bits_per_frame} 位")
print(f"文件保存到: {OUTPUT_PATH}")
print(f"总位数: {num_frames * bits_per_frame}")

# 验证生成的数据
with open(OUTPUT_PATH, "r") as f:
    data = f.read().strip()
    print(f"\n验证: 文件总长度 = {len(data)} 位")

    # 检查前几帧的 ID
    print("\n前 5 帧的 ID:")
    for i in range(5):
        frame_start = i * bits_per_frame
        frame_id_bits = data[frame_start : frame_start + 8]
        frame_id_decimal = int(frame_id_bits, 2)
        print(f"  Frame {i}: {frame_id_bits} = {frame_id_decimal}")


# from random import randint, choice

# with open(
#     "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/self/project1/input.txt",
#     "w",
# ) as f:
#     for i in range(10000):
#         f.write(randint(0, 1).__str__())

# # with open(
# #     "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/self/project1/input.txt",
# #     "r",
# # ) as f:
# #     data = f.read().strip()
# #     print(len(data))
