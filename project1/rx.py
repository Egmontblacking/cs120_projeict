import pyaudio
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

CHUNK = 1024
FORMAT = pyaudio.paFloat32
# MAX_VOLUME = 32768  # 16 位音频的最大幅度
# VOLUME = 16384  # 设置音量为 50%
CHANNELS = 1
RATE = 48000
OUTPUT_TXT_TILENAME = "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/project1/OUTPUT.txt"
p = pyaudio.PyAudio()

in_stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

# 生成 preamble（前导码）
f_p = np.concatenate(
    [np.linspace(10e3 - 8e3, 10e3, 220), np.linspace(10e3, 10e3 - 8e3, 220)]
)
omega = 2 * np.pi * np.cumsum(f_p) / RATE
preamble = np.sin(omega)

state = 0
power = 0
syncFIFO = np.zeros(440)
syncPower_localMax = 0
start_index = 0
start_index_debug = {}
RxFIFO = np.array([])
decodeFIFO = np.array([])

print("start receiving...")

preamble_count = 0

# 打开输出文件，准备写入解码结果
output_file = open(OUTPUT_TXT_TILENAME, "w")
print(f"输出文件已打开: {OUTPUT_TXT_TILENAME}")

try:
    while True:
        # 逐个 CHUNK 读取音频数据
        data = in_stream.read(CHUNK, exception_on_overflow=False)
        # audio_data = np.frombuffer(data, dtype=np.int16)

        # 归一化到 [-1, 1]
        # audio_data_normalized = audio_data.astype(np.float32) / MAX_VOLUME
        audio_data_normalized = np.frombuffer(data, dtype=np.float32)

        # 添加到接收缓冲区
        RxFIFO = np.concatenate([RxFIFO, audio_data_normalized])

        # 处理每个样本
        for i, current_sample in enumerate(audio_data_normalized):
            global_index = len(RxFIFO) - len(audio_data_normalized) + i
            power = power * (1 - 1 / 64) + current_sample**2 / 64

            if state == 0:
                # 更新同步 FIFO
                syncFIFO = np.roll(syncFIFO, -1)
                syncFIFO[-1] = current_sample

                # 计算相关性
                corr = np.sum(syncFIFO * preamble) / 200

                # 检测前导码
                if (corr > power * 2) and (corr > syncPower_localMax) and (corr > 0.05):
                    syncPower_localMax = corr
                    start_index = global_index
                elif (global_index - start_index > 200) and (start_index != 0):
                    # print(f"检测到前导码！起始位置: {start_index}")
                    preamble_count += 1
                    # start_index_debug[start_index] = 1.5

                    # restore states
                    syncPower_localMax = 0
                    syncFIFO = np.zeros(len(syncFIFO))
                    state = 1

                    # 提取数据到解码缓冲区
                    decodeFIFO = RxFIFO[start_index + 1 : global_index]

            elif state == 1:
                # 添加解码逻辑
                decodeFIFO = np.append(decodeFIFO, current_sample)

                if len(decodeFIFO) == 44 * 108:
                    print(f"\n开始解码 {len(decodeFIFO)} 个样本...")

                    # 关键修复：使用连续的载波相位
                    # 从 preamble 结束位置继续计算相位
                    # fc = 10e3  # 载波频率 10kHz

                    # 关键修复：载波应该从 0 开始，与发送端一致
                    # 发送端使用 carrier[0:44], carrier[44:88] ...
                    # 所以接收端也应该从 0 开始计时
                    # t_decode = np.arange(len(decodeFIFO)) / RATE
                    # carrier_local = np.sin(2 * np.pi * fc * t_decode)

                    t = np.arange(0, 1, 1 / RATE)  # a temp time for 1 second
                    fc = 10 * 10**3  # carrier frequency 10kHz
                    carrier_local = np.sin(2 * np.pi * fc * t)  # about 1 second

                    print(f"解调载波: 从 t=0 开始，共 {len(decodeFIFO)} 个样本")

                    # 与载波相乘（解调）
                    demodulated = decodeFIFO * carrier_local[: len(decodeFIFO)]

                    # 平滑滤波（移动平均，窗口大小为 10）
                    from scipy.ndimage import uniform_filter1d

                    decodeFIFO_removecarrier = uniform_filter1d(
                        demodulated, size=10, mode="nearest"
                    )

                    # 可视化解调后的信号（可选）
                    # plt.plot(decodeFIFO_removecarrier[:400])
                    # plt.title("Demodulated Signal after Removing Carrier")
                    # plt.xlabel("Sample")
                    # plt.ylabel("Amplitude")
                    # plt.show()

                    # 计算每个比特的功率
                    decodeFIFO_power_bit = np.zeros(108)  # 108 个比特
                    for j in range(108):
                        # 对每个比特的中间部分（样本 10-30）求和
                        decodeFIFO_power_bit[j] = np.sum(
                            decodeFIFO_removecarrier[10 + j * 44 : 30 + j * 44]
                        )

                    # 可视化比特功率（可选）
                    # plt.plot(decodeFIFO_power_bit)
                    # plt.title("Bit Power")
                    # plt.xlabel("Bit Index")
                    # plt.ylabel("Power")
                    # plt.axhline(y=0, color='r', linestyle='--')
                    # plt.show()

                    print(
                        f"功率统计 - 最小: {decodeFIFO_power_bit.min():.4f}, 最大: {decodeFIFO_power_bit.max():.4f}"
                    )
                    print(
                        f"正值数量: {np.sum(decodeFIFO_power_bit > 0)}, 负值数量: {np.sum(decodeFIFO_power_bit < 0)}"
                    )

                    decodeFIFO_power_bit = (decodeFIFO_power_bit < 0).astype(int)

                    # CRC 校验（暂时注释掉）
                    # crc_check = generate_crc8(decodeFIFO_power_bit[:100])
                    # if not np.array_equal(crc_check[100:], decodeFIFO_power_bit[100:]):
                    #     print('CRC error')
                    # else:

                    # 提取前 8 位作为 ID（二进制转十进制）
                    # tempindex = 0
                    # for k in range(8):
                    #     tempindex = tempindex + decodeFIFO_power_bit[k] * (2 ** (7 - k))

                    # print(f"解码成功, ID: {tempindex}")
                    decoded_bits = "".join(map(str, decodeFIFO_power_bit))
                    print(f"数据: {decoded_bits}")

                    # 实时写入文件
                    bits_to_wt = decoded_bits[:-8]
                    output_file.write(bits_to_wt)
                    output_file.flush()  # 立即刷新到磁盘
                    print(f"已写入 {len(bits_to_wt)} 比特到文件")

                    # 重置状态，继续检测下一个前导码
                    start_index = 0
                    decodeFIFO = np.array([])
                    state = 0

        # 打印 CHUNK 信息
        # print(
        #     f"读取 CHUNK: {len(audio_data)} 样本, 总缓冲区: {len(RxFIFO)} 样本, 状态: {state}"
        # )

except KeyboardInterrupt:
    print("\nstop receiving")
finally:
    # 关闭音频流
    in_stream.stop_stream()
    in_stream.close()
    p.terminate()

    # 关闭输出文件
    output_file.close()
    print(f"输出文件已关闭: {OUTPUT_TXT_TILENAME}")

    print(f"总共检测到前导码次数: {preamble_count}")
    print("cleanup finished")
