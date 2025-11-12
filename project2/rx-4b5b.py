import pyaudio
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from pathlib import Path
from binutil import OutputWriter

CHANNELS = 1
CHUNK = 1024
RATE = 48000
FORMAT = pyaudio.paFloat32
OUTPUT_FILENAME = str(Path(__file__).parent / "OUTPUT.bin")

# 帧参数（需要与 tx.py 保持一致）
DATA_FRAME_LEN = 500  # 每帧比特数
FRAME_TOTAL_BITS = DATA_FRAME_LEN + 8  # 数据 + CRC8
SAMPLES_PER_BIT = 8  # 每个比特的样本数（可调整）


# CRC8 生成器（与发送端相同）
def crc8_generate(data_bits):
    """
    计算 CRC8 校验码

    参数:
        data_bits: NumPy 数组、列表或字符串，包含 0 和 1

    返回:
        8 位 CRC 校验码（列表）
    """
    # CRC 多项式: x^8+x^7+x^5+x^2+x+1
    # 二进制表示: [1, 1, 0, 1, 0, 0, 1, 1, 1]
    polynomial = [1, 1, 0, 1, 0, 0, 1, 1, 1]

    # 将输入转换为列表
    if isinstance(data_bits, str):
        data = [int(b) for b in data_bits]
    elif isinstance(data_bits, np.ndarray):
        data = data_bits.tolist()
    else:
        data = list(data_bits)

    # 添加 8 个零（CRC 的位数）
    message = data + [0] * 8

    # 执行多项式除法
    for i in range(len(data)):
        if message[i] == 1:
            for j in range(len(polynomial)):
                message[i + j] ^= polynomial[j]

    # 返回最后 8 位作为 CRC
    crc = message[-8:]
    return crc


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

m_OutputWriter = OutputWriter()

try:
    while True:
        # 逐个 CHUNK 读取音频数据
        data = in_stream.read(CHUNK, exception_on_overflow=False)
        # audio_data = np.frombuffer(data, dtype=np.int16)

        audio_data = np.frombuffer(data, dtype=np.float32)

        RxFIFO = np.concatenate([RxFIFO, audio_data])

        # 处理每个样本
        for i, current_sample in enumerate(audio_data):
            global_index = len(RxFIFO) - len(audio_data) + i
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

                if len(decodeFIFO) == SAMPLES_PER_BIT * FRAME_TOTAL_BITS:
                    print(f"\n开始解码 {len(decodeFIFO)} 个样本...")

                    t = np.arange(0, 1, 1 / RATE)  # a temp time for 1 second
                    fc = 10 * 10**3  # carrier frequency 10kHz
                    carrier_local = np.sin(2 * np.pi * fc * t)  # about 1 second

                    # PSK demodulation
                    # # 与载波相乘（解调）
                    # demodulated = decodeFIFO * carrier_local[: len(decodeFIFO)]

                    # # 平滑滤波（移动平均，窗口大小为 10）
                    # from scipy.ndimage import uniform_filter1d

                    # decodeFIFO_removecarrier = uniform_filter1d(
                    #     demodulated, size=10, mode="nearest"
                    # )

                    # # 计算每个比特的功率
                    # decodeFIFO_power_bit = np.zeros(108)  # 108 个比特
                    # for j in range(108):
                    #     # 对每个比特的中间部分（样本 10-30）求和
                    #     decodeFIFO_power_bit[j] = np.sum(
                    #         decodeFIFO_removecarrier[10 + j * 44 : 30 + j * 44]
                    #     )

                    # print(
                    #     f"功率统计 - 最小: {decodeFIFO_power_bit.min():.4f}, 最大: {decodeFIFO_power_bit.max():.4f}"
                    # )
                    # print(
                    #     f"正值数量: {np.sum(decodeFIFO_power_bit > 0)}, 负值数量: {np.sum(decodeFIFO_power_bit < 0)}"
                    # )

                    # decodeFIFO_power_bit = (decodeFIFO_power_bit < 0).astype(int)

                    # OOK demodulation
                    # FIXME: Don't know where to stop
                    # 计算功率包络
                    receive_power = decodeFIFO**2
                    # 平滑处理
                    receive_power_smooth = uniform_filter1d(
                        receive_power, size=SAMPLES_PER_BIT, mode="nearest"
                    )

                    # 采样点：每个比特的中间位置
                    sampling_points = np.arange(
                        SAMPLES_PER_BIT // 2, len(receive_power_smooth), SAMPLES_PER_BIT
                    )
                    samples = receive_power_smooth[sampling_points[:FRAME_TOTAL_BITS]]

                    # 判决阈值：功率平均值
                    threshold = np.mean(receive_power)
                    decodeFIFO_power_bit = (samples > threshold).astype(int)

                    # CRC 校验
                    received_data = decodeFIFO_power_bit[:DATA_FRAME_LEN]  # 数据部分
                    received_crc = decodeFIFO_power_bit[
                        DATA_FRAME_LEN:FRAME_TOTAL_BITS
                    ]  # CRC 部分

                    # 计算 CRC 校验码
                    calculated_crc = crc8_generate(received_data)

                    # 比较 CRC
                    if np.array_equal(calculated_crc, received_crc):
                        print("✓ CRC 校验通过")
                        crc_status = "PASS"
                    else:
                        print("✗ CRC 校验失败")
                        print(f"  接收 CRC: {''.join(map(str, received_crc))}")
                        print(f"  计算 CRC: {''.join(map(str, calculated_crc))}")
                        crc_status = "FAIL"

                    # 提取前 8 位作为 ID（二进制转十进制）
                    frame_id = 0
                    for k in range(8):
                        frame_id = frame_id + received_data[k] * (2 ** (7 - k))

                    print(f"解码成功, 帧 ID: {frame_id}, CRC: {crc_status}")
                    decoded_bits = "".join(map(str, decodeFIFO_power_bit))
                    print(f"数据: {decoded_bits}")

                    # 实时写入文件（只写入数据部分，不包含 CRC）
                    bits_to_wt = decoded_bits[:DATA_FRAME_LEN]  # 只写数据部分
                    for bit in bits_to_wt:
                        m_OutputWriter.append_bit(bit)
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

    m_OutputWriter.close()
    print(f"输出文件已关闭: {OUTPUT_FILENAME}")

    print(f"总共检测到前导码次数: {preamble_count}")
    print("cleanup finished")
