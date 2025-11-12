import pyaudio
import numpy as np
import soundfile as sf

# import matplotlib as plt
from matplotlib import pyplot as plt
import random

CHUNK = 1024
FORMAT = pyaudio.paFloat32
# MAX_VOLUME = 32768  # 16 位音频的最大幅度
# # VOLUME = 16384  # 设置音量为 50%
# VOLUME = MAX_VOLUME
CHANNELS = 1
RATE = 48000
# OUTPUT_FILENAME = "output.wav"  # 输出文件名
TEST_FILENAME = "wav/CHECK.wav"
INPUT_TXT_TILENAME = "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/project1/INPUT.txt"
DATA_FRAME_LEN = 100

p = pyaudio.PyAudio()

in_stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)
out_stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK
)


# 生成 C 音符 (C4 = 261.63 Hz) 的音频数据
# def generate_c_note(duration_chunks=10):
#     """生成 C 音符的音频数据"""
#     frequency = 261.63  # C4 音符频率
#     print(f"播放 C 音符 ({frequency} Hz)，持续 {duration_chunks} 个 chunk...")

#     for i in range(duration_chunks):
#         # 为每个 chunk 生成正弦波
#         t = np.linspace(i * CHUNK / RATE, (i + 1) * CHUNK / RATE, CHUNK, endpoint=False)
#         sine_wave = np.sin(2 * np.pi * frequency * t)
#         # 转换为 16 位整数格式，音量调整为 50%
#         audio_data = (sine_wave * VOLUME).astype(np.int16)
#         out_stream.write(audio_data.tobytes())

#     print("播放完成！")

# generate_c_note()

# t = np.arange(440) / RATE
f_p = np.concatenate(
    [np.linspace(10e3 - 8e3, 10e3, 220), np.linspace(10e3, 10e3 - 8e3, 220)]
)
omega = 2 * np.pi * np.cumsum(f_p) / RATE
preamble = np.sin(omega)
# plt.plot(f_p)
plt.plot(preamble)
plt.title("Preamble Waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
# plt.show()

t = np.linspace(0, 1, RATE)  # 10 seconds duration
fc = 10 * 10**3  # carrier frequency 10kHz
carrier = np.sin(2 * np.pi * fc * t)  # about 10 seconds


# CRC8 生成器
# 多项式: x^8+x^7+x^5+x^2+x+1 = [1,1,0,1,0,0,1,1,1]
def crc8_generate(data_bits):
    """
    计算 CRC8 校验码

    参数:
        data_bits: 字符串或列表，包含 0 和 1

    返回:
        8 位 CRC 校验码（列表）
    """
    # CRC 多项式: x^8+x^7+x^5+x^2+x+1
    # 二进制表示: [1, 1, 0, 1, 0, 0, 1, 1, 1]
    # 对应位: x^8, x^7, x^6, x^5, x^4, x^3, x^2, x^1, x^0
    polynomial = [1, 1, 0, 1, 0, 0, 1, 1, 1]

    # 将输入转换为列表
    if isinstance(data_bits, str):
        data = [int(b) for b in data_bits]
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


audio_data = np.array([])
# input process
with open(INPUT_TXT_TILENAME, "r") as f:
    data = f.read().strip()
    data_len = len(data)
    # print(data_len)
    frame_num = (data_len + DATA_FRAME_LEN - 1) // DATA_FRAME_LEN
    # print(frame_num)

    # TODO:
    # frame_num = 1  # for test
    # data = data[:DATA_FRAME_LEN]  # for test

    for i in range(frame_num):
        print(f"Generating frame {i+1}/{frame_num}")
        if i == frame_num - 1:
            frame_data = data[i * DATA_FRAME_LEN :]
            # print(frame_data)
            # TODO: pack data len in header
        else:
            frame_data = data[i * DATA_FRAME_LEN : (i + 1) * DATA_FRAME_LEN]

        # print(len(frame_data))

        # 计算 CRC8 校验码
        crc_bits = crc8_generate(frame_data)
        print(f"Frame {i+1} CRC8: {''.join(map(str, crc_bits))}")

        # 将 CRC 添加到数据后面
        frame_data_with_crc = frame_data + "".join(map(str, crc_bits))
        print(f"Frame {i+1} Data with CRC: {frame_data_with_crc}")
        print(f"Data length with CRC: {len(frame_data_with_crc)} bits")

        # modulation (现在调制 108 个比特: 100 数据 + 8 CRC)
        frame_wave = np.zeros(len(frame_data_with_crc) * 44)
        for j in range(len(frame_data_with_crc)):
            frame_wave[j * 44 : (j + 1) * 44] = carrier[j * 44 : (j + 1) * 44] * (
                int(frame_data_with_crc[j]) * 2 - 1
            )

        frame_wave = np.concatenate([preamble, frame_wave])

        # CRC 已添加

        audio_data = np.concatenate([audio_data, np.zeros(random.randint(0, 1000))])
        audio_data = np.concatenate([audio_data, frame_wave])
        audio_data = np.concatenate([audio_data, np.zeros(random.randint(0, 1000))])

# audio_data = (preamble * VOLUME).astype(np.int16)
# audio_data = preamble
# out_stream.write(audio_data.tobytes())

# write audio into file
with sf.SoundFile(
    TEST_FILENAME, mode="w", samplerate=RATE, channels=CHANNELS, subtype="PCM_32"
) as file:
    print(f"Saving test audio to {TEST_FILENAME}...")
    file.write(audio_data)


# def audio_processing_func():
#     shouldStop = False

#     with sf.SoundFile(
#         OUTPUT_FILENAME, mode="w", samplerate=RATE, channels=CHANNELS, subtype="PCM_16"
#     ) as file:
#         print("开始录音...")
#         while not shouldStop:
#             try:
#                 data = in_stream.read(
#                     CHUNK, exception_on_overflow=False
#                 )  # 添加防止溢出
#                 audio_data = np.frombuffer(data, dtype=np.int16)
#                 file.write(audio_data)
#             except KeyboardInterrupt:
#                 shouldStop = True
#                 continue

#         print("录音结束")

#         # 停止并关闭流
#         in_stream.stop_stream()
#         in_stream.close()
#         p.terminate()

#     print(f"录音已保存到 {OUTPUT_FILENAME}")


# audio_processing_func()
