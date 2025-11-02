import pyaudio
import numpy as np
import soundfile as sf

# import matplotlib as plt
from matplotlib import pyplot as plt
import random

CHUNK = 1024
FORMAT = pyaudio.paInt16  # 改为 Int16 格式，这是 WAV 文件的标准格式
MAX_VOLUME = 32768  # 16 位音频的最大幅度
VOLUME = 16384  # 设置音量为 50%
CHANNELS = 1
RATE = 44100
OUTPUT_FILENAME = "output.wav"  # 输出文件名
TEST_FILENAME = "test.wav"
INPUT_TXT_TILENAME = "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/self/project1/input.txt"
DATA_FRAME_LEN = 100

p = pyaudio.PyAudio()

in_stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)
out_stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK
)


# 生成 C 音符 (C4 = 261.63 Hz) 的音频数据
def generate_c_note(duration_chunks=10):
    """生成 C 音符的音频数据"""
    frequency = 261.63  # C4 音符频率
    print(f"播放 C 音符 ({frequency} Hz)，持续 {duration_chunks} 个 chunk...")

    for i in range(duration_chunks):
        # 为每个 chunk 生成正弦波
        t = np.linspace(i * CHUNK / RATE, (i + 1) * CHUNK / RATE, CHUNK, endpoint=False)
        sine_wave = np.sin(2 * np.pi * frequency * t)
        # 转换为 16 位整数格式，音量调整为 50%
        audio_data = (sine_wave * VOLUME).astype(np.int16)
        out_stream.write(audio_data.tobytes())

    print("播放完成！")


# generate_c_note()

t = np.arange(440) / RATE
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

t = np.linspace(0, 1, RATE)  # 1 second duration
fc = 10 * 10**3  # carrier frequency 10kHz
carrier = np.sin(2 * np.pi * fc * t)  # about 1 second


audio_data = np.array([])
# input process
with open(INPUT_TXT_TILENAME, "r") as f:
    data = f.read().strip()
    data_len = len(data)
    frame_num = (data_len + DATA_FRAME_LEN - 1) // DATA_FRAME_LEN

    for i in range(frame_num):
        print(f"Generating frame {i+1}/{frame_num}")
        frame_data = data[i * DATA_FRAME_LEN : (i + 1) * DATA_FRAME_LEN + 1]

        # modulation
        frame_wave = np.zeros(len(frame_data) * 44)
        for j in range(len(frame_data)):
            frame_wave[j * 44 : (j + 1) * 44] = carrier[j * 44 : (j + 1) * 44] * (
                int(frame_data[j]) * 2 - 1
            )

        frame_wave = np.concatenate([preamble, frame_wave])

        audio_data = np.concatenate([audio_data, np.zeros(random.randint(0, 100))])
        audio_data = np.concatenate(
            [audio_data, (frame_wave * VOLUME).astype(np.int16)]
        )
        audio_data = np.concatenate([audio_data, np.zeros(random.randint(0, 100))])

# audio_data = (preamble * VOLUME).astype(np.int16)
# out_stream.write(audio_data.tobytes())
with sf.SoundFile(
    TEST_FILENAME, mode="w", samplerate=RATE, channels=CHANNELS, subtype="PCM_16"
) as file:
    print(f"Saving test audio to {TEST_FILENAME}...")
    file.write(audio_data)


def audio_processing_func():
    shouldStop = False

    with sf.SoundFile(
        OUTPUT_FILENAME, mode="w", samplerate=RATE, channels=CHANNELS, subtype="PCM_16"
    ) as file:
        print("开始录音...")
        while not shouldStop:
            try:
                data = in_stream.read(
                    CHUNK, exception_on_overflow=False
                )  # 添加防止溢出
                audio_data = np.frombuffer(data, dtype=np.int16)
                file.write(audio_data)
            except KeyboardInterrupt:
                shouldStop = True
                continue

        print("录音结束")

        # 停止并关闭流
        in_stream.stop_stream()
        in_stream.close()
        p.terminate()

    print(f"录音已保存到 {OUTPUT_FILENAME}")


# audio_processing_func()
