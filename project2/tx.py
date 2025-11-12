import pyaudio
import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
import random
from pathlib import Path
from binutil import read_input

CHANNELS = 1
CHUNK = 1024
RATE = 48000
FORMAT = pyaudio.paFloat32
INPUT_FILENAME = str(Path(__file__).parent / "INPUT.bin")
OUTPUT_FILENAME = str(Path(__file__).parent / "wav" / "CHECK.wav")
DATA_FRAME_LEN = 100

p = pyaudio.PyAudio()

out_stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK
)

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


def crc8_generate(data_bits):
    """
    计算 CRC8 校验码

    参数:
        data_bits: 字符串或列表，包含 0 和 1

    返回:
        8 位 CRC 校验码（列表）
    """

    polynomial = [1, 1, 0, 1, 0, 0, 1, 1, 1]

    if isinstance(data_bits, str):
        data = [int(b) for b in data_bits]
    else:
        data = list(data_bits)

    message = data + [0] * 8

    for i in range(len(data)):
        if message[i] == 1:
            for j in range(len(polynomial)):
                message[i + j] ^= polynomial[j]

    crc = message[-8:]
    return crc


audio_data = np.array([])
# input process
with open(INPUT_FILENAME, "r") as f:
    # data = f.read().strip()
    data = read_input()
    data_len = len(data)
    frame_num = (data_len + DATA_FRAME_LEN - 1) // DATA_FRAME_LEN
    print(f"data_len: {data_len}")
    print(f"frame_num: {frame_num}")

    # TODO:
    # frame_num = 1  # for test
    # data = data[:DATA_FRAME_LEN]  # for test

    for i in range(frame_num):
        print(f"Generating frame {i+1}/{frame_num}")
        if i == frame_num - 1:
            frame_data = data[i * DATA_FRAME_LEN :]
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

        # FIXME: temporary disable CRC
        # frame_data_with_crc = frame_data

        # modulation (现在调制 108 个比特: 100 数据 + 8 CRC)
        frame_wave = np.zeros(len(frame_data_with_crc) * 44)
        for j in range(len(frame_data_with_crc)):

            # OOK modulation
            if frame_data_with_crc[j] == "1":
                frame_wave[j * 44 : (j + 1) * 44] = carrier[j * 44 : (j + 1) * 44]
            else:
                frame_wave[j * 44 : (j + 1) * 44] = 0

            # PSK modulation
            # frame_wave[j * 44 : (j + 1) * 44] = carrier[j * 44 : (j + 1) * 44] * (
            #     int(frame_data_with_crc[j]) * 2 - 1
            # )

        frame_wave = np.concatenate([preamble, frame_wave])

        audio_data = np.concatenate([audio_data, np.zeros(random.randint(0, 50))])
        audio_data = np.concatenate([audio_data, frame_wave])
        # audio_data = np.concatenate([audio_data, np.zeros(random.randint(0, 1000))])

# audio_data = (preamble * VOLUME).astype(np.int16)
# audio_data = preamble
# out_stream.write(audio_data.tobytes())

# write audio into file
# TODO: write into output stream
with sf.SoundFile(
    OUTPUT_FILENAME, mode="w", samplerate=RATE, channels=CHANNELS, subtype="PCM_32"
) as file:
    print(f"Saving test audio to {OUTPUT_FILENAME}...")
    file.write(audio_data)
