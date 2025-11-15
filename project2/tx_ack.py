import pyaudio
import numpy as np
import soundfile as sf
import random
from pathlib import Path
from binutil import read_input

CHANNELS = 1
CHUNK = 1024
RATE = 48000
FORMAT = pyaudio.paFloat32
INPUT_FILENAME = str(Path(__file__).parent / "INPUT.bin")
OUTPUT_FILENAME = str(Path(__file__).parent / "wav" / "ack_check.wav")

DATA_FRAME_LEN = 500  # 数据长度（bits）
SAMPLES_PER_BIT = 8   # 每bit样本数
MAX_RETRANSMIT = 5    # 最大发送次数
TIMEOUT = 0.1         # ACK超时
SRC_ADDR = "01"       # 源端地址
DEST_ADDR = "02"      # 目的端地址
TYPE_DATA = "00"      # DATA帧类型
TYPE_ACK  = "01"      # ACK帧类型

def crc8_generate(data_bits):
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

# 物理层前导码
f_p = np.concatenate([np.linspace(10e3 - 8e3, 10e3, 220), np.linspace(10e3, 10e3 - 8e3, 220)])
omega = 2 * np.pi * np.cumsum(f_p) / RATE
preamble = np.sin(omega)
t = np.linspace(0, 1, RATE)
fc = 10000
carrier = np.sin(2 * np.pi * fc * t)

ADDR_LEN = 2
TYPE_LEN = 2
MAC_HEADER_LEN = ADDR_LEN * 2 + TYPE_LEN
FRAME_CRC_LEN = 8
FRAME_TOTAL_BITS = MAC_HEADER_LEN + DATA_FRAME_LEN + FRAME_CRC_LEN

def make_frame(type_bits, src_bits, dest_bits, payload_bits):
    frame_bits = dest_bits + src_bits + type_bits + payload_bits
    crc_bits = crc8_generate([int(b) for b in frame_bits])
    full_bits = frame_bits + "".join(map(str, crc_bits))
    return full_bits

def modulate(bits):
    wave = np.zeros(len(bits) * SAMPLES_PER_BIT)
    for j in range(len(bits)):
        if bits[j] == "1":
            wave[j*SAMPLES_PER_BIT:(j+1)*SAMPLES_PER_BIT] = carrier[j*SAMPLES_PER_BIT:(j+1)*SAMPLES_PER_BIT]
    return np.concatenate([preamble, wave])

def simulate_ack_received():
    # 实际工程应替换为物理层接收ACK判别
    return random.random() > 0.3   # 概率上 70%收到ACK

def send_data_with_ack(data_bits):
    audio_data = np.array([])
    frame_num = (len(data_bits) + DATA_FRAME_LEN - 1) // DATA_FRAME_LEN

    for i in range(frame_num):
        print(f"\n[MAC] Sending Frame {i+1}/{frame_num}")
        if i == frame_num - 1:
            payload = data_bits[i * DATA_FRAME_LEN :]
        else:
            payload = data_bits[i * DATA_FRAME_LEN : (i + 1) * DATA_FRAME_LEN]

        data_frame_bits = make_frame(TYPE_DATA, SRC_ADDR, DEST_ADDR, payload)

        tx_attempt = 0
        acked = False
        while tx_attempt < MAX_RETRANSMIT and not acked:
            print(f"[MAC] Tx attempt {tx_attempt+1}")
            frame_wave = modulate(data_frame_bits)
            audio_data = np.concatenate([audio_data, np.zeros(random.randint(0, 50))])
            audio_data = np.concatenate([audio_data, frame_wave])

            print(f"[MAC] Waiting for ACK (timeout={TIMEOUT}s)...")
            # TODO:此处为演示，理想流程需接收物理/解调结果判别ACK
            if simulate_ack_received():
                print(f"[MAC] ACK received! [{i+1}/{frame_num}]")
                acked = True
            else:
                print(f"[MAC] ACK Timeout, will retransmit.")
            tx_attempt += 1

        if not acked:
            print(f"[MAC] !!! Link Error: Unable to transmit frame after {MAX_RETRANSMIT} attempts.")
            audio_data = np.concatenate([audio_data, np.zeros(RATE//4)])

    return audio_data

if __name__ == "__main__":
    with open(INPUT_FILENAME, "r") as f:
        data_bits = read_input()
        print(f"Loaded input bits: {len(data_bits)}")
        audio_data = send_data_with_ack(data_bits)

    Path(OUTPUT_FILENAME).parent.mkdir(exist_ok=True, parents=True)
    with sf.SoundFile(OUTPUT_FILENAME, mode="w", samplerate=RATE, channels=CHANNELS, subtype="PCM_32") as file:
        print(f"[Output] Saving ACK audio to {OUTPUT_FILENAME}...")
        file.write(audio_data)
    print("[Done] Transmission with ACK protocol complete.")