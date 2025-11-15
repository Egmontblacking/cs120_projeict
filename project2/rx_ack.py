import pyaudio
import numpy as np
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from binutil import OutputWriter

CHANNELS = 1
CHUNK = 1024
RATE = 48000
FORMAT = pyaudio.paFloat32

OUTPUT_FILENAME = str(Path(__file__).parent / "OUTPUT.bin")

DATA_FRAME_LEN = 500      # 数据比特数
SAMPLES_PER_BIT = 8       # 每个比特的样本数量
FRAME_CRC_LEN = 8         # CRC长度
ADDR_LEN = 2              # 地址长度
TYPE_LEN = 2              # 类型长度
MAC_HEADER_LEN = ADDR_LEN * 2 + TYPE_LEN
FRAME_TOTAL_BITS = MAC_HEADER_LEN + DATA_FRAME_LEN + FRAME_CRC_LEN

SRC_ADDR = "02"
DEST_ADDR = "01"
TYPE_ACK = "01"
TYPE_DATA = "00"

def crc8_generate(data_bits):
    polynomial = [1, 1, 0, 1, 0, 0, 1, 1, 1]
    if isinstance(data_bits, str):
        data = [int(b) for b in data_bits]
    elif isinstance(data_bits, np.ndarray):
        data = data_bits.tolist()
    else:
        data = list(data_bits)
    message = data + [0] * 8
    for i in range(len(data)):
        if message[i] == 1:
            for j in range(len(polynomial)):
                message[i + j] ^= polynomial[j]
    crc = message[-8:]
    return crc

p = pyaudio.PyAudio()
in_stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

# 生成前导码
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
RxFIFO = np.array([])
decodeFIFO = np.array([])

print("start receiving with ACK protocol...")

preamble_count = 0
m_OutputWriter = OutputWriter()

def demodulate_decode(decodeFIFO):
    receive_power = decodeFIFO**2
    receive_power_smooth = uniform_filter1d(receive_power, size=SAMPLES_PER_BIT, mode="nearest")
    sampling_points = np.arange(SAMPLES_PER_BIT // 2, len(receive_power_smooth), SAMPLES_PER_BIT)
    samples = receive_power_smooth[sampling_points[:FRAME_TOTAL_BITS]]
    decode_bit = (samples > np.mean(receive_power)).astype(int)
    return decode_bit

def verify_and_parse_mac_frame(bits):
    # 解析并校验MAC层帧结构
    header = bits[:MAC_HEADER_LEN]
    payload = bits[MAC_HEADER_LEN:MAC_HEADER_LEN + DATA_FRAME_LEN]
    crc_field = bits[MAC_HEADER_LEN + DATA_FRAME_LEN:]
    calc_crc = crc8_generate(list(header) + list(payload))
    if np.array_equal(calc_crc, crc_field):
        dest = ''.join(map(str, header[:ADDR_LEN]))
        src = ''.join(map(str, header[ADDR_LEN:ADDR_LEN*2]))
        frame_type = ''.join(map(str, header[ADDR_LEN*2:]))
        return True, frame_type, src, dest, payload
    else:
        return False, None, None, None, None

def send_ack_sim():
    # 演示“发回ACK”，实际场景可结合物理层发送ACK
    print(">>> ACK sent.")

try:
    while True:
        data = in_stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.float32)
        RxFIFO = np.concatenate([RxFIFO, audio_data])

        for i, current_sample in enumerate(audio_data):
            global_index = len(RxFIFO) - len(audio_data) + i
            power = power * (1 - 1 / 64) + current_sample**2 / 64

            if state == 0:
                syncFIFO = np.roll(syncFIFO, -1)
                syncFIFO[-1] = current_sample
                corr = np.sum(syncFIFO * preamble) / 200

                if (corr > power * 2) and (corr > syncPower_localMax) and (corr > 0.05):
                    syncPower_localMax = corr
                    start_index = global_index
                elif (global_index - start_index > 200) and (start_index != 0):
                    preamble_count += 1
                    syncPower_localMax = 0
                    syncFIFO = np.zeros(len(syncFIFO))
                    state = 1
                    decodeFIFO = RxFIFO[start_index + 1 : global_index]

            elif state == 1:
                decodeFIFO = np.append(decodeFIFO, current_sample)
                if len(decodeFIFO) == SAMPLES_PER_BIT * FRAME_TOTAL_BITS:
                    bits = demodulate_decode(decodeFIFO)
                    ok, frame_type, src, dest, payload = verify_and_parse_mac_frame(bits)
                    # 严格只写数据部分
                    if ok and frame_type == TYPE_DATA:
                        send_ack_sim()
                        bits_to_write = payload
                        if len(bits_to_write) == DATA_FRAME_LEN:
                            for bit in bits_to_write:
                                m_OutputWriter.append_bit(str(bit))
                            print(f"已写入 {len(bits_to_write)} 位")
                        else:
                            print(f"警告：写入比特长度异常！实际: {len(bits_to_write)}，期望: {DATA_FRAME_LEN}")
                    # 若是ACK帧可自行拓展逻辑
                    start_index = 0
                    decodeFIFO = np.array([])
                    state = 0

except KeyboardInterrupt:
    print("\nstop receiving with ACK")
finally:
    in_stream.stop_stream()
    in_stream.close()
    p.terminate()
    m_OutputWriter.close()
    print(f"输出文件已关闭: {OUTPUT_FILENAME}")
    print(f"总共检测到前导码次数: {preamble_count}")
    print("cleanup finished")