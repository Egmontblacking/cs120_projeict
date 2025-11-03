import pyaudio
import numpy as np
import soundfile as sf

CHUNK = 1024
FORMAT = pyaudio.paInt16  # 改为 Int16 格式，这是 WAV 文件的标准格式
MAX_VOLUME = 32768  # 16 位音频的最大幅度
VOLUME = 16384  # 设置音量为 50%
CHANNELS = 1
RATE = 44100
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
try:
    while True:
        # 逐个 CHUNK 读取音频数据
        data = in_stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # 归一化到 [-1, 1]
        audio_data_normalized = audio_data.astype(np.float32) / MAX_VOLUME

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
                    print(preamble_count)
                    start_index_debug[start_index] = 1.5
                    syncPower_localMax = 0
                    syncFIFO = np.zeros(len(syncFIFO))
                    state = 1

                    # 提取数据到解码缓冲区
                    decodeFIFO = RxFIFO[start_index + 1 : global_index]
                    # print(f"提取了 {len(decodeFIFO)} 个样本用于解码")

                    # 重置状态继续检测
                    state = 0
                    start_index = 0
            elif state == 1:
                pass  # TODO: 添加解码逻辑

        # 打印 CHUNK 信息
        # print(
        # f"读取 CHUNK: {len(audio_data)} 样本, 总缓冲区: {len(RxFIFO)} 样本, 状态: {state}"
        # )

except KeyboardInterrupt:
    print("\nstop receiving")
finally:
    in_stream.stop_stream()
    in_stream.close()
    p.terminate()
    print(f"总共检测到前导码次数: {preamble_count}")
    print("cleanup finished")
