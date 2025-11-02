import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pyaudio
import numpy as np
import wave

import librosa
import librosa.display

CHUNK = 2048  # 增加 CHUNK 大小以减少溢出
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # 录音时长(秒)
OUTPUT_FILENAME = "output.wav"  # 输出文件名

plt.rc("font", family="SimHei")  # 支持中文显示
p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=None,
)


class AudioVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [])
        self.ax.set_xlim(0, CHUNK)
        self.ax.set_ylim(-1, 1)
        self.ax.set_title("实时音频波形")
        self.ax.set_xlabel("样本")
        self.ax.set_ylabel("幅度")

    def update(self, frame):
        try:
            # 添加异常处理，避免溢出错误导致程序崩溃
            audio_data = np.frombuffer(
                stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32
            )
            self.line.set_data(range(len(audio_data)), audio_data)
        except Exception as e:
            print(f"读取音频数据时出错: {e}")
        return (self.line,)

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, interval=50, blit=True)
        plt.show()


if __name__ == "__main__":
    m_AudioVisualizer = AudioVisualizer()
    try:
        m_AudioVisualizer.animate()
    except KeyboardInterrupt:
        print("\n停止可视化...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("资源已清理")
