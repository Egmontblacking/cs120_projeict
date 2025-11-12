import random
import os
from pathlib import Path

key = 42  # 示例 key，范围在 0-99 之间
count_bytes = 6250  # 生成文件的字节数
random.seed(key)
data = random.randbytes(count_bytes)


def read_input(INPUT=str(Path(__file__).parent / "INPUT.bin")):
    with open(INPUT, "rb") as f:
        data = f.read()
        print(f"num of bytes: {len(data)}")
        # print(f"前 20 字节: {data[:20]}")
        # print(data[:20].hex())
        binary_str = "".join(format(byte, "08b") for byte in data)
        # print(f"前 160 位: {binary_str[:160]}")
        print(f"num of bits: {len(binary_str)}")
        # print(type(binary_str))
    return binary_str


class OutputWriter:
    def __init__(self, OUTPUT=str(Path(__file__).parent / "OUTPUT.bin")):
        self.OUTPUT = OUTPUT
        self.f = open(self.OUTPUT, "wb")
        self.buffer = ""
        self.count = 0

    def append_bit(self, bit: str):
        """添加单个比特，累积到8个比特后写入一个字节"""
        self.buffer += bit
        self.count += 1

        if self.count == 8:
            byte_value = int(self.buffer, 2)
            self.f.write(bytes([byte_value]))
            self.buffer = ""
            self.count = 0

    def flush(self):
        """将剩余的比特写入（如果不足8位，补0）"""
        if self.count > 0:
            self.buffer += "0" * (8 - self.count)
            byte_value = int(self.buffer, 2)
            self.f.write(bytes([byte_value]))
            self.buffer = ""
            self.count = 0

    def close(self):
        """关闭文件"""
        self.flush()
        self.f.close()


if __name__ == "__main__":
    writer = OutputWriter()
    binary_data = read_input()
    for bit in binary_data:
        writer.append_bit(bit)
    writer.close()
