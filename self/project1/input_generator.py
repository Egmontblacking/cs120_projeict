from random import randint, choice

with open(
    "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/self/project1/input.txt",
    "w",
) as f:
    for i in range(10000):
        f.write(randint(0, 1).__str__())

# with open(
#     "/Users/fanyuxin/Library/Mobile Documents/com~apple~CloudDocs/ShanghiTech/2025_fall/计算机网络/cs120_projeict/self/project1/input.txt",
#     "r",
# ) as f:
#     data = f.read().strip()
#     print(len(data))
