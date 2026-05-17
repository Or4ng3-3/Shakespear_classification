import pandas as pd


def create_dataset(file_path, label, chunk_size=100):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 按照空格分割成单词
    words = text.split()

    # 每 100 个单词切成一个片段 (Chunk)
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]

    # 去掉最后一个可能特别短的片段
    if len(chunks[-1].split()) < (chunk_size / 2):
        chunks.pop()

    df = pd.DataFrame({"text": chunks, "label": label})
    return df


# 1. 读取你生成的三个 cleaned 文件
# 假设 0: Richard III (早), 1: Macbeth (中), 2: Tempest (晚)
df_early = create_dataset("./rawdata/Richard_III_cleaned.txt", 0)
df_middle = create_dataset("./rawdata/Macbeth_cleaned.txt", 1)
df_late = create_dataset("./rawdata/Tempest_cleaned.txt", 2)

# 2. 合并并打乱数据
df_all = pd.concat([df_early, df_middle, df_late], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"总共生成了 {len(df_all)} 个训练片段！")
# 你可以看看前五行
print(df_all.head())
