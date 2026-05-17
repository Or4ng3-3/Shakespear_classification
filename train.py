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

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
import torch

# 1. 划分训练集和测试集 (80%训练, 20%测试)
train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 2. 加载 MacBERTh 的 Tokenizer
model_name = "emanjavacas/MacBERTh"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 3. 加载 MacBERTh 模型 (设置分为 3 类)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 4. 设置训练参数 (使用 GPU 训练)
training_args = TrainingArguments(
    output_dir="./shakespeare_results",
    eval_strategy="epoch",  # 每个 epoch 评估一次
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # 跑 3 轮
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# 5. 开始微调！
print("开始微调模型...")
trainer.train()

# 保存模型
trainer.save_model("./my_shakespeare_classifier")
tokenizer.save_pretrained("./my_shakespeare_classifier")
print("微调完成，模型已保存！")
