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
    evaluation_strategy="epoch",  # 每个 epoch 评估一次
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
