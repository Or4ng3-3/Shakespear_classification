import shap
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from IPython.display import display, HTML

# 1. 加载你训练好的模型和分词器
model_path = "./my_shakespeare_classifier"  # 请确保这是你保存模型的路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. 创建一个用于分类的 pipeline
# top_k=None 会让模型输出所有类别的分数，这对 SHAP 很重要
pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
)

# 3. 定义标签映射（为了方便阅读）
# 0: Early (Richard III), 1: Middle (Macbeth), 2: Late (The Tempest)
label_map = {
    "LABEL_0": "Early (Richard III)",
    "LABEL_1": "Middle (Macbeth)",
    "LABEL_2": "Late (The Tempest)",
}

# 4. 准备你要分析的文本片段
# 建议从你的 cleaned 数据里找一段，或者手动输入一段
sample_text = """
[CHARACTER] Good, speak to the [CHARACTER]: fall to't, yarely, 
or we run ourselves aground: bestir, bestir. 
[CHARACTER] Heigh, my hearts! cheerly, cheerly, my hearts! 
[CHARACTER] Tend to the [CHARACTER] Blow, till thou burst thy wind, 
if room enough!
"""

# 5. 初始化 SHAP 解释器
# SHAP 会根据模型预测所有类别的情况来计算贡献度
explainer = shap.Explainer(pipe)
shap_values = explainer([sample_text])

# --- 核心：可视化部分 ---

# 步骤 A: 在 Notebook 环境中必须初始化 JS
shap.initjs()

print(f"\n--- 正在为以下文本生成解释 ---")
print(sample_text.strip())
print(f"----------------------------------\n")

# 步骤 B: 尝试显示交互式 HTML 图片
# 我们主要观察 LABEL_2 (晚期)，因为 sample_text 来自《暴风雨》
# 如果你想看模型为什么觉得它是晚期，就选 [:, "LABEL_2"]
try:
    # 强制尝试显示第一个样本对 LABEL_2 的解释
    print("正在生成可视化热力图（红色代表支持该分类，蓝色代表反对）...")
    shap.plots.text(shap_values[0, :, "LABEL_2"])
except Exception as e:
    print(f"可视化组件显示失败，错误信息: {e}")

# --- 步骤 C: 备份方案 —— 打印文字版贡献度排名 ---
print("\n--- 文字版贡献度分析 (针对 Late/晚期 标签) ---")
# 获取单词列表
tokens = shap_values.data[0]
# 获取对 LABEL_2 的贡献分数
# shap_values.values 的维度通常是 [样本数, token数, 类别数]
# 这里的 2 对应 LABEL_2 (晚期)
scores = shap_values.values[0][:, 2]

# 将单词和分数配对并排序
word_importance = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)

print(f"{'单词 (Token)':<20} | {'对【晚期】风格的贡献分':<20}")
print("-" * 45)
for token, score in word_importance:
    if abs(score) > 0.01:  # 只打印有意义的词
        indicator = "🔴" if score > 0 else "🔵"
        print(f"{token:<20} | {score:>8.4f} {indicator}")

print("\n分析建议：")
print("1. 观察红色的词：这些词是模型判断它是【晚期】的关键证据。")
print("2. 检查 [CHARACTER]：如果它的分数接近 0，说明模型成功忽略了角色名。")
print("3. 观察标点符号：标点是否有高分？这反映了晚期作品特有的节奏感。")
