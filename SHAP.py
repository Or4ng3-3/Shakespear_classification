import shap
import torch
import numpy as np
from transformers import pipeline
from IPython.display import display, HTML

# 1. 初始化 Pipeline (确保加载你的模型)
model_path = "./my_shakespeare_classifier"
pipe = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
)

# 2. 准备文本 (这段是罗密欧与朱丽叶，属于早期)
sample_text = """
[CHARACTER] And too soon marr'd are those so early made.
The earth hath swallow'd all my hopes but she,
She is the hopeful lady of my earth:
But woo her, gentle [CHARACTER], get her heart,
My will to her consent is but a part;
"""

# 3. 运行预测并获取概率最高的标签
preds = pipe(sample_text)[0]
top_pred = sorted(preds, key=lambda x: x["score"], reverse=True)[0]
top_label_index = int(top_pred["label"].split("_")[1])  # 获取数字索引 0, 1, or 2
print(f"✅ 模型预测结果: {top_pred}")

# 4. 初始化 SHAP 解释器
explainer = shap.Explainer(pipe)
shap_values = explainer([sample_text])

# --- 可视化与分析 ---

# 尝试强制显示 HTML
shap.initjs()
try:
    # 显示模型认为最可能的那个标签的贡献情况
    display(shap.plots.text(shap_values[0, :, top_pred["label"]]))
except Exception as e:
    print("HTML 绘图失败，请直接查看下方的文字分析报告。")

# --- 稳健的文字版分析报告 ---
print(f"\n--- 文体贡献度分析 (针对预测标签: {top_pred['label']}) ---")

# 提取 Token 和对应的 SHAP 值
# 注意：Transformers 的 SHAP 输出结构中，shap_values.values[0] 的维度是 [Tokens, Labels]
tokens = shap_values.data[0]
# 找到对应 Top 标签的那一列分数
scores = shap_values.values[0][:, top_label_index]

# 配对、排序并过滤掉微小的噪音
word_importance = []
for t, s in zip(tokens, scores):
    word_importance.append((t, s))

# 按绝对值大小排序，看影响力最大的词
word_importance = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)

print(f"{'单词 (Token)':<20} | {'贡献分':<10} | {'影响'}")
print("-" * 50)

for token, score in word_importance:
    if abs(score) > 0.001:  # 降低阈值，确保能看到输出
        direction = "🔴 增加概率" if score > 0 else "🔵 降低概率"
        # 清理 token 中的特殊字符（如 BERT 的 ## 或   符号）
        clean_token = token.replace(" ", "").replace("##", "")
        if clean_token.strip():
            print(f"{clean_token:<20} | {score:>8.4f} | {direction}")
