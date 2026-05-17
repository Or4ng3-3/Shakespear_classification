import shap
from transformers import pipeline

# 1. 创建分类 Pipeline，加载你刚才保存的模型
# device=0 表示使用GPU，如果是CPU则为 device=-1
pipe = pipeline(
    "text-classification",
    model="./my_shakespeare_classifier",
    tokenizer="./my_shakespeare_classifier",
    device=0,
)

# 2. 准备一段你想测试的文本 (从你的数据里挑一段，或者找一段模型没见过的)
sample_text = """
[CHARACTER] Good, speak to the [CHARACTER]: fall to't, yarely,
or we run ourselves aground: bestir, bestir.
[CHARACTER] Heigh, my hearts! cheerly, cheerly, my hearts!
"""

print(f"模型的预测结果: {pipe(sample_text)}")

# 3. 初始化 SHAP 解释器
# SHAP 会自动扰动文本中的词，计算每个词对最终分类结果的贡献度
explainer = shap.Explainer(pipe)

# 4. 计算 SHAP 值
shap_values = explainer([sample_text])

# 5. 可视化！(在 Jupyter/Colab 中运行会生成交互式彩色图表)
shap.plots.text(shap_values)
