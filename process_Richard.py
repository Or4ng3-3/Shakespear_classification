import re


def clean_shakespeare_final(input_file, output_file, character_names):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. 去掉所有方括号及其中的内容 (舞台提示)
    text = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)

    # 2. 去除数字 (行号)
    text = re.sub(r"\d+", "", text)

    # 3. 核心规则逻辑：处理行首说话者标签
    # 只要行首开始的句号 . 后面跟着字母，就判定为说话者，换成 [CHARACTER]
    # (?=[ \t]*[a-zA-Z]) 表示后面可以有空格但必须有字母
    text = re.sub(
        r"^\s*[^.\n]+\.(?=[ \t]*[a-zA-Z])", "[CHARACTER]", text, flags=re.MULTILINE
    )

    # 4. 替换文中提到的所有人名 (Inline Mentions)
    # 将名单按长度倒序排列，防止短名字先匹配导致长名字损坏
    sorted_names = sorted(character_names, key=len, reverse=True)

    for name in sorted_names:
        # \b 确保是独立单词匹配，re.IGNORECASE 忽略大小写
        name_pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        text = name_pattern.sub("[CHARACTER]", text)

    # 5. 最后清理：合并多余空行，去掉行首位空格
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"全自动化清洗完成！已生成: {output_file}")


# --- 配置区 ---

# 建议尽可能详尽地列出该剧本中的人名、头衔
# 即使“句号规则”漏掉了一些不带句号的标签，这里的人名匹配也能补刀
richard_iii_names = [
    "Richard III",
    "Richard",
    "Gloucester",
    "Duke of Gloucester",
    "George Plantagenet",
    "Clarence",
    "Duke of Clarence",
    "King Edward IV",
    "Edward",
    "King Edward",
    "Lady Anne",
    "Anne",
    "Queen Margaret",
    "Margaret",
    "Elizabeth",
    "Queen Elizabeth",
    "Buckingham",
    "Duke of Buckingham",
    "Rivers",
    "Lord Rivers",
    "Hastings",
    "Lord Hastings",
    "Stanley",
    "Lord Stanley",
    "Richmond",
    "Henry Tudor",
    "York",
    "Duchess of York",
    "Dorset",
    "Grey",
    "Vaughan",
]

# 执行
clean_shakespeare_final("Richard_III.txt", "Richard_III_cleaned.txt", richard_iii_names)
