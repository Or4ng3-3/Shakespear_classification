import re


def clean_shakespeare_final(input_file, output_file, character_names):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. 首先去除 --- 之间的块内容（包含换行）
    # 常用语去除 Project Gutenberg 的版权声明或页眉页脚
    text = re.sub(r"---+.*?---+", "", text, flags=re.DOTALL)

    # 2. 去掉所有方括号及其中的内容 (舞台提示)
    text = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)

    # 3. 去除数字 (行号)
    text = re.sub(r"\d+", "", text)

    # 4. 核心规则逻辑：处理行首说话者标签 (句号后有字母)
    text = re.sub(
        r"^\s*[^.\n]+\.(?=[ \t]*[a-zA-Z])", "[CHARACTER]", text, flags=re.MULTILINE
    )

    # 5. 替换文中提到的所有人名
    sorted_names = sorted(character_names, key=len, reverse=True)
    for name in sorted_names:
        name_pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        text = name_pattern.sub("[CHARACTER]", text)

    # 6. 最后清理多余空行
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"全自动化清洗完成！已生成: {output_file}")


# --- 配置区 ---

# 建议尽可能详尽地列出该剧本中的人名、头衔
# 即使“句号规则”漏掉了一些不带句号的标签，这里的人名匹配也能补刀
macbeth_names = [
    # 主要人物及其全称/头衔
    "Lady Macbeth",
    "Thane of Cawdor",
    "Thane of Glamis",
    "Thane of Fife",
    "King Duncan",
    "Lady Macduff",
    "Young Siward",
    "Old Siward",
    "Weird Sisters",
    "First Witch",
    "Second Witch",
    "Third Witch",
    "Prince of Cumberland",
    "Lord Macduff",
    "Macbeth",
    "Banquo",
    "Duncan",
    "Malcolm",
    "Donalbain",
    "Macduff",
    "Lennox",
    "Ross",
    "Menteith",
    "Angus",
    "Caithness",
    "Fleance",
    "Siward",
    "Seyton",
    "Hecate",
    "Seyward",  # Siward的变体拼写
    # 身份/职业（常在剧中作为代称出现）
    "Gentlewoman",
    "Messenger",
    "Attendant",
    "Doctor",
    "Porter",
    "Murderer",
    "Apparition",
    "General",
    "King",
    "Queen",
]
# 执行
clean_shakespeare_final(
    "./rawdata/Macbeth_original.txt", "./rawdata/Macbeth_cleaned.txt", macbeth_names
)
