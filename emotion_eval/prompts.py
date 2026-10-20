"""分类提示词：其他模块可直接导入，填入候选类别后作为参数传递。"""

EMOTION_PROMPT = (
    "What is the primary emotion conveyed by the Image?\n"
    "Choose exactly one label from: {labels}.\n"
    "Reply with only that label, using the capitalization shown above, without any explanation."
)
