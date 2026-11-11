"""分类提示词：其他模块可直接导入，填入候选类别后作为参数传递。"""

EMOTION_PROMPT = (
    "What is the primary emotion conveyed by the Image?\n"
    "Choose exactly one label from: {labels}.\n"
    "Reply with only that label, using the capitalization shown above, without any explanation."
)

DEEPSEEK_EMOTION_PROMPT = (
    "What is the primary emotion conveyed by the Image?\n"
    "Choose exactly one label from: {labels}.\n"
    "First, briefly explain the visual evidence for your choice in 2-4 sentences, "
    "placing the explanation inside <draft> and </draft>.\n"
    "Then give your final answer inside \\boxed{{}}, using exactly one label "
    "with the capitalization shown above.\n"
    "Use exactly the following format, with no text outside it:\n"
    "<draft>\nYour explanation.\n</draft>\n"
    "\\boxed{{LABEL}}"
)

EMOTION_ANALYSIS_DRAFT_PROMPT = (
    "What is the primary emotion conveyed by the image?\n"
    "Choose exactly one label from: {labels}.\n"
    "First, List 1-3 emotion-relevant analysis angles for this image in <draft>...</draft>, without analysis. " 
    "Then jointly analyze the visible evidence from these angles in one concise paragraph to infer the emotion. "
    "Finally, output the chosen label in \\boxed{{}}."
)

