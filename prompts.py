"""Prompt pieces shared by training data, GRPO data and evaluation.

The system prompt and MCQ question template must stay identical between the SFT/DPO/GRPO
training data and evaluation (eval/run_inference_vllm.py).
"""

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it. The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning "
    "process here </think> <answer> answer here </answer>."
)


def mcq_prompt(question: str, choices_text: str) -> str:
    """User prompt for a multiple-choice question; `choices_text` has one "(A) ..." per line."""
    return (
        f"{question} Choose the correct option from the following options:\n"
        f"{choices_text}\n"
        "Please think and reason about the input audio before you respond."
    )
