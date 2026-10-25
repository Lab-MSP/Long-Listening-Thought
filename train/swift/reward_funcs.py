"""MCQ accuracy reward for ms-swift GRPO (used with the built-in `format` reward).

    swift rlhf --rlhf_type grpo \
        --external_plugins train/swift/reward_funcs.py \
        --reward_funcs mcq_accuracy format --reward_weights 1.0 0.05 ...

`mcq_accuracy` gives 1.0 when the text inside <answer></answer> exactly matches the gold
option string (e.g. "(B)"), else 0.0. `format` is ms-swift's built-in reward that checks
`<think>...</think> <answer>...</answer>`.
"""

import re
from typing import List, Optional

from swift.plugin import ORM, orms


def extract_choice(text: str) -> Optional[str]:
    """Return the content between <answer></answer> tags, or None."""
    if not text:
        return None
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class MCQAccuracyReward(ORM):

    def __call__(
        self, completions: List[str], solution: Optional[List[str]] = None, **kwargs
    ) -> List[float]:
        rewards = []
        for completion in completions:
            # per_device_train_batch_size=1: all completions in a call share one prompt
            gold = solution[0]
            pred = extract_choice(completion)
            rewards.append(1.0 if pred is not None and gold is not None and pred == gold else 0.0)
        return rewards


orms["mcq_accuracy"] = MCQAccuracyReward
