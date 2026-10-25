import gc
import json
import logging
import os
import sys
import time

# Set logging level BEFORE importing other libraries
logging.basicConfig(level=logging.ERROR)
for logger_name in [
    "transformers",
    "processing_qwen2_5_omni",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
from vllm import LLM, SamplingParams  # vLLM import

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prompts import SYSTEM_PROMPT, mcq_prompt  # noqa: E402

logger = logging.getLogger(__name__)

# Set vLLM environment variables --> For reproduciblity of the results
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"



class QwenRunner:
    def __init__(self, model_name="Qwen/Qwen2.5-Omni-7B"):
        self.model_name = model_name
        print(f"Loading model: {model_name}")

        # Processor is used only for:
        #   - applying the chat template (turning conversation into the <|im_start|> format)
        #   - NOT for tokenization/inference (vLLM handles that)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print("Processor loaded.")

        # Suppress noisy logs (especially system prompt warnings)
        logger.setLevel(logging.ERROR)

        # --- vLLM: initialize LLM engine ---
        self.llm = LLM(
            model=model_name,
            max_model_len=8192,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_prefix_caching=False,
            enforce_eager=True,
            limit_mm_per_prompt={
                "audio": 2,
                "image": 1,
                "video": 1,
            },
        )

        # Sampling parameters for generation (greedy decoding)
        self.sampling_params = SamplingParams(
            max_tokens=4096,
            temperature=0,  # greedy decoding (deterministic)
            top_k=-1,
            seed=1234,
        )
        print(f"In sampling mode: {self.sampling_params}")
        print("vLLM model loaded.")

    def clear_gpu_cache(self):
        """Clear PyTorch CUDA cache (does not touch vLLM KV cache)."""
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU available: {torch.cuda.is_available()}")

    def load_data(self, data_file="./MMAU/mmau-test-mini.json"):
        with open(data_file, "r") as f:
            self.data = json.load(f)

    def format_choices(self, choices):
        """Format choices with labels (A), (B), etc. if not already present."""
        import re

        formatted = []
        label_pattern = re.compile(r"^\s*(\([A-Z]\)|[A-Z]\.)\s")

        for i, choice in enumerate(choices):
            if not choice:
                continue  # Skip empty choices
            if label_pattern.match(choice):
                formatted.append(choice)
            else:
                label = (
                    chr(ord("A") + i)
                    if i < 26
                    else chr(ord("A") + i // 26 - 1) + chr(ord("A") + i % 26)
                )
                formatted.append(f"({label}) {choice}")
        return "\n".join(formatted)

    def build_trained_conversation(
        self, item, audio_path_key="audio_path", audio_base_path=""
    ):
        """Prompt style for the SFT model that uses <think>...</think> and <answer>...</answer>."""
        audio_path = item[audio_path_key]
        if audio_base_path:
            # Remove leading ./ or / from audio_path
            if audio_path.startswith("./"):
                audio_path = audio_path[2:]
            elif audio_path.startswith("/"):
                audio_path = audio_path[1:]
            audio_path = os.path.join(audio_base_path, audio_path)

        question = item["question"]
        choices = item["choices"]
        choices_text = self.format_choices(choices)

        prompt = mcq_prompt(question, choices_text).strip()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return conversation

    def build_mmsu_conversation(
        self, item, audio_path_key="audio_path", audio_base_path=""
    ):
        """MMSU prompt style matching mmsu_inference.py format."""
        audio_path = item[audio_path_key]
        if audio_base_path:
            # Remove leading ./ or / from audio_path
            if audio_path.startswith("./"):
                audio_path = audio_path[2:]
            elif audio_path.startswith("/"):
                audio_path = audio_path[1:]
            audio_path = os.path.join(audio_base_path, audio_path)

        question = item["question"]
        choice_a = item["choice_a"]
        choice_b = item["choice_b"]
        choice_c = item.get("choice_c", "")
        choice_d = item.get("choice_d", "")

        choices = []
        for choice in [choice_a, choice_b, choice_c, choice_d]:
            if choice:
                choices.append(str(choice))

        choices_text = self.format_choices(choices)

        prompt = mcq_prompt(question, choices_text).strip()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return conversation

    def extract_answer(self, model_response):
        """Extract answer from <answer></answer> tags."""
        import re

        match = re.search(r"<answer>\s*(.*?)\s*</answer>", model_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def choice_letter_to_text(self, letter: str, choices: list) -> str:
        """Convert choice letter (A, B, ...) to actual choice text."""
        idx = ord(letter.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
        return ""

    def _build_vllm_inputs(self, text, audios, images, videos, use_audio_in_video):
        """
        Build the input dict in the format expected by vLLM:

            {
                "prompt": prompt_string,
                "multi_modal_data": {
                    "audio": ...,
                    "image": ...,
                    "video": ...,
                },
                "mm_processor_kwargs": {...}  # optional
            }

        NOTE:
        - We simply pass through what `process_mm_info` returns and let vLLM/Qwen
          handle the details, assuming the formats are already compatible.
        """
        multi_modal_data = {}
        mm_processor_kwargs = {}

        if audios:
            multi_modal_data["audio"] = audios
        if images:
            multi_modal_data["image"] = images
        if videos:
            multi_modal_data["video"] = videos
        if use_audio_in_video:
            mm_processor_kwargs["use_audio_in_video"] = True

        inputs = {"prompt": text}
        if multi_modal_data:
            inputs["multi_modal_data"] = multi_modal_data
        if mm_processor_kwargs:
            inputs["mm_processor_kwargs"] = mm_processor_kwargs

        return inputs

    def run_inference(
        self,
        out_file="MMAU-predictions.json",
        audio_path_key="audio_path",
        audio_base_path="",
        model_output_key="model_prediction",
        save_responses=True,
        dataset_type="default",
    ):
        """
        Batched vLLM inference:
        - Build all prompts + multimodal inputs first
        - Call vLLM.generate once on the whole list (or in chunks if needed)
        - Map outputs back to self.data in order

        Args:
            dataset_type: One of "default", "mmsu". Use "mmsu" for MMSU benchmark.
            audio_base_path: Base path for audio files (will be prepended to audio_path_key)
        """
        print_response = True
        response_dict = {} if save_responses else None

        # 1) Build all requests first
        vllm_requests = []  # list of (index, item, vllm_inputs)
        USE_AUDIO_IN_VIDEO = False

        for idx, item in enumerate(self.data):
            if dataset_type == "mmsu":
                conversation = self.build_mmsu_conversation(
                    item, audio_path_key=audio_path_key, audio_base_path=audio_base_path
                )
            else:
                conversation = self.build_trained_conversation(
                    item, audio_path_key=audio_path_key, audio_base_path=audio_base_path
                )

            # Apply chat template to get the text prompt
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Extract multimodal inputs
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )

            # Build vLLM input dict
            vllm_inputs = self._build_vllm_inputs(
                text=text,
                audios=audios,
                images=images,
                videos=videos,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )

            vllm_requests.append((idx, item, vllm_inputs))

        if not vllm_requests:
            return response_dict

        # 2) Run vLLM generate in batch
        # If you want to chunk (e.g., for memory reasons), you can slice vllm_inputs_list.
        vllm_inputs_list = [req[2] for req in vllm_requests]

        outputs = self.llm.generate(
            vllm_inputs_list,
            sampling_params=self.sampling_params,
        )

        # 3) Map outputs back to items in the same order
        for (idx, item, _), out in zip(vllm_requests, outputs):
            # vLLM's RequestOutput format: out.outputs[0].text is the generated string
            model_response = out.outputs[0].text.strip()

            # For SFT model, try to extract content inside <answer>...</answer>
            answer = self.extract_answer(model_response)
            if answer:
                # Check if answer is in (A), (B) format or single letter A, B format
                import re

                match = re.match(r"\(([A-Z])\)", answer) or re.match(
                    r"^([A-Z])$", answer
                )
                if match:
                    letter = match.group(1)
                    if dataset_type == "mmsu":
                        item[model_output_key] = letter
                    else:
                        item[model_output_key] = self.choice_letter_to_text(
                            letter, item["choices"]
                        )
                else:
                    item[model_output_key] = answer
            else:
                item[model_output_key] = model_response

            if save_responses:
                # Use the sample's id as key if present, otherwise index
                key = item.get("id", idx)
                response_dict[key] = model_response

            # Print the first response for inspection
            if print_response:
                print(f"Prompt: {conversation}")
                print(f"Model Response (idx={idx}): {model_response}")
                print(f"Extracted Answer: {item[model_output_key]}")
                print_response = False

        # 4) Save predictions to file
        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # For MMSU, save as JSONL for compatibility with evaluation.py
        if dataset_type == "mmsu":
            with open(out_file, "w") as f:
                for item in self.data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(out_file, "w") as f:
                json.dump(self.data, f, indent=2)

        return response_dict


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="+",
        default=["MMAR/MMAR-meta.json"],
        help="Path(s) to the input data file(s). Can specify multiple.",
    )
    parser.add_argument(
        "--out_files",
        type=str,
        nargs="+",
        default=["MMAR-predictions.json"],
        help="Path(s) to the output predictions file(s). Must match data_files count.",
    )
    parser.add_argument(
        "--response_files",
        type=str,
        nargs="+",
        default=[""],
        help="Path(s) to save full response as JSON. Must match data_files count.",
    )
    parser.add_argument(
        "--audio_base_paths",
        type=str,
        nargs="+",
        default=[""],
        help="Base path(s) for audio files (will be prepended to audio_path_key). Must match data_files count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()

    # Validate argument counts match
    num_datasets = len(args.data_files)
    if len(args.out_files) != num_datasets:
        raise ValueError(
            f"Number of out_files ({len(args.out_files)}) must match data_files ({num_datasets})"
        )

    # Pad response_files with empty strings if needed
    response_files = args.response_files
    while len(response_files) < num_datasets:
        response_files.append("")

    # Pad audio_base_paths with empty strings if needed
    audio_base_paths = args.audio_base_paths
    while len(audio_base_paths) < num_datasets:
        audio_base_paths.append("")

    # Load model once
    runner = QwenRunner(model_name=args.model_name)
    runner.clear_gpu_cache()

    total_samples = 0

    # Process each dataset
    for i, (data_file, out_file, response_file, audio_base_path) in enumerate(
        zip(args.data_files, args.out_files, response_files, audio_base_paths)
    ):
        print(f"\n{'='*50}")
        print(f"Processing dataset {i+1}/{num_datasets}: {data_file}")
        print(f"{'='*50}")

        dataset_start = time.time()

        runner.load_data(data_file=data_file)

        # Detect dataset type and set appropriate settings
        data_file_lower = data_file.lower()
        if "mmsu" in data_file_lower:
            dataset_type = "mmsu"
            audio_path_key = "audio_path"
            model_output_key = "response"  # key read by the official MMSU scorer
        elif "mmau" in data_file_lower:
            dataset_type = "default"
            audio_path_key = "audio_id"
            model_output_key = "model_output"  # key read by the official MMAU scorer
        else:
            dataset_type = "default"
            audio_path_key = "audio_path"
            model_output_key = "answer_prediction"  # key read by the official MMAR scorer

        print(f"Dataset type: {dataset_type}, audio_path_key: {audio_path_key}")

        response_dict = runner.run_inference(
            out_file=out_file,
            audio_path_key=audio_path_key,
            audio_base_path=audio_base_path,
            model_output_key=model_output_key,
            dataset_type=dataset_type,
        )

        if response_file:
            resp_dir = os.path.dirname(response_file)
            if resp_dir:
                os.makedirs(resp_dir, exist_ok=True)
            with open(response_file, "w") as f:
                json.dump(response_dict, f, indent=2)
            print(f"Full responses saved to {response_file}")

        dataset_elapsed = time.time() - dataset_start
        print(f"Dataset {i+1} completed in {dataset_elapsed:.2f} seconds")
        print(
            f"Samples: {len(runner.data)}, Avg: {dataset_elapsed/len(runner.data):.2f} sec/sample"
        )

        total_samples += len(runner.data)

    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"ALL DATASETS COMPLETE")
    print(f"{'='*50}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Total samples processed: {total_samples}")
    print(f"Average time per sample: {elapsed_time/total_samples:.2f} seconds")
    print(f"{'='*50}")
