import argparse
import json
import os
from queue import Queue
from threading import Lock, Thread

from openai import OpenAI
from tqdm import tqdm

file_lock = Lock()

SYSTEM_PROMPT = (
    "You are a question generation assistant.\n"
    "Your task is to generate more challenging questions based on audio and original questions.\n"
    "You must strictly follow the user's instructions and output format.\n"
    "Return ONLY valid JSON and nothing else."
)


def json_post_process(response: str) -> dict:
    start = response.find("{")
    end = response.rfind("}")

    json_str = response[start : end + 1]

    try:
        response_json = json.loads(json_str)
        return response_json
    except Exception as e:
        return None


def save_result(key: str, response_json: dict, save_path: str):
    """Thread-safe result saving"""
    with file_lock:
        file_path = os.path.join(save_path, f"{key}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(response_json, f, indent=2, ensure_ascii=False)


def build_user_prompt(
    data: dict, audio_description: str, user_prompt_template: str
) -> str:

    prompt = (
        user_prompt_template.replace("__ORIGINAL_QUESTION__", data["question"])
        .replace("__ORIGINAL_RESPONSE__", data["response"])
        .replace("__AUDIO_DESCRIPTION__", audio_description)
    )

    return prompt


def worker_fn(
    task_queue: Queue,
    result_queue: Queue,
    progress_bar: tqdm,
    prompt: dict,
    save_path: str,
    args,
):
    """Worker function for processing with vLLM"""
    client = OpenAI(
        base_url=args.vllm_base_url,
        api_key="EMPTY",
    )
    model = args.vllm_model

    while True:
        data = task_queue.get()
        if data is None:  # Termination signal
            task_queue.task_done()
            break

        try:
            key, entry, audio_description = data
            user_prompt = build_user_prompt(entry, audio_description, prompt["user"])

            # API call
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048,
                extra_body={"top_k": 20, "min_p": 0},
            )

            message = completion.choices[0].message
            content = message.content.strip() if message.content else ""
            if content == "":
                raise ValueError("Model response is empty")

            response_json = json_post_process(content)

            # Save result immediately
            if response_json is not None:
                save_result(key, response_json, save_path)
                result_queue.put((key, response_json, None))
            else:
                result_queue.put((key, None, "JSON parsing error"))

        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            result_queue.put((key, None, str(e)))
        finally:
            task_queue.task_done()
            progress_bar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generating harder questions based on original questions and audio descriptions"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Input JSON file containing metadata with questions and responses.",
    )
    parser.add_argument(
        "--user_prompt",
        "-up",
        type=str,
        default="qa_generate_user_prompt.txt",
        help="File with user prompt",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        default="../../workdir/questions_hard",
        help="Path to save generated harder questions.",
    )
    parser.add_argument(
        "--description_path",
        "-d",
        type=str,
        default="../../workdir/descriptions",
        help="Path to audio descriptions",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=256,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://127.0.0.1:8003/v1",
        help="Base URL for vLLM server",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        help="Model name for vLLM server",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompt and input data
    prompt = {
        "user": open(args.user_prompt, "r").read(),
        "system": SYSTEM_PROMPT,
    }

    input_data = json.load(open(args.input, "r"))
    print(f"Loaded {len(input_data)} entries from {args.input}")

    os.makedirs(args.save_path, exist_ok=True)

    # Prepare tasks - skip already processed files
    tasks = []
    skipped = 0

    for key, data in tqdm(
        list(input_data.items()), dynamic_ncols=True, desc="Checking existing results"
    ):
        description_file = os.path.join(args.description_path, f"{key}.txt")
        if not os.path.exists(description_file):
            print(f"Warning: Description file not found for {key}, skipping.")
            continue
        if "question" in data and "response" in data:
            # Check if result file already exists
            result_file = os.path.join(args.save_path, f"{key}.json")

            if os.path.exists(result_file):
                skipped += 1
                continue

            audio_description = open(description_file, "r").read().strip()
            tasks.append((key, data, audio_description))
        else:
            print(f"Warning: No question or response found for {key}")

    print(f"Total files: {len(input_data)}")
    print(f"Already processed: {skipped}")
    print(f"To process: {len(tasks)}")

    if len(tasks) == 0:
        print("All files already processed!")
        return

    # Setup queues and progress bar
    task_queue = Queue(maxsize=args.num_workers * 2)
    result_queue = Queue()
    progress_bar = tqdm(
        total=len(tasks), desc="Generating harder questions", dynamic_ncols=True
    )

    # Start worker threads
    threads = []
    for _ in range(args.num_workers):
        thread = Thread(
            target=worker_fn,
            args=(task_queue, result_queue, progress_bar, prompt, args.save_path, args),
        )
        thread.start()
        threads.append(thread)

    # Add tasks to queue
    for task in tasks:
        task_queue.put(task)

    # Send termination signals
    for _ in range(args.num_workers):
        task_queue.put(None)

    # Wait for completion
    task_queue.join()
    for thread in threads:
        thread.join()

    progress_bar.close()

    # Collect results
    result_cnt = 0
    errors = []

    while not result_queue.empty():
        key, response_json, error = result_queue.get()
        if error:
            errors.append(f"{key}: {error}")
        else:
            result_cnt += 1

    # Print summary
    print(f"\nProcessed: {len(tasks)}")
    print(f"Successful: {result_cnt}")
    print(f"Errors: {len(errors)}")
    print(f"Skipped (already done): {skipped}")
    print(f"Individual files saved to: {args.save_path}")

    if errors:
        error_file = "generate_questions_errors.txt"
        with open(error_file, "w") as f:
            f.write("\n".join(errors))
        print(f"Errors saved to {error_file}")


if __name__ == "__main__":
    main()
