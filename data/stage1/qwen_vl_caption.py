import argparse
import json
import os
from queue import Queue
from threading import Lock, Thread

from openai import OpenAI
from tqdm import tqdm

file_lock = Lock()


def json_post_process(response: str) -> str:
    """Extract Description field from JSON response."""
    # Find JSON in response
    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        return response  # Return raw response if no JSON found

    json_str = response[start : end + 1]

    try:
        response_json = json.loads(json_str)
        # Extract Description field (case-insensitive)
        for key in ["Description", "description", "DESCRIPTION"]:
            if key in response_json:
                return response_json[key]
        # If no Description field, return the whole JSON as string
        return json.dumps(response_json, ensure_ascii=False)
    except Exception:
        return response  # Return raw response if JSON parsing fails


def save_result(key: str, response: str, save_path: str):
    """Thread-safe result saving"""
    with file_lock:
        file_path = os.path.join(save_path, f"{key}.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(response)


def worker_fn(
    task_queue: Queue,
    result_queue: Queue,
    progress_bar: tqdm,
    prompt: dict,
    save_path: str,
    args,
):
    """Worker function for processing spectrograms"""
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
            key, image_path = data

            absolute_image_path = os.path.abspath(image_path)
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"file://{absolute_image_path}"},
            }

            # API call
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {"type": "text", "text": prompt["user"]},
                        ],
                    },
                ],
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                max_tokens=4096,
                extra_body={"top_k": 20, "repetition_penalty": 1.0},
            )

            response = completion.choices[0].message.content
            response = response.strip()

            # Extract Description field from JSON response
            description = json_post_process(response)

            # Save result immediately as .txt
            save_result(key, description, save_path)

            result_queue.put((key, None))

        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            result_queue.put((key, str(e)))
        finally:
            task_queue.task_done()
            progress_bar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Acoustic description from spectrogram using Qwen-VL-32B-Instruct"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Input JSON file containing spectrogram image paths to generate description.",
    )
    parser.add_argument(
        "--user_prompt",
        "-up",
        type=str,
        default="spectrogram_user_prompt.txt",
        help="File with prompt",
    )
    parser.add_argument(
        "--system_prompt",
        "-sp",
        type=str,
        default="spectrogram_system_prompt.txt",
        help="File with system prompt",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        default="../../workdir/acoustic_descriptions",
        help="Path to save individual description files.",
    )
    parser.add_argument(
        "--spectrogram_path",
        "-spec",
        type=str,
        default="../../workdir/spectrograms",
        help="Path to spectrogram images.",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=32,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://127.0.0.1:8002/v1",
        help="Base URL for vLLM server",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Model name for vLLM server",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompt and input data
    prompt = {
        "user": open(args.user_prompt, "r").read(),
        "system": open(args.system_prompt, "r").read(),
    }
    input_data = json.load(open(args.input, "r"))
    os.makedirs(args.save_path, exist_ok=True)

    # Prepare tasks
    tasks = []
    skipped = 0
    for key, data in tqdm(
        list(input_data.items()), desc="Checking existing results", dynamic_ncols=True
    ):
        result_file = os.path.join(args.save_path, f"{key}.txt")
        if os.path.exists(result_file):
            skipped += 1
            continue
        spectrogram_path = os.path.join(args.spectrogram_path, f"{key}.jpg")
        if os.path.exists(spectrogram_path):
            tasks.append((key, spectrogram_path))
        else:
            print(f"Warning: No spectrogram_path found for {key}")

    print(f"Total files: {len(input_data)}")
    print(f"Already processed: {skipped}")
    print(f"To process: {len(tasks)}")

    # Setup queues and progress bar
    task_queue = Queue(maxsize=args.num_workers * 2)
    result_queue = Queue()
    progress_bar = tqdm(
        total=len(tasks), desc="Processing spectrograms", dynamic_ncols=True
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
    errors = []
    successful = 0

    while not result_queue.empty():
        key, error = result_queue.get()
        if error:
            errors.append(f"{key}: {error}")
        else:
            successful += 1

    # Print summary
    print(f"\nProcessed: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Errors: {len(errors)}")
    print(f"Individual files saved to: {args.save_path}")

    if errors:
        error_file = "acoustic_description_errors.txt"
        with open(error_file, "w") as f:
            f.write("\n".join(errors))
        print(f"Errors saved to {error_file}")


if __name__ == "__main__":
    main()
