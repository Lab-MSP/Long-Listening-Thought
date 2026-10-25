import argparse
import json
import os
import shlex
from queue import Queue
from threading import Lock, Thread

from openai import OpenAI
from tqdm import tqdm

file_lock = Lock()


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
    save_path: str,
    args,
):
    """Worker function for processing audio files"""
    client = OpenAI(
        base_url=args.vllm_base_url,
        api_key="EMPTY",
    )

    while True:
        data = task_queue.get()
        if data is None:  # Termination signal
            task_queue.task_done()
            break

        try:
            key, audio_path = data

            # API call
            absolute_audio_path = os.path.abspath(audio_path)
            absolute_audio_path = shlex.quote(absolute_audio_path)
            completion = client.chat.completions.create(
                model=args.vllm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {
                                    "url": f"file://{absolute_audio_path}",
                                },
                            }
                        ],
                    }
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=2048,
                extra_body={"top_k": 20},
            )
            response = completion.choices[0].message.content
            response = response.strip()

            save_result(key, response, save_path)

            result_queue.put((key, response, None))

        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            result_queue.put((key, None, str(e)))
        finally:
            task_queue.task_done()
            progress_bar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic description from audio using Qwen3-Omni-Captioner"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Input JSON file containing audio file paths.",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="",
        help="Root directory of the AF-Think audio; metadata wav_path is relative to it",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        default="../../workdir/semantic_descriptions",
        help="Path to save individual description files.",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=16,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://127.0.0.1:8001/v1",
        help="Base URL for vLLM server",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        help="Model name for vLLM server",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load input data
    input_data = json.load(open(args.input, "r"))
    os.makedirs(args.save_path, exist_ok=True)

    # Prepare tasks - skip already processed files
    tasks = []
    skipped = 0

    for key, data in tqdm(
        list(input_data.items()), desc="Checking existing results", dynamic_ncols=True
    ):
        if "wav_path" in data:
            # Check if result file already exists
            result_file = os.path.join(args.save_path, f"{key}.txt")

            if os.path.exists(result_file):
                skipped += 1
                continue
            tasks.append((key, os.path.abspath(os.path.join(args.audio_root, data["wav_path"]))))
        else:
            print(f"Warning: No wav_path found for {key}")

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
        total=len(tasks), desc="Processing wav files", dynamic_ncols=True
    )

    # Start worker threads
    threads = []
    for _ in range(args.num_workers):
        thread = Thread(
            target=worker_fn,
            args=(task_queue, result_queue, progress_bar, args.save_path, args),
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

    # Collect results (optional - for final JSON output)
    results = {}
    errors = []

    while not result_queue.empty():
        key, response, error = result_queue.get()
        if error:
            errors.append(f"{key}: {error}")
        else:
            results[key] = response

    # Print summary
    print(f"\nProcessed: {len(tasks)}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Skipped (already done): {skipped}")
    print(f"Individual files saved to: {args.save_path}")

    if errors:
        error_file = "semantic_description_errors.txt"
        with open(error_file, "w") as f:
            f.write("\n".join(errors))
        print(f"Errors saved to {error_file}")


if __name__ == "__main__":
    main()
