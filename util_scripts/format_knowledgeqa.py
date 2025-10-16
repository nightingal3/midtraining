import json
from pathlib import Path
import os
from tqdm import tqdm
import random 
import multiprocessing
from itertools import islice

KNOWLEDGEQA_PATH = "/projects/bfcu/mliu7/all_in_one_pretrainingknowledge_data"
OUT_DIR = "/data/tir/projects/tir3/users/mengyan3/manifold_data/knowledgeqa_formatted_revised"
SUBDIRS = [
    "code",
    "dialogue",
    "openhermes",
    "triviaqa",
    "ultrachat"
]

# adapted from minicpm
def transform_chat(data, num_sample: int):
    instruction = ""
    input_text = ""
    output = data["response"]

    if data.get("from") == "dialogue_eng":
        assert num_sample == 0, "In context not allowed"
        del data["candidates"]

        prompt_style = random.randint(0, 1)
        if prompt_style == 0:
            instruction = "Assuming you are talking to your partner, please respond based on the following information and context:"
            if data["persona"]:
                input_text += "Your personal information:\n" + "\n".join(data["persona"]) + "\n\n"
            if data["context"]:
                input_text += "Context:\n" + data["context"] + "\n\n"
            input_text += "Chat history:\n"
            my_id = len(data["history"]) & 1
            for i, msg in enumerate(data["history"]):
                speaker = "You" if (i & 1) == my_id else "Partner"
                input_text += f"{speaker}: {msg}\n"
            input_text += "You: "
        elif prompt_style == 1:
            instruction = "Write the next line of dialogue for person A based on the following information:"
            if data["persona"]:
                input_text += "Personal information of A:\n" + "\n".join(data["persona"]) + "\n\n"
            if data["context"]:
                input_text += "Context:\n" + data["context"] + "\n\n"
            input_text += "Chat history:\n"
            my_id = len(data["history"]) & 1
            for i, msg in enumerate(data["history"]):
                speaker = "A" if (i & 1) == my_id else "B"
                input_text += f"{speaker}: {msg}\n"
            input_text += "A: "

    else:
        assert num_sample == 0, "In context not allowed"
        del data["candidates"]

        if data.get("context") == "you are talking to a chat bot":
            instruction = "Provide the next user message in a conversation with a chatbot:"
            input_text = "Chat history:\n"
            my_id = len(data["history"]) & 1
            for i, msg in enumerate(data["history"]):
                speaker = "You" if (i & 1) == my_id else "Bot"
                input_text += f"{speaker}: {msg}\n"
            input_text += "You: "
        elif len(data["persona"]) == 1 and data["persona"][0] == "bot":
            instruction = "You are a chatbot. Respond politely to the user's message:"
            input_text = "Chat transcript:\n"
            my_id = len(data["history"]) & 1
            for i, msg in enumerate(data["history"]):
                speaker = "Bot" if (i & 1) == my_id else "User"
                input_text += f"{speaker}: {msg}\n"
            input_text += "Bot: "
        elif data["context"]:
            instruction = f"Write your next message in a conversation about {data['context']}:"
            input_text = "Chat history:\n"
            my_id = len(data["history"]) & 1
            for i, msg in enumerate(data["history"]):
                speaker = "You" if (i & 1) == my_id else "Partner"
                input_text += f"{speaker}: {msg}\n"
            input_text += "You: "
        else:
            instruction = "Continue the conversation with your friend:"
            input_text = "Conversation:\n"
            my_id = len(data["history"]) & 1
            for i, msg in enumerate(data["history"]):
                speaker = "You" if (i & 1) == my_id else "Friend"
                input_text += f"{speaker}: {msg}\n"
            input_text += "You: "

    return {
        "instruction": instruction,
        "input": "",
        "output": input_text.strip() + "\n" + output
    }


def transform_messages(data):
    messages = data["messages"]
    
    # Ensure we have at least a user message and an assistant response
    if len(messages) < 2 or messages[-2]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None  # Skip this entry if it doesn't meet our criteria

    instruction = "Continue the conversation based on the following chat history:"
    input_text = "Chat history:\n"
    
    # Process all messages except the last assistant response
    for msg in messages[:-1]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        input_text += f"{role}: {msg['content']}\n"
    
    input_text += "Assistant: "
    output = messages[-1]["content"]  # The last assistant message is our target output

    # Randomly choose between two prompt styles
    if random.randint(0, 1) == 0:
        instruction = "As a chatbot, respond to the user based on the following conversation:"
    
    return {
        "instruction": instruction,
        "input": "",
        "output": input_text.strip() + "\n" + output
    }


def transform_qa(data):
    question = data["question"]
    answer = data["answer"]

    # List of different instruction formats
    instructions = [
        "Answer the following question:",
        "Provide a response to this query:",
        "Please address the following question:",
        "Respond to the given question:",
        "Here's a question for you to answer:",
        ""
    ]

    # List of different input formats
    input_formats = [
        "Question: {}\nAnswer:",
        "Q: {}\nA:",
        "Query: {}\nResponse:",
        "Problem: {}\nSolution:",
        "{}\nAnswer:"
    ]

    # Randomly select an instruction and input format
    instruction = random.choice(instructions)
    input_format = random.choice(input_formats)

    input_text = input_format.format(question)

    return {
        "instruction": instruction,
        "input": "",
        "output": input_text.strip() + "\n" + answer
    }


def transform_stackexchange(data):
    question = data["original_content"]["question"]
    answer = data["original_content"]["answer"]
    title = data["original_content"]["title"]

    # List of different instruction formats
    instructions = [
        "Provide an answer to the following question:",
        "Answer this question from a forum:",
        "Provide a helpful response:",
        "Respond to this query with a helpful answer:",
        "Give an informative answer to the following question:"
        ""
    ]

    # List of different input formats, some with title and some without
    input_formats = [
        "Title: {title}\n\nQuestion: {question}\n\nAnswer:",
        "Q: {question}\n\nA:",
        "{title}\n\n{question}\n\nResponse:",
        "Question: {question}\n\nCommunity Answer:",
        "Title: {title}\n\nProblem: {question}\n\nSolution:",
        "{question}\n\nProvide an answer:"
    ]

    # Randomly select an instruction and input format
    instruction = random.choice(instructions)
    input_format = random.choice(input_formats)

    # Format the input text, handling cases where the format doesn't include a title
    if "{title}" in input_format:
        input_text = input_format.format(title=title, question=question)
    else:
        input_text = input_format.format(question=question)

    return {
        "instruction": instruction,
        "input": "",
        "output": input_text.strip() + "\n" + answer
    }

# I think having extraneous kwargs should be fine?
# TODO: put these into formats in litgpt
# Actually, nvm it's easier to do processing here, we should just use "plain" format
def transform_to_litgpt_format(json_data: dict, subdir: str) -> dict:
    if subdir == "code":
        return transform_stackexchange(json_data)
    elif subdir == "dialogue":
        return transform_chat(json_data, 0)
    elif subdir == "openhermes":
        return transform_messages(json_data)
    elif subdir == "triviaqa":
        return transform_qa(json_data)
    elif subdir == "ultrachat":
        return {
            "instruction": json_data["input"],
            "input": "",
            "output": json_data["output"],
        }

def process_chunk(chunk, subdir):
    results = []
    for line in chunk:
        data = json.loads(line)
        mapped_data = transform_to_litgpt_format(data, subdir)
        results.append(json.dumps(mapped_data))
    return results

def read_in_chunks(file, chunk_size, max_lines=None):
    lines_read = 0
    while max_lines is None or lines_read < max_lines:
        chunk = list(islice(file, chunk_size))
        if not chunk:
            break
        lines_read += len(chunk)
        yield chunk

        if max_lines and lines_read >= max_lines:
            break

def process_file(file_path: str, out_path: str, subdir: str, max_lines=None, chunk_size=10000):
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes")
    all_results = []
    
    # Ensure the output directory exists
    out_dir = os.path.dirname(out_path)
    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory created/verified: {out_dir}")
    except Exception as e:
        print(f"Error creating output directory {out_dir}: {e}")
        return

    # Check if we have write permissions in the output directory
    if not os.access(out_dir, os.W_OK):
        print(f"No write permission in directory: {out_dir}")
        return

    try:
        with open(file_path, "r") as f:
            pool = multiprocessing.Pool(processes=num_processes)
            try:
                with open(out_path, "w") as out_file:
                    print(f"Successfully opened output file: {out_path}")
                    with tqdm(desc=f"Processing {file_path}") as pbar:
                        for chunk in read_in_chunks(f, chunk_size, max_lines=max_lines):
                            results = pool.apply_async(process_chunk, (chunk, subdir))
                            all_results.extend(results.get())
                            pbar.update(len(chunk))
                        out_file.write("\n".join(all_results))
            except IOError as e:
                print(f"Error writing to output file {out_path}: {e}")
            finally:
                pool.close()
                pool.join()
                pbar.close()
    except IOError as e:
        print(f"Error reading input file {file_path}: {e}")

if __name__ == "__main__":
    for subdir in SUBDIRS:
        subdir_path = os.path.join(KNOWLEDGEQA_PATH, subdir)
        print(f"Processing {subdir_path}")
        for file in os.listdir(subdir_path):
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir_path, file)
                out_path = os.path.join(OUT_DIR, subdir, "train.jsonl")
                # delete the file if it already exists
                if os.path.exists(out_path):
                    print(f"Deleting existing file: {out_path}")
                    os.remove(out_path)
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                process_file(file_path, out_path, subdir, max_lines=None)
                print(f"Processed {file_path} and saved to {out_path}")
    
    print("All files processed!")
