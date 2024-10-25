import os
import json
import random
import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
import wandb
from tqdm import tqdm

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune a model on MMLU auxiliary set")
parser.add_argument("--mask_prompt", action="store_true", help="Mask out the prompt from loss calculation")
parser.add_argument("--wandb_project", type=str, default="mmlu-finetuning", help="wandb project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
args = parser.parse_args()


def mcq_to_text(answers: list, labels: list, correct_label: str):
    labels_to_answers = {lab: ans for ans, lab in zip(answers, labels)}
    return labels_to_answers[correct_label]

def make_mcq_string(choices):
    alpha_order = string.ascii_uppercase
    choices_str = "\n".join([f"{alpha_order[i]}. {choice}" for i, choice in enumerate(choices)])
    return choices_str

# Initialize wandb
wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

# Load the model and tokenizer
load_llama = True

if load_llama:
    MODEL_NAME = "EleutherAI/pythia-1b"  # Replace with your model
    TOKENIZER_NAME = "EleutherAI/pythia-1b"  # Replace with your tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
else:
    MODEL_NAME = "./fw_checkpoint_hf/pytorch_model.bin"  # Replace with your model
    TOKENIZER_NAME = "./fw_checkpoint_hf/tokenizer.model"  # Replace with your tokenizer
    state_dict = torch.load(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("./fw_checkpoint_hf", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("./fw_checkpoint_hf", local_files_only=True, state_dict=state_dict)

# Load MMLU auxiliary dataset
dataset_dict = load_dataset("cais/mmlu", "all")
dataset_dict["train"] = dataset_dict["auxiliary_train"]
del dataset_dict["auxiliary_train"]

def format_data(example):
    prompt = f"Question: {example['question']}\nA: {example['choices'][0]}\nB: {example['choices'][1]}\nC: {example['choices'][2]}\nD: {example['choices'][3]}\nAnswer:"
    target = f" {example['answer']}\n"
    
    # Combine prompt and target
    full_text = prompt + target
    
    return tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=2048)


# Preprocess the dataset
formatted_train = dataset_dict["train"].map(format_data, remove_columns=dataset_dict["train"].column_names)
formatted_validation = dataset_dict["validation"].map(format_data, remove_columns=dataset_dict["validation"].column_names)
formatted_test = dataset_dict["test"].map(format_data, remove_columns=dataset_dict["test"].column_names)

tokenized_dataset = DatasetDict({
    "train": formatted_train,
    "validation": formatted_validation,
    "test": formatted_test
})
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size of 32
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=20,
    bf16=True,
    bf16_full_eval=True,
    report_to="wandb", 
)

training_args.set_lr_scheduler(name="cosine", warmup_ratio=0.03)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Start training
trainer.train()

# Save the model
model_save_path = f"./mmlu_finetuned_model_masked_custom_{pytorch_model}" if args.mask_prompt else f"./mmlu_finetuned_model_unmasked_custom_{pytorch_model}"
trainer.save_model(model_save_path)

print(f"Training complete. Model saved to {model_save_path}")

# Finish wandb run
wandb.finish()
