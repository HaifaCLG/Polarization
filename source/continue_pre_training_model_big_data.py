import json
import os
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_from_disk, Dataset, concatenate_datasets
import numpy as np
import math
from params_and_config import *
from sharded_text_dataset_class import ShardedDataset
from transformers import TrainerCallback

import torch

print(f'hello this is the right script')
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

model_output_dir = ""
n_gpus = torch.cuda.device_count()

if torch.cuda.is_available():
    print(f"Num of Available GPUs: {n_gpus}")
    print("Available GPUs:")
    print("this is here?")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("here too")
    print("No GPUs available.")




# Seed setting
np.random.seed(42)
torch.manual_seed(42)

# SLURM and sharding configurations
SLURM = True
SHARDED = True

# Model setup
model_name = 'dicta-il/dictabert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


# Enabling anomaly detection to find the operation that failed
torch.autograd.set_detect_anomaly(True)

for param in model.parameters():
    assert param.requires_grad, "All model parameters should have requires_grad=True"

# model = model.to('cuda:1')

# Output directory and training flags
# model_output_dir = "dicta-bert-knesset-finetuned"

FIRST_TRAINING = True
CREATE_NEW_DATASET = False

# Sharded dataset directories
txt_shard_directories = [os.path.join(knesset_txt_files_path, "plenary_text_shards"), os.path.join(knesset_txt_files_path, "committee_text_shards")]
dataset_path = "/tokenized_dataset"
train_data_dir = os.path.join(dataset_path,'train_data')
val_data_dir = os.path.join(dataset_path,'val_data')
test_data_dir = os.path.join(dataset_path,'test_data')




def setup_distributed_environment():
    torch.distributed.init_process_group(backend='nccl')

# Tokenize function
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Group texts function
def group_texts(examples, chunk_size, pad_token_id):
    # Flatten all tokenized sentences into a single list
    concatenated_tokens = sum(examples['input_ids'], [])

    # Split the tokens into chunks of up to 256 tokens
    chunked_tokens = []
    current_chunk = []
    for token in concatenated_tokens:
        current_chunk.append(token)
        if len(current_chunk) == chunk_size:
            chunked_tokens.append(current_chunk)
            current_chunk = []

        # Pad the last chunk if it's shorter than chunk_size
    if current_chunk:
        current_chunk.extend([pad_token_id] * (chunk_size - len(current_chunk)))  # Pad with the pad token ID
        chunked_tokens.append(current_chunk)

    # Prepare the final result
    result = {'input_ids': chunked_tokens}
    result['labels'] = result['input_ids'].copy()  # Duplicate for labels if needed
    return result


# def group_texts(examples):
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     total_length = (total_length // chunk_size) * chunk_size
#     result = {
#         k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

chunk_size = 256
if SLURM:
    batch_size = 32
else:
    batch_size = 64
print(f'batch_size is {batch_size}')

gradient_accumulation_steps = 4  # This effectively doubles the batch size without increasing memory usage

if SLURM:
    setup_distributed_environment()
# Processing and saving each shard
def process_and_save_shard(file_path, train_dir, val_dir, test_dir, batch_size=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    input_ids_batches = []
    labels_batches = []
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i + batch_size]
        tokenized_batch = tokenizer(batch_lines, truncation=False, padding=False)
        grouped_batch = group_texts(tokenized_batch, chunk_size, pad_token_id=tokenizer.pad_token_id)

        input_ids_batches.extend(grouped_batch["input_ids"])
        labels_batches.extend(grouped_batch["labels"])
    print(f'number of samples (grouped sentences of 250 tokens in shard: {len(input_ids_batches)}')
    indices = list(range(len(input_ids_batches)))
    random.shuffle(indices)

    train_end = int(len(indices) * split_ratios[0])
    val_end = train_end + int(len(indices) * split_ratios[1])

    train_input_ids = [input_ids_batches[i] for i in indices[:train_end]]
    train_labels = [labels_batches[i] for i in indices[:train_end]]
    val_input_ids = [input_ids_batches[i] for i in indices[train_end:val_end]]
    val_labels = [labels_batches[i] for i in indices[train_end:val_end]]
    test_input_ids = [input_ids_batches[i] for i in indices[val_end:]]
    test_labels = [labels_batches[i] for i in indices[val_end:]]

    Dataset.from_dict({'input_ids': train_input_ids, 'labels': train_labels}).save_to_disk(os.path.join(train_dir, os.path.basename(file_path)))
    Dataset.from_dict({'input_ids': val_input_ids, 'labels': val_labels}).save_to_disk(os.path.join(val_dir, os.path.basename(file_path)))
    Dataset.from_dict({'input_ids': test_input_ids, 'labels': test_labels}).save_to_disk(os.path.join(test_dir, os.path.basename(file_path)))



# Creating sharded dataset
def create_sharded_dataset(shard_directories, train_dir, val_dir, test_dir, batch_size=1000):
    for dir in shard_directories:
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            process_and_save_shard(file_path, train_dir, val_dir, test_dir, batch_size)

split_ratios = (0.8, 0.1, 0.1)  # Train, Validation, Test split ratios
if CREATE_NEW_DATASET:
    create_sharded_dataset(txt_shard_directories, train_data_dir, val_data_dir, test_data_dir)


# Instantiate ShardedDataset for train, validation, and test sets
train_dataset = ShardedDataset(train_data_dir)
val_dataset = ShardedDataset(val_data_dir)
test_dataset = ShardedDataset(test_data_dir)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
logging_steps = len(train_dataset) // batch_size
eval_steps=round(0.02*logging_steps)
print(f'logging steps is: {logging_steps}')
print(f'eval_stepsize: {eval_steps}')
print(f'train dataset length: {len(train_dataset)}')


class NaNDetectionCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_detected = True
                    print(f"NaN gradient detected at step {state.global_step}")
                    # Zeroing out the gradients
                    param.grad = None
                    break

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.01,
)

# Function to remove 'module.' prefix from state dict keys
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` only if it exists
        new_state_dict[name] = v
    return new_state_dict


# Trainer configuration
overwrite_output_dir = True
if not FIRST_TRAINING:
    checkpoint_path = os.path.join(model_output_dir, "checkpoint-11144")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    # model = model.to(device)
    overwrite_output_dir = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"Model device: {next(model.parameters()).device}")


training_args = TrainingArguments(
    #output_dir=f'./{model_output_dir}',
    output_dir=model_output_dir,
    logging_dir=os.path.join("/app","logs"),
    overwrite_output_dir=overwrite_output_dir,
    evaluation_strategy="steps",
    gradient_accumulation_steps=gradient_accumulation_steps,
    ddp_find_unused_parameters=False,
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    eval_steps=eval_steps,
    save_total_limit=10,
    save_steps=eval_steps ,
    local_rank=torch.distributed.get_rank() if SLURM else -1,
    num_train_epochs=2,#ToDO change to 2
    load_best_model_at_end=True,
    fp16=True,
    max_grad_norm=1.0,
    save_safetensors=False
)

# Trainer instantiation
trainer = Trainer(

    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate), None),
    callbacks=[NaNDetectionCallback, early_stopping_callback]
)

original_dicta_trainer = Trainer(
    model=AutoModelForMaskedLM.from_pretrained(model_name),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizers=(torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate), None),
    callbacks=[early_stopping_callback]
)
print(f'checkpoint perplexity:')
try:
    eval_results = trainer.evaluate()
    print(f">>> Val Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    test_eval_results = trainer.evaluate(test_dataset)

    print(f">>> Test Perplexity: {math.exp(test_eval_results['eval_loss']):.2f}")

    # dicta_eval_results = original_dicta_trainer.evaluate()
    # print(f">>> original dicta bert Val Perplexity: {math.exp(dicta_eval_results['eval_loss']):.2f}")
    # dicta_test_eval_results = original_dicta_trainer.evaluate(test_dataset)
    #
    # print(f">>> original dicta bert Test Perplexity: {math.exp(dicta_test_eval_results['eval_loss']):.2f}")
except Exception as e:
    print(f"Error during evaluation: {e}")
# Training and evaluation
torch.cuda.empty_cache()
try:
    trainer.train()
    print(f'finished training', flush=True)
except Exception as e:
    if "out of memory" in str(e):
        print(f"Caught an out-of-memory error: {e}")
        # Handle or log the OOM error as needed
    else:
        print(f'training failed. error was: {e}')
    torch.cuda.empty_cache()
try:
    best_model_checkpoint = trainer.state.best_model_checkpoint
    if best_model_checkpoint:
        print(f"The best model checkpoint is: {best_model_checkpoint}")
    else:
        print("Best model checkpoint not found.")
    print(f'starting trained model evaluation', flush=True)
    eval_results = trainer.evaluate()
    print(f">>> Val Perplexity: {math.exp(eval_results['eval_loss']):.2f}", flush=True)
    test_eval_results = trainer.evaluate(test_dataset)
    print(f">>> Test Perplexity: {math.exp(test_eval_results['eval_loss']):.2f}", flush=True)
    print(f'finished perplexity evaluation', flush=True)
except Exception as e:
    print(f"Error during evaluation: {e}")

# Save the fine-tuned model
try:
    final_model_dir  = os.path.join(model_output_dir,"final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    training_args_dict = training_args.to_dict()
    with open(os.path.join(final_model_dir, "training_config.json"), 'w') as f:
        json.dump(training_args_dict, f, indent=4)
except Exception as e:
    print(f'error saving the model: {e}')

# Plotting script
import matplotlib.pyplot as plt

history = trainer.state.log_history
train_loss = [h['loss'] for h in history if 'loss' in h]
eval_loss = [h['eval_loss'] for h in history if 'eval_loss' in h]

plt.figure(figsize=(10, 8))
plt.plot(train_loss, label='Training Loss')
plt.plot(np.linspace(0, len(train_loss), len(eval_loss)), eval_loss, label='Evaluation Loss')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Loss')
if SLURM:
    plt.savefig(os.path.join("/app","train and eval loss"))
else:
    plt.show()
