import os
import random
from collections import OrderedDict

import torch
import torch.distributed as dist
from transformers import AutoConfig, AdamW

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_from_disk, Dataset, concatenate_datasets
import numpy as np
import math
from params_and_config import *
from sharded_text_dataset_class import ShardedDataset
from transformers import TrainerCallback
import matplotlib.pyplot as plt

print(f'hello this is the right script multilingual-e5-large')
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

n_gpus = torch.cuda.device_count()

if torch.cuda.is_available():
    print(f"Num of Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")

# Seed setting
np.random.seed(42)
torch.manual_seed(42)

# SLURM and sharding configurations
SLURM = True
SHARDED = True

# Model setup
model_name = 'intfloat/multilingual-e5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
torch.autograd.set_detect_anomaly(True)
config = AutoConfig.from_pretrained(model_name)
print(config)

for param in model.parameters():
    assert param.requires_grad, "All model parameters should have requires_grad=True"

# Output directory and training flags
model_output_dir = "/model"
FIRST_TRAINING = True
CREATE_NEW_DATASET = False

# Sharded dataset directories
txt_shard_directories = [os.path.join(knesset_txt_files_path, "plenary_text_shards"), os.path.join(knesset_txt_files_path, "committee_text_shards")]
dataset_path = "/tokenized_dataset"
train_data_dir = os.path.join(dataset_path, 'train_data')
val_data_dir = os.path.join(dataset_path, 'val_data')
test_data_dir = os.path.join(dataset_path, 'test_data')

def setup_distributed_environment():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

   # torch.distributed.init_process_group(backend='nccl')

# Function to format texts for the multilingual model
def process_input_texts_for_multi_model(input_texts_list):
    new_inputs = [f'query: {text}' for text in input_texts_list]
    return new_inputs

# Tokenize function with processed inputs
def tokenize_function(examples):
    processed_texts = process_input_texts_for_multi_model(examples["text"])
    result = tokenizer(processed_texts, truncation=True, padding="max_length", max_length=512)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Group texts function
def group_texts(examples, chunk_size, pad_token_id):
    concatenated_tokens = sum(examples['input_ids'], [])
    chunked_tokens = []
    current_chunk = []
    for token in concatenated_tokens:
        current_chunk.append(token)
        if len(current_chunk) == chunk_size:
            chunked_tokens.append(current_chunk)
            current_chunk = []
    if current_chunk:
        current_chunk.extend([pad_token_id] * (chunk_size - len(current_chunk)))
        chunked_tokens.append(current_chunk)
    return {'input_ids': chunked_tokens, 'labels': chunked_tokens.copy()}

chunk_size = 256
batch_size_per_device = 8 if SLURM else 64
gradient_accumulation_steps = 4

if SLURM:
    setup_distributed_environment()

# Processing and saving each shard
def process_and_save_shard(file_path, train_dir, val_dir, test_dir, batch_size=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    processed_lines = process_input_texts_for_multi_model(lines)
    input_ids_batches = []
    labels_batches = []
    for i in range(0, len(processed_lines), batch_size):
        batch_lines = processed_lines[i:i + batch_size]
        tokenized_batch = tokenizer(batch_lines, truncation=True, padding=True, max_length=512)
        grouped_batch = group_texts(tokenized_batch, chunk_size, pad_token_id=tokenizer.pad_token_id)
        input_ids_batches.extend(grouped_batch["input_ids"])
        labels_batches.extend(grouped_batch["labels"])
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

# Data collator, logging steps, and training arguments setup
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
logging_steps = len(train_dataset) // batch_size_per_device
eval_steps = round(0.02 * logging_steps)
print(f'Logging steps: {logging_steps}')
print(f'Eval step size: {eval_steps}')
print(f'Train dataset length: {len(train_dataset)}')

class NaNDetectionCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected at step {state.global_step}")
                param.grad = None
                break




early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.01,
)

overwrite_output_dir = True
if not FIRST_TRAINING:
    checkpoint_path = os.path.join(model_output_dir, "checkpoint-11144")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    overwrite_output_dir = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"Model device: {next(model.parameters()).device}")
# Collecting parameters into named groups
no_decay = ["bias", "LayerNorm.weight"]  # layers without decay

decay_params = set()
no_decay_params = set()
lm_head_params = set()

for n, p in model.named_parameters():
    if "lm_head" in n:
        lm_head_params.add(p)
    elif any(nd in n for nd in no_decay_params):
        no_decay_params.add(p)
    else:
        decay_params.add(p)

# Ensuring no overlap
lm_head_params = lm_head_params - no_decay_params - decay_params
no_decay_params = no_decay_params - decay_params

# Defining optimizer parameters groups
optimizer_grouped_parameters = [
    {"params": list(decay_params), "weight_decay": 0.01, "lr": 1e-6},
    {"params": list(no_decay_params), "weight_decay": 0.0, "lr": 1e-6},
    {"params": list(lm_head_params), "weight_decay": 0.01, "lr": 1e-4}
]
training_args = TrainingArguments(
    output_dir=model_output_dir,
    logging_dir=os.path.join("/app","logs"),
    overwrite_output_dir=overwrite_output_dir,
    evaluation_strategy="steps",
    gradient_accumulation_steps=gradient_accumulation_steps,
    ddp_find_unused_parameters=False,
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size_per_device,
    per_device_eval_batch_size=batch_size_per_device,
    push_to_hub=False,
    eval_steps=eval_steps,
    save_total_limit=10,
    save_steps=eval_steps,
    local_rank=torch.distributed.get_rank() if SLURM else -1,
    num_train_epochs=1,
    load_best_model_at_end=True,
    fp16=False,
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
    callbacks=[NaNDetectionCallback, early_stopping_callback],
    optimizers=(AdamW(optimizer_grouped_parameters), None)  # (optimizer, lr_scheduler)
)

# Training and evaluation
try:
    trainer.train()
    print("Finished training")
    eval_results = trainer.evaluate()
    print(f">>> Val Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    test_eval_results = trainer.evaluate(test_dataset)
    print(f">>> Test Perplexity: {math.exp(test_eval_results['eval_loss']):.2f}")
except Exception as e:
    print(f"Training failed with error: {e}")

# Plotting training and evaluation losses
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
    plt.savefig(os.path.join("/app", "train and eval loss"))
else:
    plt.show()

# Save the fine-tuned model
try:
    model.save_pretrained(os.path.join(model_output_dir, "final_model"))
except Exception as e:
    print(f"Error saving the model: {e}")
