from datasets import load_dataset

# Load your Hebrew text data and split it into train and validation sets
dataset = load_dataset("text", data_files={"train": "hebrew_data.txt"})
dataset = dataset.train_test_split(test_size=0.1)
train_dataset, val_dataset = dataset["train"], dataset["test"]

from transformers import AutoTokenizer

# Choose a tokenizer for Hebrew (using a pre-trained Hebrew tokenizer)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenize the Hebrew text data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4)

# Set the format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

from transformers import BertConfig, BertForMaskedLM
import torch

# Define a new BERT configuration
config = BertConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=128,
    num_attention_heads=12,
    num_hidden_layers=12,
    hidden_size=768,
    type_vocab_size=2,
    pad_token_id=tokenizer.pad_token_id,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

# Create a new BERT model for masked language modeling
model = BertForMaskedLM(config)

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=2_500,
    warmup_steps=500,
    learning_rate=5e-5,
)

# Define the data collator for masked language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()


# Save the trained model
trainer.save_model("hebrew_bert_from_scratch")

#Load the trained model and tokenizer:
from transformers import BertForMaskedLM, AutoTokenizer

model_path = "hebrew_bert_from_scratch"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

import torch

#Create a function to get the embeddings:

def get_word_embeddings(sentence, model, tokenizer):
    # Tokenize the input sentence and convert it to a tensor
    inputs = tokenizer(sentence, return_tensors="pt")

    # Run the model to get the embeddings (hidden states)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings from the last hidden layer
    embeddings = outputs.last_hidden_state

    # Remove the batch dimension
    embeddings = embeddings.squeeze(0)

    # Convert the embeddings tensor to a NumPy array
    embeddings = embeddings.detach().numpy()

    # Create a dictionary to store tokens and their corresponding embeddings
    token_embeddings = {}
    for token_id, embedding in zip(inputs["input_ids"].squeeze(0), embeddings):
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        token_embeddings[token] = embedding

    return token_embeddings
#Use the function to obtain embeddings for words in a sentence:
sentence = "בדוגמה זו נקבל את השיבוצים של המילים במשפט"
word_embeddings = get_word_embeddings(sentence, model, tokenizer)

# Print the embeddings for each token in the sentence
for token, embedding in word_embeddings.items():
    print(f"{token}: {embedding}")
###########from here with fasttext##########
import fasttext
def fasttext_word_embedding():
    # Train a FastText model on your Hebrew data
    model = fasttext.train_unsupervised('hebrew_data.txt', epoch=5, min_count=5)

    # Save the trained model to a file
    model.save_object('hebrew_fasttext.bin')
    # Load the trained FastText model
    model = fasttext.load_object('hebrew_fasttext.bin')

    # Get the word embedding for a specific word
    word = "מילה"
    embedding = model.get_word_vector(word)

    print(f"Embedding for '{word}': {embedding}")
    # Find the top 5 most similar words to a given word
    similar_words = model.get_nearest_neighbors(word, k=5)

    print(f"Top 5 most similar words to '{word}':")
    for score, neighbor in similar_words:
        print(f"{neighbor}: {score}")


def preprocess_hebrew_text(text):
    # Remove any unwanted characters (e.g., punctuation, numbers, etc.)
    cleaned_text = re.sub(r'[^\u0590-\u05FF\s]', ' ', text)

    # Remove extra whitespace and newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def read_large_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line

input_file_path = 'hebrew_data_large.txt'
output_file_path = 'hebrew_data_cleaned.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in read_large_file(input_file_path):
        cleaned_line = preprocess_hebrew_text(line)
        if cleaned_line:
            output_file.write(cleaned_line + '\n')

##fasttext online for big files###
import fasttext

def train_fasttext_online(input_file, output_file, epochs=5, min_count=5):
    model = fasttext.train_unsupervised('dummy.txt', epoch=1, min_count=min_count)
    for _ in range(epochs):
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                # Split the line into words and update the model
                words = line.strip().split()
                model.train_partial(words)
    model.save_object(output_file)

input_file_path = 'hebrew_data_cleaned.txt'
model_output_path = 'hebrew_fasttext_online.bin'

train_fasttext_online(input_file_path, model_output_path)

# Load the trained FastText model
model = fasttext.load_object('hebrew_fasttext_online.bin')

# Get the word embedding for a specific word
word = "מילה"
embedding = model.get_word_vector(word)
print(f"Embedding for '{word}': {embedding}")

# Find the top 5 most similar words to a given word
similar_words = model.get_nearest_neighbors(word, k=5)
print(f"Top 5 most similar words to '{word}':")
for score, neighbor in similar_words:
    print(f"{neighbor}: {score}")


