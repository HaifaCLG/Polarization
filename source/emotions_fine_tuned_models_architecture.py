import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AdamW
from tqdm import tqdm
class EmotionDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return sentence, torch.tensor(label)

    def __len__(self):
        return len(self.labels)



class EmotionClassifier(nn.Module):
    def __init__(self, sentence_transformer_model, num_labels):
        super(EmotionClassifier, self).__init__()
        self.sentence_transformer_model = sentence_transformer_model
        self.classifier = nn.Linear(self.sentence_transformer_model.get_sentence_embedding_dimension(), num_labels)

    def forward(self, sentences):
        embeddings = self.sentence_transformer_model.encode(sentences, convert_to_tensor=True)
        logits = self.classifier(embeddings)
        return logits

# Function to create DataLoader
def create_data_loader(sentences, labels, batch_size=16):
    dataset = EmotionDataset(sentences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, sentences, labels, num_epochs=6):
    data_loader = create_data_loader(sentences, labels)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for epoch in range(num_epochs):  # Number of epochs
        model.train()
        train_loss = 0
        for batch in tqdm(data_loader):
            batch_sentences, batch_labels = batch
            batch_sentences = list(batch_sentences)  # Convert from tuple to list
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_sentences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(data_loader)}")

    return model