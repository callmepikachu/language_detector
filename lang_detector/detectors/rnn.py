import torch
import torch.nn as nn
from lang_detector.detectors.base import BaseLanguageDetector
from lang_detector.schemas import LanguageDetectionRequest, LanguageDetectionResponse
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = [self.char_to_idx.get(c, 0) for c in text[:self.max_len]]
        encoded += [0] * (self.max_len - len(encoded))

        return torch.tensor(encoded), torch.tensor(label)


class RNNLanguageClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNLanguageClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))


class RNNLanguageDetector(BaseLanguageDetector):
    def __init__(self, num_epochs=10, batch_size=64, lr=0.001, max_len=100):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_len = max_len

        self.label_encoder = LabelEncoder()
        self.char_to_idx = {}
        self.model = None
        self.classes_ = []

    def fit(self, texts, labels):
        # Build character vocabulary
        all_chars = set(''.join(texts))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(all_chars)}  # 0 is padding

        y = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)

        dataset = CharDataset(texts, y, self.char_to_idx, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # set hyperparameters
        vocab_size = len(self.char_to_idx) + 1  # +1 for padding
        embedding_dim = 64
        hidden_dim = 128
        output_dim = len(self.classes_)

        self.model = RNNLanguageClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for epoch in range(self.num_epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item():.4f}")

    def detect(self, request: LanguageDetectionRequest) -> LanguageDetectionResponse:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Preprocess
        encoded = [self.char_to_idx.get(c, 0) for c in request.text[:self.max_len]]
        encoded += [0] * (self.max_len - len(encoded))
        X = torch.tensor([encoded]).to(device)

        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0][pred_idx].item()
            lang = self.label_encoder.inverse_transform([pred_idx])[0]

        return LanguageDetectionResponse(
            language=lang,
            confidence=confidence,
            possible_languages=[
                self.label_encoder.inverse_transform([i])[0] for i in torch.argsort(probs[0], descending=True).tolist()
            ]
        )