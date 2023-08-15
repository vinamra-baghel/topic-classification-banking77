import torch
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset

class DistilBERTPreprocess:
    """
    All functions related to tokenization and preprocessing
    """
    def __init__(self, model_name = "distilbert-base-uncased"):
        # Load pre-trained DistilBERT model and tokenizer
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

    def tokenize(self, train_texts, test_texts, train_labels, test_labels):
        # Tokenize and preprocess input data
        max_length = 128
        train_encoding = self.tokenizer(train_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        test_encoding = self.tokenizer(test_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        train_input_ids = train_encoding["input_ids"]
        train_attention_mask = train_encoding["attention_mask"]
        test_input_ids = test_encoding["input_ids"]
        test_attention_mask = test_encoding["attention_mask"]

        train_dataset = TensorDataset(train_input_ids, train_attention_mask, torch.tensor(train_labels))
        test_dataset = TensorDataset(test_input_ids, test_attention_mask, torch.tensor(test_labels))

        return train_dataset, test_dataset, self.tokenizer