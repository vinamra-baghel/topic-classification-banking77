import torch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader

class DistilBERTFineTune:
    """
    Model fine tuning
    """
    def __init__(self, train_dataset, batch_size = 8, numEpochs = 2):
        self.batch_size = batch_size
        self.numEpochs = numEpochs
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels = 77) # Banking77 has 77 classes
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.lossFunction = torch.nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.numEpochs):
            self.model.train()
            total_loss = 0
            for i, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                labels = batch[2]
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                print(f"Batch number: {i}")
            print(f"Epoch {epoch+1}: Average Loss: {total_loss / len(self.train_dataloader)}")
        return self.model