import torch
from torch.utils.data import DataLoader

class DistilBERTTest:
    """
    Model evaluation
    """
    def __init__(self, model):
        self.model = model

    def evaluate(self, test_dataset):
        self.model.eval()
        correct = 0
        total = 0

        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct/total