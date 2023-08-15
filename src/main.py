from sklearn.model_selection import train_test_split
from datasets import load_dataset
import time

from utils.preprocess import DistilBERTPreprocess
from utils.fine_tune import DistilBERTFineTune
from utils.test import DistilBERTTest

# Load and preprocess dataset
dataset = load_dataset("PolyAI/banking77")
train_texts, test_texts = train_test_split(dataset["train"]["text"], test_size=0.2, random_state=42)
train_labels, test_labels = train_test_split(dataset["train"]["label"], test_size=0.2, random_state=42)

# Tokenize and preprocess dataset
dbPreprocess = DistilBERTPreprocess("distilbert-base-uncased")
train_dataset, test_dataset, tokenizer = dbPreprocess.tokenize(train_texts, test_texts, train_labels, test_labels)

# Fine tune model
ft = DistilBERTFineTune(train_dataset, batch_size = 8, numEpochs = 2)
start = time.time()
model = ft.train()
end = time.time()
trainingTime = end - start
print(f"Training took {trainingTime:.2f} seconds")

# Save the fine-tuned model
model.save_pretrained("model")
tokenizer.save_pretrained("model")

# Evaluate the model
eval = DistilBERTTest(model)
accuracy = eval.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.2%}")