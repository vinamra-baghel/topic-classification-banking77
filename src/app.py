import gradio as gr
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

from utils.mapping import Banking77Map

# Load the fine-tuned model
model = DistilBertForSequenceClassification.from_pretrained('./../model')
tokenizer = DistilBertTokenizer.from_pretrained('./../model')

# Load the Banking77 mapping
bankMap = Banking77Map()
mapping = bankMap.mappingDict()

# Define a function to make predictions
def predict_Topic(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    Topic = mapping[str(predicted_class)]
    return Topic

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_Topic,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your text here"),
    outputs=gr.outputs.Textbox(label="Topic Prediction"),
)

# Launch the Gradio interface
iface.launch()