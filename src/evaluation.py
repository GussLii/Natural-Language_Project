import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
import torch.nn as nn

'''
evaluation function can be used during traiuning
'''
def evaluate(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_masked = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = None, labels=labels)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Mask to focus on masked tokens only
            mask = labels != -100

            # Calculate accuracy
            total_correct += (predictions[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()

    return total_correct / total_masked if total_masked > 0 else 0

'''
Loading evaluation model
'''
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

'''
use LSTM model predict the label
'''
def predict(text, model, tokenizer, device, max_length):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].flatten().to(device)
        print(input_ids)
        outputs = model(input_ids)

        # normalize the scores to all positive and sum to 1
        outputs = torch.nn.functional.softmax(outputs, dim=0)

        return outputs
    
def evaluate_model(df, model, model_eval, tokenizer, device, max_length):
    model.eval()
    predictions, labels = [], []
    
    with torch.no_grad():
        for i in range(len(df)):
            source_sentence = df.iloc[i]['input_text']
            target_sentence = df.iloc[i]['target_text']
            input_text = f"transform the written style of {source_sentence} to the written style of {target_sentence}"
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            outputs = model.generate(input_ids=input_ids, max_length=max_length, num_beams=10, early_stopping=True)
            transformed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicition = predict(transformed_sentence, model_eval, tokenizer, device, max_length)
            # apply softmax
            predicition = torch.nn.functional.softmax(predicition, dim=0)
            # use argmax to get the label
            predicition = torch.argmax(predicition, dim=0)
            # convert to int
            predicition = predicition.item()
            predictions.append(predicition)
            labels.append(df.iloc[i]['transform_style'])

    return predictions, labels

def initial_LSTM(tokenizer, device):
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = 2 

    VOCAB_SIZE = tokenizer.vocab_size
    model_eval = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, OUTPUT_DIM).to(device)
    model_eval.load_state_dict(torch.load('./adversal_network/lstm_model.pth'))
    return model_eval

def evaluate_process(test_df, model, tokenizer, device, max_length):
    model_eval = initial_LSTM(tokenizer, device)
    predictions, labels = evaluate_model(test_df, model,model_eval,tokenizer, device, max_length)
    return predictions, labels

def show_sentence(index, test_df, tokenizer, model, device, max_length):
    model.eval()
    with torch.no_grad():
        source_sentence = test_df.iloc[index]['input_text']
        target_sentence = test_df.iloc[index]['target_text']
        input_text = f"transform the written style of {source_sentence} to the written style of {target_sentence}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        outputs = model.generate(input_ids=input_ids, max_length=max_length, num_beams=10, early_stopping=True)
        transformed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Original Sentence: " + source_sentence)
        print("Target Sentence style: " + target_sentence)
        print("Generated Sentence: " + transformed_sentence)