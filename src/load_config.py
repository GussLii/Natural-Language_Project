
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

"""
This function will load default model and tokenizer    
"""
def load_config(max_length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained T5 model and tokenizer
    model_name = 't5-small'
    model = T5ForConditionalGeneration.from_pretrained(model_name, max_length = max_length)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    return model, tokenizer, device