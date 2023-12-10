import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

def mask_tokens(inputs, tokenizer, mlm_probability=0.5):
    """
    Prepare masked tokens inputs/labels for T5: 15% random spans
    """
    # Create a mask array
    labels = inputs.clone()
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Replace masked input tokens with a sentinel token (for T5)
    sentinel_token_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    inputs[masked_indices] = sentinel_token_id

    return inputs, labels


def get_random_sentence_from_different_label(current_label, grouped_sentences):
    possible_labels = list(grouped_sentences.keys())
    possible_labels.remove(current_label)  # Remove current label
    random_label = random.choice(possible_labels)  # Choose a different label
    return random.choice(grouped_sentences[random_label])

def initial_dataloader(df, tokenizer, max_length):
    # Assuming your columns are named 'sentence' and 'label'
    df['input_text'] = df['sentence']
    
    # create target text that is a random sentence different label of input text 
    grouped_sentences = df.groupby('label')['input_text'].apply(list).to_dict()
    df['target_text'] = df.apply(lambda row: get_random_sentence_from_different_label(row['label'], grouped_sentences), axis=1)
    # invert the label
    df['transform_style'] = df['label'].apply(lambda x: 1 if x == 0 else 0)

    # Split into training and validation sets

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df, test_df = train_test_split(train_df, test_size=0.01)

    def tokenize_and_mask(batch, tokenizer, max_length):
        tokenized_input = tokenizer(batch['input_text'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        masked_input, labels = mask_tokens(tokenized_input['input_ids'], tokenizer)
        return {
            'input_ids': masked_input,
            'labels': labels,
            'attention_mask': tokenized_input['attention_mask']
        }

    train_data = train_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)
    val_data = val_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)
    test_data = test_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)

    class T5Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data.iloc[idx]
            return {
                'input_ids': torch.tensor(item['input_ids'], dtype=torch.long).squeeze(0),
                'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long).squeeze(0),
                'labels': torch.tensor(item['labels'], dtype=torch.long).squeeze(0)
        }

    train_dataset = T5Dataset(train_data)
    val_dataset = T5Dataset(val_data)
    test_dataset = T5Dataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_df, val_df,test_df, train_dataset,val_dataset, test_dataset,train_loader, val_loader, test_loader


def initial_dataloader_reconstruct(df, tokenizer, max_length):
    # Assuming your columns are named 'sentence' and 'label'
    df['input_text'] = df['sentence']
    
    # create target text that is a random sentence different label of input text 
    grouped_sentences = df.groupby('label')['input_text'].apply(list).to_dict()
    df['target_text'] = df.apply(lambda row: get_random_sentence_from_different_label(row['label'], grouped_sentences), axis=1)
    # invert the label
    df['transform_style'] = df['label'].apply(lambda x: 1 if x == 0 else 0)

    # Split into training and validation sets

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df, test_df = train_test_split(train_df, test_size=0.01)
        
    def tokenize_and_mask(batch, tokenizer, max_length):
        tokenized_input = tokenizer(batch['input_text'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        return {
            'input_ids': tokenized_input['input_ids'].squeeze(0),
            'labels': tokenized_input['input_ids'].squeeze(0),
            'attention_mask': tokenized_input['attention_mask'].squeeze(0)
        }


    train_data = train_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)
    val_data = val_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)
    test_data = test_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)

    class T5Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data.iloc[idx]
            return {
                'input_ids': torch.tensor(item['input_ids'], dtype=torch.long).squeeze(0),
                'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long).squeeze(0),
                'labels': torch.tensor(item['labels'], dtype=torch.long).squeeze(0)
        }

    train_dataset = T5Dataset(train_data)
    val_dataset = T5Dataset(val_data)
    test_dataset = T5Dataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_df, val_df,test_df, train_loader, val_loader, test_loader

def initial_dataloader_vector_slicing(df, tokenizer, max_length,batch_size):
        # Assuming your columns are named 'sentence' and 'label'
    df['input_text'] = df['sentence']
    
    # create target text that is a random sentence different label of input text 
    grouped_sentences = df.groupby('label')['input_text'].apply(list).to_dict()
    df['target_text'] = df.apply(lambda row: get_random_sentence_from_different_label(row['label'], grouped_sentences), axis=1)
    # invert the label
    df['transform_style'] = df['label'].apply(lambda x: 1 if x == 0 else 0)

    # Split into training and validation sets

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df, test_df = train_test_split(train_df, test_size=0.01)
        
    def tokenize_and_mask(batch, tokenizer, max_length):
        tokenized_input_input_text1 = tokenizer(batch['input_text'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        tokenized_input_input_text2 = tokenizer(batch['target_text'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        return {
            'input_ids1': tokenized_input_input_text1['input_ids'].squeeze(0),
            'input_ids2': tokenized_input_input_text2['input_ids'].squeeze(0),
            'labels1': tokenized_input_input_text1['input_ids'].squeeze(0),
            'labels2': tokenized_input_input_text2['input_ids'].squeeze(0),
            'attention_mask1': tokenized_input_input_text1['attention_mask'].squeeze(0),
            'attention_mask2': tokenized_input_input_text2['attention_mask'].squeeze(0),
            'sentence1_style': batch['label'],
            'sentence2_style': batch['transform_style']
        }


    train_data = train_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)
    val_data = val_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)
    test_data = test_df.apply(lambda x: tokenize_and_mask(x, tokenizer, max_length), axis=1)

    class T5Dataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data.iloc[idx]
            return {
                'input_ids1': torch.tensor(item['input_ids1'], dtype=torch.long).squeeze(0),
                'input_ids2': torch.tensor(item['input_ids2'], dtype=torch.long).squeeze(0),
                'attention_mask1': torch.tensor(item['attention_mask1'], dtype=torch.long).squeeze(0),
                'attention_mask2': torch.tensor(item['attention_mask2'], dtype=torch.long).squeeze(0),
                'labels1': torch.tensor(item['labels1'], dtype=torch.long).squeeze(0),
                'labels2': torch.tensor(item['labels2'], dtype=torch.long).squeeze(0),
                'sentence1_style': torch.tensor(item['sentence1_style'], dtype=torch.long).squeeze(0),
                'sentence2_style': torch.tensor(item['sentence2_style'], dtype=torch.long).squeeze(0)
        }

    train_dataset = T5Dataset(train_data)
    val_dataset = T5Dataset(val_data)
    test_dataset = T5Dataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return  train_df, val_df,test_df, train_dataset,val_dataset, test_dataset,train_loader, val_loader, test_loader