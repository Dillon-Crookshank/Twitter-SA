import sys
import os
import time
import datetime
import math
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer


def main():
    # Arguments
    # The name of the file that holds the model
    model_name = 'BERT.pth'

    # The name of the directory that holds the model
    model_dir = 'models/'

    # The name of the directory that holds the data
    data_dir = 'data/'

    # The % of the dataset that will be used to train the model
    data_frac = 0.03125

    # The random state used throughout the script, to keep consistency
    state = 42

    # Will stop after this many epochs of the model failing to improve loss.
    bad_epoch_limit = 2

    # batch size of data loaders
    batch_size = 15

    # Learning rate applied to adam optimizers. Default is 2e-5.
    learning_rate = 2e-6

    # Toggle console output.
    verbose = True
    
    
    if verbose: print('### Preparing Data ###')
    
    # Create and save a preprocessed dataset if it dosen't already exist
    if not os.path.exists(os.path.join(data_dir, 's140-prepared.csv')):
        # Retrieve data
        data = pd.read_csv(os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv'))

        # Prepare data
        X = data['text'].to_list()
        y = data['target'].to_list()

        # Clean the text
        X = clean_texts(X, verbose=verbose)

        # Prepare the labels for binary classification
        y = [1 if element == 4 else 0 for element in y]

        # Save the processed data to avoid the preparing process later
        pd.DataFrame({'target': y, 'text' : X}).to_csv(os.path.join(data_dir, 's140-prepared.csv'), index=False, encoding='utf-8')


    # Read and separate the data
    data = pd.read_csv(os.path.join(data_dir, 's140-prepared.csv'))

    # Split our data into negative and positive examples, so our dataset contains an equal number of each example
    data_neg = data[data['target'] == 0]
    data_pos = data[data['target'] == 1]

    # Randomly shuffle and truncate each data frame
    data_neg = data_neg.sample(frac=data_frac, random_state=state)
    data_pos = data_pos.sample(frac=data_frac, random_state=state)

    # Combine the shuffled and truncated data frames
    data = pd.concat([data_neg, data_pos])

    #
    X = data['text'].to_list()
    y = data['target'].to_list()

    X = [str(x) for x in X]


    # Split the data into training, testing, and validation sets
    X_sub, X_test, y_sub, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    X_train, X_val, y_train, y_val = train_test_split(X_sub, y_sub, test_size=0.25, random_state=state)

    # Check the sizes of the sets
    if (len(y_train) != 30_000 or len(y_test) != 10_000 or len(y_val) != 10_000):
        print(f'ERROR :: Dataset sizes incorrect -> Train: {len(y_train)}\tTest: {len(y_test)}\tVal: {len(y_val)}')
        sys.exit(-1)

    # Create dataloaders for easy batching
    train_dataloader = create_bert_dataloader(X_train, y_train, batch_size)
    test_dataloader = create_bert_dataloader(X_test, y_test, batch_size, False)
    val_dataloader = create_bert_dataloader(X_val, y_val, batch_size, False)

    # Find best device before training
    device = find_device(verbose=verbose)

    
    # Initialize the model
    if os.path.exists(os.path.join(model_dir, model_name)):
        if verbose: print('### Loaded Existing BERT ###')
        model = load_bert(os.path.join(model_dir, model_name))
    else:
        if verbose: print('### Loaded Blank BERT ###')
        model = bert_for_binary_classification()

    # Move model to best found device
    model.to(device)


    # Create lists so we can store and plot accuracy and loss over time
    epochs = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    epoch_count = 0
    bad_epoch_count = 0
    best_val_loss = float('inf')

    # Do training epochs until the model hasn't improved for n epochs
    if verbose: print('### Begin Training ###')

    while bad_epoch_count < bad_epoch_limit:
        
        print(f'### Epoch {epoch_count+1} ###')
        train_loss, train_accuracy = bert_epoch(model, train_dataloader, device, learning_rate, verbose=verbose)

        print('### Validation ###')
        val_loss, val_accuracy = bert_benchmark(model, val_dataloader, device, verbose=verbose)

        # Save the loss and accuracy history
        epochs.append(epoch_count+1)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print the most recent values to the log file
        with open('plots/log.txt', 'a') as file:
            file.write(f'Epoch {epoch_count+1} :: Train Loss: {train_loss} :: Train Accuracy: {train_accuracy} :: Val Loss: {val_loss} :: Val Accuracy: {val_accuracy}\n')
        

        # Check if model has improved
        if val_loss <= best_val_loss:
            if verbose: print('### Loss Improved -- Saving Model ###')
            save_bert(model, os.path.join(model_dir, model_name))
            best_val_loss = val_loss
            bad_epoch_count = 0
        else:
            bad_epoch_count += 1

        epoch_count += 1
             
    if verbose: print('### Training Done ###')

    # Test the model on the test set
    if verbose: print('### Testing ###')
    test_loss, test_accuracy = bert_benchmark(model, test_dataloader, device, verbose=verbose)

    with open('plots/log.txt', 'a') as file:
        file.write(f'Test Loss: {test_loss} :: Test Accuracy: {test_accuracy}\n')


    # Plot the loss history
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    # Display and save the plot
    plt.savefig('plots/loss_curve.png')
    plt.show()
    

    # Clear the figure before making the next plot
    plt.clf()


    # Plot the accuracy history
    plt.plot(epochs, train_accuracies, label='Training Loss')
    plt.plot(epochs, val_accuracies, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    # Display and save the plot
    plt.savefig('plots/accuracy_curve.png')
    plt.show()
    



def clean_texts(texts, verbose=False):
    cleaned = []
    for index, text in enumerate(texts):
        if verbose: print(f'clean_texts :: ({index+1}/{len(texts)}) {progress_bar(index+1, len(texts), 50)}', end='\r')
        
        # Convert text to lowercase
        text = text.lower()
    
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)

        # Remove external links
        text = re.sub(r'([hH][tT]{2}[pP][sS]?:\/\/)?[\w~-]+(\.[\w~-]+)+(\/[\w~-]*)*', '', text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove punctuation and digits
        tokens = [re.sub(r'[\W\d_]+', '', token) for token in tokens]

        # Remove short/empty tokens
        tokens = [token for token in tokens if len(token) > 1]

        cleaned.append(' '.join(tokens))
    
    print()

    return cleaned


def find_device(verbose=False):
    # Check if a CUDA-enabled GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose: print("### Cuda Found :: Using GPU ###")
    
    else:
        device = torch.device("cpu")
        if verbose: print("### Cuda not found :: Using CPU ###")

    return device



# Define the model
'''
class bert_for_binary_classification(nn.Module):
    def __init__(self):
        super(bert_for_binary_classification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities.squeeze()
    
'''

class bert_for_binary_classification(nn.Module):
    def __init__(self):
        super(bert_for_binary_classification, self).__init__()
        
        # Fetch a BERT embedding
        self.bert = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        # Define a MLP to pass the pooled output of BERT through
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get pooled output from BERT embedding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        
        # Pass pooled output through MLP
        probabilities = self.mlp(x)
        
        return probabilities.squeeze()


def save_bert(bert, dir):
    torch.save(bert.state_dict(), dir)

def load_bert(dir):
    model= bert_for_binary_classification()

    model.load_state_dict(torch.load(dir))

    return model


# Function that generates dataloaders
def create_bert_dataloader(texts, labels, batch_size, shuffle=True):
    # Fetch a bert tokenizer, since the default one will suffice
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

    # Tokenize texts to pytorch tensors
    inputs = tokenizer(texts, max_length=300, padding='max_length', truncation=True, return_tensors="pt")
    
    # Convert labels to PyTorch tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Create DataLoader for the dataset
    dataset = TensorDataset(inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"], labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


def bert_epoch(model, dataloader, device, learning_rate=2e-5, verbose=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Note the time before we begin the epoch
    init_time = time.time()

    model.train()
    for batch_index, batch in enumerate(dataloader):
        # Fetch data from batch and move to the target device
        input_ids, token_type_ids, attention_mask, targets = batch
        
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        
        # Reset gradient
        optimizer.zero_grad()
        
        # Run the model
        outputs = model(input_ids, attention_mask, token_type_ids)

        # Find the loss and accumulate
        loss = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

        #Find the number of correct classifications and accumulate
        predictions = (outputs > 0.5).float()
        correct = (predictions == targets).sum().item()
        total_correct += correct

        # Back propagate
        loss.backward()
        optimizer.step()

        # Estimate remaining runtime
        curr_time = time.time()
        avg_runtime = (curr_time - init_time) / (batch_index + 1)
        remaining_batches = len(dataloader) - (batch_index + 1)
        remaining_time = avg_runtime * remaining_batches

        # Clear the line of output
        if verbose: print(' ' * 100, end='\r')
        # Print the progress, loss, and accuracy
        if verbose: print(f'Batch {batch_index+1}/{len(dataloader)} {progress_bar(batch_index+1, len(dataloader), 50)}'
                          f' - Loss: {total_loss / total_samples:.4f}'
                          f' - Accuracy: {total_correct / total_samples * 100:.4f}%'
                          f' - Remaining Time: {int(remaining_time // 60)}:{int(remaining_time % 60):02}'
                          , end='\r')
    print()

    return (total_loss/total_samples), (total_correct/total_samples)


def bert_benchmark(model, dataloader, device, verbose=False):
    criterion = nn.BCELoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Note the time before we begin the benchmark
    init_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            # Fetch data from batch and move to the target device
            input_ids, token_type_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Run the model
            outputs = model(input_ids, attention_mask, token_type_ids)

            # Find the loss and accumulate
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            
            # Find the accuracy and accumulate
            predictions = (outputs > 0.5).float()
            correct = (predictions == targets).sum().item()
            total_correct += correct

            total_samples += targets.size(0)

            # Estimate remaining runtime
            curr_time = time.time()
            avg_runtime = (curr_time - init_time) / (batch_index + 1)
            remaining_batches = len(dataloader) - (batch_index + 1)
            remaining_time = avg_runtime * remaining_batches


            # Clear the line of output
            if verbose: print(' ' * 100, end='\r')
            # Print the progress, loss, and accuracy
            if verbose: print(f'Batch {batch_index+1}/{len(dataloader)} {progress_bar(batch_index+1, len(dataloader), 50)}'
                              f' - Loss: {total_loss / total_samples:.4f}'
                              f' - Accuracy: {total_correct / total_samples * 100:.4f}%'
                              f' - Remaining Time: {int(remaining_time // 60)}:{int(remaining_time % 60):02}'
                              , end='\r')
    print()

    return (total_loss/total_samples), (total_correct/total_samples)

def progress_bar(numerator, denominator, length):
    progress_bar_fill = math.floor((numerator / denominator) * length)

    return '[' + ('=' * progress_bar_fill) + (' ' * (length - progress_bar_fill)) + ']'

if __name__ == '__main__':
    main()