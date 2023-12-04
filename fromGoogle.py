import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder #igor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from tensorboard.plugins.hparams import api as hp
import optuna
import sys

# ADJUST SETTINGS TO RUN THE PROGRAM IN PREFFERED MODE:
# [comment out unwanted]

# CHOSE WHETHER THE MODEL SHOULD BE LOADED FROM A FILE (BE SURE A FILE IS PROVIDED AND MATCHES THE DATASET)
LOAD_MODE = [
    # 'LOAD',
    'NOLOAD'
]

# CHOOSE WHETHER THE MODEL SHOULD BE SAVED AFTER PROGRAM IS FINISHED
SAVE_MODE = [
    'SAVE',
    # 'NOSAVE'
]

# CHOOSE MODE OF OPERATION OF THE PROGRAM
RUN_MODE = [
    'PARAMETER_EXPERIMENTS',  # Give a list of parameters and run few epochs to test which settings are best
    # 'TRAIN_AND_EVAL',           # Classic training and evaluating without specifying parameters
    # 'PREDICT_FOR_INPUT'       # Prediction of class for given INPUT_TEXT
]

# Adjust number of epochs
NUM_EPOCH = 4

# Adjust Base Run Parameters
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Prepare Hyperparameters for optimization experiments
HP_DROPOUT_MIN = 0.1
HP_DROPOUT_MAX = 0.5 # zwiekszyc do 0.5
# HP_BATCH_SIZE = [16, 32, 64, 128, 256] # nie szukac
HP_LEARNING_RATE_MIN = 2e-5 # Po przedziałach
HP_LEARNING_RATE_MAX = 0.1 # Po przedziałach

# dodać weight decay

# Provide Input for prediction mode
INPUT_TEXT = 'Your sample TED Talk description goes here. Technologi IT Computers'

# ============================================================================================================

# The program
writer = SummaryWriter('./model_logs', )

# This is a custom dataset class that helps organize movie reviews and their sentiments for our BERT model.
# It takes care of tokenizing the text, handling the sequence length, and providing a neat package with input IDs, attention masks, and labels for our model to learn from.
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label, dtype=torch.long)}

# Our BERTClassifier takes in some input IDs and attention masks, and runs them through BERT and the extra layers we added.
# The classifier returns our output as class scores.
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits

# Main
if __name__ == '__main__':
    def load_data(data_file):
        df = pd.read_csv(data_file)
        texts = df['description'].tolist()
        labels = df['topic']
        # Use LabelEncoder to map labels to integers
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        return texts, labels

    data_file = "./test.csv"
    texts, labels = load_data(data_file)

    global_step = 0 # Initialize global_step used in logging

    def train(model, data_loader, optimizer, scheduler, device):
        global global_step
        model.train() # set model into training mode
        for batch in data_loader:
            optimizer.zero_grad() # reset gradients
            input_ids = batch['input_ids'].to(device) # send id to computing device
            attention_mask = batch['attention_mask'].to(device) # send masks to computing device
            labels = batch['label'].to(device) # set labels to computing device
            outputs = model(input_ids=input_ids, attention_mask=attention_mask) # feed the data to the model
            loss = nn.CrossEntropyLoss()(outputs, labels) # calculate loss function
            loss.backward() # propagate
            optimizer.step()
            scheduler.step()
            # Log training loss
            global_step += 1
            writer.add_scalar('Train/Loss', loss.item(), global_step)

    def evaluate(model, data_loader, device):
        model.eval() # set model to evaluation mode
        total_val_loss = 0.0 # used in logging
        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device) # send id to computing device
                attention_mask = batch['attention_mask'].to(device) # send masks to computing device
                labels = batch['label'].to(device) # send lables to computing device
                outputs = model(input_ids=input_ids, attention_mask=attention_mask) # feed data to model
                _, preds = torch.max(outputs, dim=1) # calculate predictions
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
                # Calculate loss
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_val_loss += loss.item()
        # Log validation metrics
        writer.add_scalar('Validation/Average Loss', total_val_loss / len(val_loader), global_step)
        writer.add_scalar('Validation/Accuracy', accuracy_score(actual_labels, predictions), global_step)
        writer.add_scalar('Validation/Precision', precision_score(actual_labels, predictions, average='weighted'), global_step)
        writer.add_scalar('Validation/Recall', recall_score(actual_labels, predictions, average='weighted'), global_step)
        writer.add_scalar('Validation/F1', f1_score(actual_labels, predictions, average='weighted'), global_step)
        # return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions) # calculate scores
        return accuracy_score(actual_labels, predictions), total_val_loss / len(val_loader) # calculate scores

    def predict(text, model, tokenizer, device, max_length=128):
        model.eval() # set model into evaluation mode
        encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True) # encode text to be predicted with tokenizer
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask) # feed data to the model
            _, preds = torch.max(outputs, dim=1) # calculate predictions
        return preds.item() # return predicted class

    def create_and_train_model_for_experiments(trial, train_loader, val_loader, writer):
        # Sample hyperparameters from the trial
        dropout = trial.suggest_float('dropout', HP_DROPOUT_MIN, HP_DROPOUT_MAX)
        learning_rate = trial.suggest_float('learning_rate', HP_LEARNING_RATE_MIN, HP_LEARNING_RATE_MAX)
        #Init:
        model = BERTClassifier(bert_model_name, num_classes, dropout).to(device)

        #Load:
        if LOAD_MODE[0] == 'LOAD':
            model.load_state_dict(torch.load("./bert_ted_model.pth"))

        # Set up Optimizer, and Scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Training and evaluating the model
        for epoch in range(num_epochs):
            train(model, train_loader, optimizer, scheduler, device)
            _, validation_loss = evaluate(model, val_loader, device)

        # Log hyperparameters and validation loss to TensorBoard
        writer.add_scalar('Experiment/dropout', dropout)
        # writer.add_scalar('Experiment/batch_size', batch_size)
        writer.add_scalar('Experiment/learning_rate', learning_rate)
        writer.add_scalar('Experiment/validation_loss', validation_loss)
        return validation_loss  # Return the validation loss for optimization

    def objective(trial):
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # Fixed batch size for validation

        # Create TensorBoard SummaryWriter for each trial
        writer2 = SummaryWriter(log_dir=f'model_logs/trial_{trial.number}')

        # Train the model for optimization
        validation_loss = create_and_train_model_for_experiments(trial, train_loader, val_loader, writer2)

        # Close the TensorBoard SummaryWriter
        writer2.close()

        return validation_loss

    # Set up parameters
    bert_model_name = BERT_MODEL_NAME
    num_classes = len(np.unique(labels))
    max_length = MAX_LENGTH
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCH
    learning_rate = LEARNING_RATE

    # Loading and splitting the data.
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Tokenizing and Preparing DataLoader
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize or Load Model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Init:
    model = BERTClassifier(bert_model_name, num_classes, 0.1).to(device)

    #Load:
    if LOAD_MODE[0] == 'LOAD':
        model.load_state_dict(torch.load("./bert_ted_model.pth"))

    # Set up Optimizer, and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Adding model graph to the TensorBoard
    # Dummy input on CPU
    dummy_input_ids = torch.randint(0, 100, (1, max_length))
    dummy_attention_mask = torch.randint(0, 2, (1, max_length))

    # Move dummy input to GPU
    dummy_input_ids = dummy_input_ids.to('cuda:0')
    dummy_attention_mask = dummy_attention_mask.to('cuda:0')

    # Write out model graph
    writer.add_graph(model, (dummy_input_ids, dummy_attention_mask))
    writer.close()

    # RUNNING THE PROGRAM BASED ON RUN_MODE SETTING
    if RUN_MODE[0] == 'TRAIN_AND_EVAL':
        # Training and evaluating the model
        for epoch in range(num_epochs):
            train(model, train_loader, optimizer, scheduler, device)
            accuracy, _ = evaluate(model, val_loader, device)
            print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy}')
            # Log hyperparameters to TensorBoard
            writer.add_scalar('Normal_Run/dropout', 0.1)
            writer.add_scalar('Normal_Run/batch_size', batch_size)
            writer.add_scalar('Normal_Run/learning_rate', learning_rate)
    elif RUN_MODE[0] == 'PREDICT_FOR_INPUT':
        # Load model
        model.load_state_dict(torch.load("./bert_ted_model.pth"))
        # Evaluating model’s performance by predicting on a sample text
        df = pd.read_csv(data_file)
        encoded_to_class_mapping = dict(zip(labels, df['topic']))
        sample_text = "Your sample TED Talk description goes here. Technologi IT Computers" # Example prediction
        predicted_class = predict(INPUT_TEXT, model, tokenizer, device, max_length)
        original_class_name = encoded_to_class_mapping.get(predicted_class, f'Unknown label: {predicted_class}')
        print(f'Predicted Class: {original_class_name}')
    elif RUN_MODE[0] == 'PARAMETER_EXPERIMENTS':
        # Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        # Print the best hyperparameters
        print("Best trial:")
        print("  Value: ", study.best_trial.value)
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

    # SAVING MODEL DEPENDING ON SAVE_MODE SETTING
    if SAVE_MODE[0] == 'SAVE':
        # Saving the final model
        torch.save(model.state_dict(), './bert_ted_model.pth')