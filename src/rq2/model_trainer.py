import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import os
import evaluate
from tqdm import tqdm

def print_gpu_memory():
    """Print GPU memory usage if available"""
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

class ModelTrainer:
    def __init__(self, model_name: str, num_labels: int = 2, device: str = None):
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Handle device initialization
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA device")
            else:
                self.device = torch.device("cpu")
                print("Using CPU device")
        else:
            self.device = torch.device(device)
            print(f"Using specified device: {device}")
        
        print(f"Device being used: {self.device}")
        
        # Initialize model and tokenizer
        print("Loading model and tokenizer...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize metrics
        self.train_accuracy = evaluate.load('accuracy')
        self.val_accuracy = evaluate.load('accuracy')
        
        print("Model and tokenizer loaded successfully")
    
    def create_dataloader(self, encodings: Dict, labels: np.ndarray, 
                         batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader from tokenized data."""
        print("Creating dataloader...")
        dataset = TensorDataset(
            torch.tensor(encodings['input_ids']),
            torch.tensor(encodings['attention_mask']),
            torch.tensor(labels)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def evaluate_model(self, dataloader: DataLoader) -> Dict:
        """Evaluate the model on a given dataset."""
        self.model.eval()
        self.val_accuracy = evaluate.load('accuracy')
        
        print("Starting evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                predictions = torch.argmax(outputs.logits, dim=1)
                self.val_accuracy.add_batch(predictions=predictions, references=labels)
        
        return self.val_accuracy.compute()
    
    def train(self, train_encodings: Dict, train_labels: np.ndarray,
              val_encodings: Dict, val_labels: np.ndarray,
              num_epochs: int = 3, batch_size: int = 32,
              learning_rate: float = 2e-5, warmup_steps: int = 0) -> Dict:
        """Train the model and return training history."""
        print("Initializing training...")
        train_loader = self.create_dataloader(train_encodings, train_labels, batch_size)
        val_loader = self.create_dataloader(val_encodings, val_labels, batch_size, shuffle=False)
        
        print("Setting up optimizer and scheduler...")
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Total training steps: {total_steps}")
        print(f"Batch size: {batch_size}")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            self.train_accuracy = evaluate.load('accuracy')
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                self.train_accuracy.add_batch(predictions=predictions, references=labels)
            
            avg_train_loss = total_loss / len(train_loader)
            train_metrics = self.train_accuracy.compute()
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            print(f"\nEpoch {epoch + 1} completed")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Print GPU memory usage if available
            print_gpu_memory()
        
        return history
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Make predictions on new text data."""
        print("Starting prediction...")
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_preds = []
        
        print(f"\nMaking predictions on {len(texts)} samples")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Making predictions"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        return np.array(all_preds)
    
    def save_model(self, output_dir: str):
        """Save the trained model and tokenizer."""
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved successfully to {output_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str, num_labels: int = 2, device: str = None):
        """Load a saved model."""
        instance = cls(model_dir, num_labels, device)
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=num_labels
        ).to(instance.device)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return instance 