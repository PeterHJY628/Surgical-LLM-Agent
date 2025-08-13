# ---------------------------Import Packages------------------------------------------------------
"""
DEFT-GaLore Fine-tuning Pipeline for Surgical AI Applications

This script implements a memory-efficient fine-tuning pipeline for large language models
using DEFT (Deterministic Energy-based Fourier Transform) GaLore (Gradient Low-Rank 
Projection) optimization. The system is specifically designed for surgical LLM Agent, 
enabling efficient training of billion-parameter models with reduced memory requirements.

Key Features:
- DEFT-GaLore optimizer with FFT Energy-based gradient low-rank projection
- Support for attention layer gradient compression (q_proj, k_proj, v_proj, o_proj)
- Comprehensive training monitoring with batch and epoch timing
- Automatic best model saving based on validation loss
- Integration with Hugging Face transformers and datasets
- Reproducible training with deterministic seed management

Architecture:
- Base models: LLaMA, Qwen
- Optimization: DEFT_GaLoreAdamW with gradient projection
- Data format: CSV files with Input/Label columns for Q&A pairs
- Evaluation: Standard language modeling loss with shifted labels

Usage:
    python main.py --model_name meta-llama/Llama-3.2-3B-Instruct \
                   --train_file path/to/train.csv \
                   --val_file path/to/val.csv \
                   --rank 128 --lr 3e-7 --num_epochs 5


Date: 2025
"""
import os
import random
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import logging
import glob
import argparse
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import torch.nn as nn
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from transformers.utils.versions import require_version
from typing import Optional
from model import load_model_and_tokenizer, save_best_model, DEFT_GaLoreAdamW
from dataloader import process_qa_samples, collate_fn, preprocess_data
import time

from huggingface_hub import login


# ------------------------- Argument Parsing -------------------------
def parse_args():
    """
    Parses command-line arguments for the DEFT-GaLore fine-tuning script.
    
    This function defines and parses all configuration parameters needed for training,
    including model settings, data paths, hyperparameters, and DEFT-GaLore specific
    parameters. It provides sensible defaults for surgical AI training scenarios.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - model_name: Pre-trained model identifier
            - train_file, val_file: Training and validation data paths
            - num_epochs, batch_size, lr: Training hyperparameters
            - rank, update_proj_gap, galore_scale, proj_type: DEFT-GaLore parameters
            - save_path: Model checkpoint save location
            - max_length, max_new_tokens: Sequence generation parameters
            - HF_TOKEN: Hugging Face authentication token
    """
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with DEFT-GaLore")
    # Model and tokenizer configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Pre-trained model name or path",
    )
    # Data paths
    parser.add_argument(
        "--train_file",
        type=str,
        default="Your_Path/Train.csv",
        help="Path to the training data CSV",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="Your_Path/Test.csv",
        help="Path to the validation data CSV",
    )
    # Training hyperparameters
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-7, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for regularization")
    # DEFT-Galore parameters
    parser.add_argument("--rank", type=int, default=128, help="Rank for low-rank gradient projection")
    parser.add_argument("--update_proj_gap", type=int, default=50, help="Iterations between projection matrix updates")
    parser.add_argument("--galore_scale", type=float, default=1.0, help="Scaling factor for projected gradients")
    parser.add_argument("--proj_type", type=str, default="reverse_std", help="Projection type: std, reverse_std, left, right, full")
    # Model save path
    parser.add_argument(
        "--save_path",
        type=str,
        default="Your_path_to_save_model",
        help="Path to save the best model",
    )
    # Other parameters
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    # New parameters
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--HF_TOKEN",
        type=str,
        default="Your_HuggingFace_Token",
        help="Hugging Face token for authentication",
    )

    return parser.parse_args()


# ------------------------- Utility Functions -------------------------
def set_seed(seed: int):
    """
    Sets random seeds for reproducible training across multiple libraries.
    
    This function ensures deterministic behavior by setting seeds for:
    - Python's random module
    - NumPy random number generator  
    - PyTorch CPU random number generator
    - PyTorch CUDA random number generators (if available)
    - CUDA backend deterministic operations
    
    Args:
        seed (int): Random seed value to use across all libraries
        
    Note:
        Setting deterministic=True and benchmark=False may reduce performance
        but ensures reproducible results across training runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Define training and validation functions
def validate(model, dataloader, criterion):
    """
    Performs validation/evaluation on the provided dataset.
    
    This function runs the model in evaluation mode without gradient computation
    to assess performance on validation data. It computes the average loss
    across all batches using the standard language modeling objective with
    shifted labels for next-token prediction.
    
    Args:
        model: The neural network model to evaluate
        dataloader: DataLoader containing validation batches
        criterion: Loss function (typically CrossEntropyLoss with ignore_index=-100)
        
    Returns:
        float: Average validation loss across all batches
        
    Note:
        - Uses torch.no_grad() for memory efficiency during evaluation
        - Implements standard autoregressive language modeling loss computation
        - Shifts logits and labels by one position for next-token prediction
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(
    model, train_loader, valid_loader, optimizer, criterion, num_epochs, save_path
):
    """
    Executes the main training loop with DEFT-GaLore optimization.
    
    This function implements the core training procedure with comprehensive monitoring:
    - Trains the model using gradient projection techniques from DEFT-GaLore
    - Tracks training and validation losses per epoch
    - Monitors batch-level and epoch-level timing for performance analysis
    - Saves model checkpoints and maintains best model based on validation loss
    - Provides detailed logging for training progress and performance metrics
    
    Args:
        model: Neural network model to train (typically a language model)
        train_loader: DataLoader for training data batches
        valid_loader: DataLoader for validation data batches  
        optimizer: DEFT_GaLoreAdamW optimizer with gradient projection capabilities
        criterion: Loss function for training (CrossEntropyLoss with ignore_index=-100)
        num_epochs (int): Number of training epochs to execute
        save_path (str): Base path for saving model checkpoints
        
    Side Effects:
        - Modifies model parameters through training
        - Saves model checkpoints to disk
        - Prints and logs training progress
        - Updates global best validation loss tracking
        
    Note:
        - Implements autoregressive language modeling with shifted labels
        - Uses detailed timing analysis for performance monitoring
        - Integrates with external logging system for experiment tracking
    """
    best_val_loss = float("inf")
    print("Start Training!")
    logging.warning("Start Training!")

    total_batches = len(train_loader)
    print(f"Total {total_batches} batches")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Record epoch start time
        model.train()
        total_train_loss = 0
        batch_times = []  # To store the time taken for each batch

        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()  # Record batch start time

            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss = criterion(shift_logits, shift_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            batch_end_time = time.time()  # Record batch end time
            batch_time = (
                batch_end_time - batch_start_time
            )  # Calculate time taken for the current batch
            logging.warning(
                f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} time: {batch_time:.4f}s"
            )
            batch_times.append(batch_time)
            print(
                f"Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches} time: {batch_time:.4f}s"
            )
            logging.warning(f"Loss: {loss.item()}")

        # Calculate the average time per batch
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_end_time = time.time()  # Record epoch end time
        epoch_time = (
            epoch_end_time - epoch_start_time
        )  # Calculate the total time for the epoch

        avg_train_loss = total_train_loss / total_batches
        avg_val_loss = validate(model, valid_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {avg_val_loss:.4f}, Epoch time: {epoch_time:.2f}s, Average Batch time: {avg_batch_time:.4f}s"
        )
        logging.warning(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {avg_val_loss:.4f}, Epoch time: {epoch_time:.2f}s, Average Batch time: {avg_batch_time:.4f}s"
        )
        best_val_loss = save_best_model(
            model, tokenizer, epoch + 1, best_val_loss, avg_val_loss, save_path
        )


# ------------------------- Main training -------------------------
if __name__ == "__main__":
    """
    Main execution block for DEFT-GaLore fine-tuning pipeline.
    
    This section orchestrates the complete training workflow:
    1. Argument parsing and environment setup
    2. Model and tokenizer initialization
    3. DEFT-GaLore parameter group configuration
    4. Data preprocessing and loader creation
    5. Training execution with gradient projection
    
    The pipeline is specifically designed for surgical AI applications with
    question-answering tasks, utilizing memory-efficient training through
    gradient compression techniques.
    """
    args = parse_args()
    set_seed(args.seed)  # Ensure reproducible training results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set environment variables (e.g., TOKENIZERS_PARALLELISM, etc.)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

    # Login to Hugging Face (token can be configured)
    login(token=args.HF_TOKEN)

    # Load pre-trained model and tokenizer with proper configuration
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    # model.to(device)  # Model is auto-distributed via device_map in load function

    # ============ DEFT-GaLore Parameter Configuration ============
    # Identify attention projection layers for gradient compression
    galore_params = []
    target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention layers
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(target_key in module_name for target_key in target_modules_list):
            continue
        print("enable GaLore for weights in module: ", module_name)
        galore_params.append(module.weight)
    
    # Separate parameters into GaLore-enabled and regular groups
    id_galore_params = [id(p) for p in galore_params]
    # Other parameters are classified as regular_params (no projection)
    regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
    
    # Define parameter groups with DEFT-GaLore configuration
    param_groups = [
        {"params": regular_params},  # Standard optimization
        {
            "params": galore_params,  # Gradient projection enabled
            "rank": args.rank,
            "update_proj_gap": args.update_proj_gap,
            "scale": args.galore_scale,
            "proj_type": args.proj_type,
        },
    ]

    trainable_params = param_groups

    # Initialize DEFT-GaLore optimizer with gradient projection capabilities
    optimizer = DEFT_GaLoreAdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    
    # CrossEntropyLoss with ignore_index=-100 for padding tokens
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # ============ Data Processing and Loading ============
    # Process CSV files into question-answer format for surgical AI training
    train_qa_samples, valid_qa_samples = process_qa_samples(
        args.train_file, args.val_file
    )
    
    # Convert to HuggingFace datasets with tokenization
    dataset_train = Dataset.from_list(train_qa_samples).map(
        lambda ex: preprocess_data(ex, tokenizer), remove_columns=["question", "answer"]
    )
    dataset_valid = Dataset.from_list(valid_qa_samples).map(
        lambda ex: preprocess_data(ex, tokenizer), remove_columns=["question", "answer"]
    )

    # Create data loaders with custom collation function
    train_loader = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ============ Execute Training ============
    # Launch main training loop with all configured components
    train(
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        args.num_epochs,
        args.save_path,
    )
