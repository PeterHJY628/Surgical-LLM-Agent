"""
Data Loading and Preprocessing Module for DEFT-GaLore Surgical AI Training

This module provides comprehensive data processing capabilities for training surgical AI
models with the DEFT-GaLore optimization framework. It handles CSV-based question-answer
datasets, applies surgical AI agent prompt formatting, and implements efficient tokenization
with proper label masking for language model training.

Key Features:
- CSV data loading with Input/Label column validation
- Surgical AI agent prompt template generation
- Text preprocessing with question/answer formatting
- Token-level label masking for autoregressive training
- Batch collation with proper tensor formatting
- Reproducible data processing with seed management

Data Format:
- Input CSV files with 'Input' and 'Label' columns
- Input: Surgical queries/questions from medical professionals
- Label: Expected model selections and prompts for surgical procedures
- Output: Tokenized sequences ready for DEFT-GaLore training

Usage:
    train_samples, val_samples = process_qa_samples("train.csv", "val.csv")
    dataset = Dataset.from_list(train_samples).map(
        lambda ex: preprocess_data(ex, tokenizer)
    )

Date: 2025
"""
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def set_seed(seed: int):
    """
    Sets random seeds for reproducible data processing and training.
    
    Ensures deterministic behavior across data loading, shuffling, and model
    initialization by configuring random number generators for Python, NumPy,
    and PyTorch. This is essential for reproducible experimental results in
    surgical AI model training.
    
    Args:
        seed (int): Random seed value to apply across all random number generators
        
    Note:
        - Configures both CPU and CUDA random number generators
        - Sets deterministic CUDA operations for complete reproducibility
        - May slightly impact performance but ensures consistent results
        - Should be called before any data loading or model initialization
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_SM(que: str) -> str:
    """
    Generates standardized system message templates for surgical AI agent training.
    
    Creates comprehensive prompt templates that instruct the AI agent on how to
    handle surgical queries by selecting appropriate models and generating
    corresponding prompts. The template provides clear guidance for both
    single-model and multi-model surgical scenarios.
    
    Args:
        que (str): Input surgical question/query from medical professionals
        
    Returns:
        str: Formatted system message containing:
            - Role definition as surgical AI agent for pituitary surgery
            - List of 5 available specialized models
            - Decision criteria for single vs. multi-model selection
            - Concrete examples for both scenarios
            - Embedded input question for context
            
    Available Models:
        - Segment-Video: Video-based surgical scene segmentation
        - Segment-MRI: MRI image segmentation for anatomical structures
        - Track-Instrument: Surgical instrument tracking and identification
        - Surgical-VQA: Visual question answering for surgical scenes
        - Overlaying: Multi-modal information overlay and visualization
        
    Note:
        The template emphasizes structured output format without extraneous
        text to ensure consistent model training and evaluation.
    """
    return (
        "You are a surgical AI agent assisting in pituitary surgery. Your job is to handle surgeons' queries efficiently by choosing appropriate text-promptable AI models and generating corresponding prompts.\n"
        "Available models: Segment-Video, Segment-MRI, Track-Instrument, Surgical-VQA, Overlaying.\n"
        "Question: {que}\n"
        "- Use ONE model if query focuses on a single, simple aspect:\n"
        "Example (single-model):\n"
        "Model: Segment-Video\nPrompt: Segment the sella in the video.\n"
        "- Use MULTIPLE models if query requires several types of information:\n"
        "Example (multi-model):\n"
        "Step1:\nModel: Segment-MRI\nPrompt: Segment the pituitary tumor from MRI.\n"
        "Step2:\nModel: Segment-Video\nPrompt: Segment the sella in the video.\n"
        "Now, follow the same format to answer the provided question—no extra text, labels, or formatting."
    ).format(que=que)


def extract_question(text):
    """
    Safely extracts and cleans question text from CSV data.
    
    Handles potential data quality issues in CSV files by checking for null
    values and stripping whitespace. This ensures robust data processing
    even with imperfect input data formats.
    
    Args:
        text: Raw text data from CSV Input column (may be NaN or contain whitespace)
        
    Returns:
        str: Cleaned question text, or empty string if input is null/invalid
        
    Note:
        Uses pandas.notna() to handle various null representations (NaN, None, etc.)
        in CSV data, ensuring consistent text processing across different data sources.
    """
    return text.strip() if pd.notna(text) else ""


def extract_answer(text):
    """
    Safely extracts and cleans answer text from CSV data.
    
    Processes label/answer data with the same robustness as question extraction,
    handling null values and whitespace to ensure clean training data.
    
    Args:
        text: Raw text data from CSV Label column (may be NaN or contain whitespace)
        
    Returns:
        str: Cleaned answer text, or empty string if input is null/invalid
        
    Note:
        Maintains consistency with extract_question() for uniform data processing
        and handles edge cases in CSV data formatting.
    """
    return text.strip() if pd.notna(text) else ""


# Process CSV data to build training and validation samples
def process_qa_samples(train_file, val_file):
    """
    Processes CSV files to create structured question-answer samples for surgical AI training.
    
    This function is the main data processing pipeline that converts raw CSV data
    into properly formatted training samples. It validates data structure,
    applies surgical AI prompt templates, and creates separate training and
    validation datasets ready for model training.
    
    Args:
        train_file (str): Path to training CSV file with Input/Label columns
        val_file (str): Path to validation CSV file with Input/Label columns
        
    Returns:
        tuple: Two lists containing processed samples:
            - train_qa_samples (list): Training question-answer pairs with prompts
            - valid_qa_samples (list): Validation question-answer pairs with prompts
            
    Data Processing Steps:
        1. Load and validate CSV file structure
        2. Extract questions and answers with null handling
        3. Apply surgical AI agent prompt templates to questions
        4. Create structured dictionaries for each sample
        5. Filter out samples with missing data
        6. Report dataset statistics and example samples
        
    Expected CSV Format:
        - Input column: Surgical queries/questions
        - Label column: Expected model selections and prompts
        
    Note:
        - Returns None if required columns are missing
        - Prints dataset statistics and sample examples for verification
        - Integrates surgical AI prompt generation for consistent training format
    """
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    for df, name in [(train_df, "Train.csv"), (val_df, "Test.csv")]:
        if "Input" not in df.columns or "Label" not in df.columns:
            print(f"CSV file {name} is missing 'Input' or 'Label' column")
            return

    train_qa_samples = []
    for _, row in train_df.iterrows():
        question = extract_question(str(row["Input"]))
        answer = extract_answer(str(row["Label"]))
        if question and answer:
            question = generate_SM(question)
            train_qa_samples.append({"question": question, "answer": answer})

    valid_qa_samples = []
    for _, row in val_df.iterrows():
        question = extract_question(str(row["Input"]))
        answer = extract_answer(str(row["Label"]))
        if question and answer:
            question = generate_SM(question)
            valid_qa_samples.append({"question": question, "answer": answer})

    print("Train sample num:", len(train_qa_samples))
    print("Test sample num:", len(valid_qa_samples))
    if train_qa_samples:
        print("Example Train Sample:", train_qa_samples[0])
    if valid_qa_samples:
        print("Example Test Sample:", valid_qa_samples[0])

    return train_qa_samples, valid_qa_samples


# Data preprocessing
def preprocess_data(example, tokenizer, max_length=260):
    """
    Preprocesses individual samples for autoregressive language model training.
    
    This function converts structured question-answer pairs into tokenized sequences
    with proper label masking for causal language modeling. It implements the
    standard approach for training conversational AI models where only the
    response tokens contribute to the loss calculation.
    
    Args:
        example (dict): Sample containing 'question' and 'answer' keys
        tokenizer: HuggingFace tokenizer for text encoding
        max_length (int, optional): Maximum sequence length for truncation/padding.
                                  Defaults to 260 tokens.
                                  
    Returns:
        dict: Tokenized sample ready for training containing:
            - input_ids: Token IDs for the complete sequence
            - attention_mask: Attention mask for padding tokens
            - labels: Label IDs with question tokens masked (-100)
            
    Processing Steps:
        1. Format input as "Query:\n{question}\nResponse:\n{answer}"
        2. Tokenize with truncation and padding to max_length
        3. Create labels by copying input_ids
        4. Calculate question length for masking
        5. Mask question tokens and padding tokens with -100
        6. Return complete tokenized sample
        
    Label Masking Strategy:
        - Question tokens: Masked with -100 (no loss contribution)
        - Padding tokens: Masked with -100 (no loss contribution)  
        - Answer tokens: Preserved for loss calculation
        - This enables the model to learn response generation only
        
    Note:
        The -100 label value is ignored by PyTorch's CrossEntropyLoss,
        effectively masking those positions from gradient computation.
    """
    input_text = f"Query:\n{example['question']}\nResponse:\n{example['answer']}"
    inputs = tokenizer(
        input_text, truncation=True, padding="max_length", max_length=max_length
    )
    labels = inputs["input_ids"].copy()
    question_length = (
        len(tokenizer(f"Query:\n{example['question']}\nResponse:\n")["input_ids"]) - 1
    )
    for i in range(len(labels)):
        if i < question_length or labels[i] == tokenizer.pad_token_id:
            labels[i] = -100
    inputs["labels"] = labels
    return inputs


def collate_fn(batch):
    """
    Custom collation function for batching preprocessed samples in DataLoader.
    
    Converts a list of individual tokenized samples into properly batched tensors
    suitable for efficient model training. This function handles the conversion
    from Python lists to PyTorch tensors with consistent shapes and data types.
    
    Args:
        batch (list): List of preprocessed samples, each containing:
                     - input_ids: List of token IDs
                     - attention_mask: List of attention values  
                     - labels: List of label IDs (with -100 for masked tokens)
                     
    Returns:
        tuple: Batched tensors ready for model input:
            - input_ids (torch.LongTensor): Shape [batch_size, max_length]
            - attention_mask (torch.LongTensor): Shape [batch_size, max_length]
            - labels (torch.LongTensor): Shape [batch_size, max_length]
            
    Tensor Properties:
        - All tensors use LongTensor dtype for token IDs and labels
        - Consistent batch dimension across all returned tensors
        - Maintains padding and masking from preprocessing step
        - Ready for direct use with model.forward() and loss computation
        
    Note:
        This function is typically passed to DataLoader as the collate_fn
        parameter to enable custom batching behavior for the surgical AI
        training pipeline.
    """
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor(
        [item["attention_mask"] for item in batch], dtype=torch.long
    )
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return input_ids, attention_mask, labels
