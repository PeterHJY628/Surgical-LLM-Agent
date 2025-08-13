# ---------------------------Import Packages------------------------------------------------------
"""
DEFT-GaLore Model Inference and Evaluation Pipeline for Surgical AI

This script provides comprehensive inference and evaluation capabilities for models trained
with the DEFT-GaLore optimization framework. It is specifically designed for surgical AI
applications involving multi-model selection and prompt generation tasks.

Key Features:
- Quantized model inference with 4-bit precision for memory efficiency
- Multi-model evaluation across different surgical AI tasks
- Comprehensive metric computation (ROUGE, BLEU, METEOR, F1)
- Model selection accuracy assessment for surgical AI agents
- Automated result aggregation and summarization
- Support for both LLaMA and Qwen model architectures

Evaluation Metrics:
- Text Generation: ROUGE-1, ROUGE-L, BLEU-1 to BLEU-4, METEOR
- Model Selection: Position-wise accuracy, F1 scores for 1/2/3 model scenarios
- Comprehensive: Aggregated results across multiple test datasets

Usage:
    python inference.py --best_model_path path/to/model \
                       --input_files test1.csv,test2.csv \
                       --output_dir results/ \
                       --model_type llama

Date: 2025
"""
import os
import random
import numpy as np
import pandas as pd
import re
import argparse
import warnings
import logging
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

import evaluate
import nltk
from nltk.tokenize import word_tokenize

# NLTK downloads
nltk.download("wordnet")
nltk.download("punkt_tab")
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """
    Parses command-line arguments for the DEFT-GaLore inference and evaluation pipeline.
    
    This function configures all parameters needed for model inference, evaluation,
    and result generation across multiple surgical AI datasets. It supports both
    single and multi-file evaluation with comprehensive metric computation.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - best_model_path: Path to the fine-tuned model checkpoint
            - input_files: Comma-separated test file paths for evaluation
            - output_dir: Directory for saving inference results and metrics
            - batch_size: Inference batch size for memory management
            - model_type: Model architecture type (llama/qwen) for proper handling
            - HF_TOKEN: Hugging Face authentication for model access
            - seed: Random seed for reproducible inference results
    """
    parser = argparse.ArgumentParser(description="Testing LLaMA with DEFT-GaLore")
    # seed
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    # Model and tokenizer configuration
    parser.add_argument(
        "--best_model_path",
        type=str,
        default="Your_path_to_best_model",
        help="Path to the best fine-tuned model",
    )
    parser.add_argument(
        "--input_files",
        type=str,
        default="Your_path_to_input_files/Surgical-VQA_V.csv,"
        "Your_path_to_input_files/Segment-MRI_V.csv,"
        "Your_path_to_input_files/Segment-Video_V.csv,"
        "Your_path_to_input_files/Track-Instrument_V.csv,"
        "Your_path_to_input_files/2-model_V.csv,"
        "Your_path_to_input_files/3-model_V.csv",
        help="Comma-separated list of evaluation file paths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Your_path_to_output_dir",
        help="Output directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for inference"
    )
    parser.add_argument(
        "--lr_description", type=str, default="3e-7", help="Learning rate description"
    )
    parser.add_argument(
        "--HF_TOKEN",
        type=str,
        default="Your_HuggingFace_Token",
        help="Hugging Face token for authentication",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=["llama", "qwen"],
        help="Model type to use for inference (llama or qwen)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """
    Sets random seeds for reproducible inference across all relevant libraries.
    
    Ensures deterministic behavior during model inference and evaluation by
    configuring random number generators for Python, NumPy, PyTorch CPU/GPU,
    and CUDA backend operations. This is crucial for consistent evaluation
    results across multiple runs.
    
    Args:
        seed (int): Random seed value to apply across all libraries
        
    Note:
        Deterministic CUDA operations may slightly impact inference speed
        but ensure reproducible results for scientific evaluation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_SM(que: str) -> str:
    """
    Generates the system message template for surgical AI agent interactions.
    
    This function creates a standardized prompt template that instructs the AI agent
    on how to handle surgical queries by selecting appropriate models and generating
    corresponding prompts. It provides clear examples for both single-model and
    multi-model scenarios in pituitary surgery contexts.
    
    Args:
        que (str): The input question/query from the surgeon
        
    Returns:
        str: Formatted system message with embedded question and instructions
             for model selection and prompt generation
             
    Note:
        The template includes 5 available models: Segment-Video, Segment-MRI,
        Track-Instrument, Surgical-VQA, and Overlaying, each specialized for
        different aspects of surgical assistance.
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


def format_data(sample):
    """
    Formats input data into the conversation structure expected by the model.
    
    Converts raw question-answer pairs into a structured conversation format
    with system and assistant roles. The system message contains the surgical
    AI agent instructions, while the assistant content contains the expected response.
    
    Args:
        sample: Tuple containing (question, answer) pair from the dataset
        
    Returns:
        list: Conversation structure with role-based message formatting:
            - System role with surgical AI agent instructions
            - Assistant role with the target response
    """
    system_message = generate_SM(sample[0])
    return [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": sample[1]},
    ]


def custom_collate_fn(sample):
    """
    Custom collation function for DataLoader that preserves sample structure.
    
    This function maintains the original format of samples returned by the DataLoader
    without applying any transformations or batching modifications. It's designed
    to work with pre-formatted conversation structures.
    
    Args:
        sample: Input sample from the dataset
        
    Returns:
        sample: Unchanged sample maintaining its original structure
    """
    # Keep the format of the samples returned by DataLoader unchanged
    return sample


def generate_answer(question, model, tokenizer):
    """
    Generates model responses for surgical AI queries using text generation.
    
    This function handles the complete inference pipeline: tokenization, generation,
    and response extraction. It uses the model's generate function with controlled
    parameters to produce surgical AI agent responses for model selection and
    prompt generation tasks.
    
    Args:
        question (str): Input surgical query/question
        model: Pre-trained language model (quantized for efficiency)
        tokenizer: Associated tokenizer for text processing
        
    Returns:
        str: Generated response containing model selections and prompts,
             extracted from the "Response:" section of the generated text
             
    Note:
        - Uses max_new_tokens=200 to limit response length
        - Input is truncated to max_length=2048 for memory management
        - Applies torch.no_grad() for memory-efficient inference
    """
    model.eval()
    input_text = f"Query:\n{question}\nResponse:\n"
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id
        )
    answer = (
        tokenizer.decode(output[0], skip_special_tokens=True)
        .split("Response:\n")[-1]
        .strip()
    )
    return answer


def extract_prompt(entry):
    """
    Extracts prompt text from structured model responses.
    
    Parses multi-line responses to identify and extract all lines that start
    with "Prompt:", then combines them into a single pipe-separated string.
    This is used for prompt-based evaluation metrics.
    
    Args:
        entry (str): Multi-line text response containing model selections and prompts
        
    Returns:
        str: Pipe-separated prompts (e.g., "prompt1|prompt2|prompt3")
             or empty string if no prompts found
             
    Example:
        Input: "Model: Segment-MRI\\nPrompt: Segment tumor\\nModel: Track-Instrument\\nPrompt: Track tools"
        Output: "Segment tumor|Track tools"
    """
    # Extract the content of all lines starting with "Prompt:" and join them with "|"
    prompts = [
        line[len("Prompt: ") :].strip()
        for line in entry.split("\n")
        if line.startswith("Prompt:")
    ]
    return "|".join(prompts) if prompts else ""


def group_by_sentence_position(all_prompts, num_sentences):
    """
    Groups prompts by their positional order for structured evaluation.
    
    Organizes pipe-separated prompts into position-based groups to enable
    evaluation of model performance at each step of multi-model scenarios.
    This is essential for analyzing surgical AI agent's sequential decision-making.
    
    Args:
        all_prompts (list): List of pipe-separated prompt strings
        num_sentences (int): Expected number of prompts per sample
        
    Returns:
        list: List of lists where each sublist contains prompts from the same
              position across all samples. Empty strings fill missing positions.
              
    Example:
        Input: ["prompt1|prompt2", "promptA|promptB"], num_sentences=2
        Output: [["prompt1", "promptA"], ["prompt2", "promptB"]]
    """
    grouped_sentences = [[] for _ in range(num_sentences)]
    for prompts in all_prompts:
        sentences = prompts.split("|")
        for i in range(num_sentences):
            if i < len(sentences):
                grouped_sentences[i].append(sentences[i])
            else:
                grouped_sentences[i].append("")
    return grouped_sentences


def compute_metrics(grouped_pred_prompts, grouped_ans_prompts):
    """
    Computes comprehensive text generation metrics using Hugging Face evaluate library.
    
    Calculates ROUGE, BLEU, and METEOR scores for grouped prompt predictions
    to assess the quality of generated surgical prompts at each position in
    multi-model scenarios. This provides detailed analysis of text generation
    performance across different steps of surgical AI workflows.

    Args:
        grouped_pred_prompts (list): List of prediction groups, each containing
                                   prompts from the same position across samples
        grouped_ans_prompts (list): List of reference groups corresponding
                                  to grouped_pred_prompts

    Returns:
        tuple: Three evaluation components:
            - rouge_results (list): ROUGE metrics for each position group
            - bleu_scores (list): BLEU-1 to BLEU-4 scores for each group  
            - meteor_scores (list): METEOR scores for each position group
            
    Note:
        - ROUGE: Measures overlap of n-grams and longest common subsequences
        - BLEU: Evaluates precision of n-gram matches with brevity penalty
        - METEOR: Considers stemming, synonymy, and word order for evaluation
    """
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    bleu_scores = []
    meteor_scores = []
    rouge_results = []

    for i in range(len(grouped_pred_prompts)):
        pred_group = grouped_pred_prompts[i]
        ans_group = grouped_ans_prompts[i]

        # ROUGE: Directly pass the list of predicted and reference texts
        rouge_result = rouge.compute(predictions=pred_group, references=ans_group)
        rouge_results.append(rouge_result)

        # BLEU: Reference texts need to be passed in a list of lists format
        bleu_result = bleu.compute(
            predictions=pred_group, references=[[ref] for ref in ans_group]
        )
        bleu_scores.append(
            {
                "bleu1": bleu_result["precisions"][0],
                "bleu2": bleu_result["precisions"][1],
                "bleu3": bleu_result["precisions"][2],
                "bleu4": bleu_result["precisions"][3],
            }
        )

        # METEOR: Directly pass the list of predicted and reference texts
        meteor_result = meteor.compute(predictions=pred_group, references=ans_group)
        meteor_scores.append(meteor_result["meteor"])

    return rouge_results, bleu_scores, meteor_scores


def extract_model(text, model_names):
    """
    Extracts mentioned model names from response text using regex matching.
    
    Identifies and extracts all occurrences of predefined surgical AI model names
    from generated responses. This is crucial for evaluating model selection
    accuracy in multi-model surgical AI scenarios.
    
    Args:
        text (str): Generated response text containing model selections
        model_names (list): List of valid model names to search for
        
    Returns:
        str: Pipe-separated string of found model names in order of appearance,
             or empty string if no models found
             
    Example:
        Input: "Use Segment-MRI and Track-Instrument", models=["Segment-MRI", "Track-Instrument", "Surgical-VQA"]
        Output: "Segment-MRI|Track-Instrument"
    """
    # Use regular expressions to match model names from the list
    matches = re.findall(
        r"\b(?:" + "|".join(map(re.escape, model_names)) + r")\b", text
    )
    return "|".join(matches) if matches else ""


def match_rate_per_Cat(pred_models_format, true_models_format):
    """
    Calculates position-wise model selection accuracy for surgical AI evaluation.
    
    Computes the accuracy of model selection at each position (1st, 2nd, 3rd)
    in multi-model surgical scenarios. This metric is crucial for evaluating
    the AI agent's ability to correctly sequence model selections for complex
    surgical workflows.
    
    Args:
        pred_models_format (list): List of predicted model sequences (pipe-separated)
        true_models_format (list): List of ground truth model sequences (pipe-separated)
        
    Returns:
        tuple: Three accuracy percentages:
            - first_model_match_rate (float): Accuracy of 1st model selection
            - second_model_match_rate (float): Accuracy of 2nd model selection  
            - third_model_match_rate (float): Accuracy of 3rd model selection
            
    Note:
        Missing predictions are padded with spaces for fair comparison.
        Percentages are calculated only for samples that have the corresponding
        number of true models.
    """
    # Initialize counters
    first_model_match_count = 0
    second_model_match_count = 0
    third_model_match_count = 0
    total_count = len(true_models_format)

    # Iterate through both lists
    for pred, true in zip(pred_models_format, true_models_format):
        # Split model names
        pred_models = pred.split("|")
        true_models = true.split("|")
        while len(pred_models) < len(true_models):
            pred_models.append(" ")

        # check if the first model matches
        if len(true_models) > 0 and pred_models[0] == true_models[0]:
            first_model_match_count += 1

        # check if the second model matches
        if len(true_models) > 1 and pred_models[1] == true_models[1]:
            second_model_match_count += 1

        if len(true_models) > 2 and pred_models[2] == true_models[2]:
            third_model_match_count += 1

    # Calculate matching percentage
    first_model_match_rate = (
        (first_model_match_count / total_count * 100) if total_count > 0 else 0
    )
    second_model_match_rate = (
        (second_model_match_count / total_count * 100) if total_count > 0 else 0
    )
    third_model_match_rate = (
        (third_model_match_count / total_count * 100) if total_count > 0 else 0
    )
    return first_model_match_rate, second_model_match_rate, third_model_match_rate


def f1_score_set(pred_list, true_list):
    """
    Computes F1 score for set-based model selection evaluation.
    
    Calculates F1 score by treating model selections as sets, measuring
    precision and recall of model choices regardless of order. This provides
    a balanced evaluation metric for multi-model selection accuracy.
    
    Args:
        pred_list (list): List of predicted model names
        true_list (list): List of ground truth model names
        
    Returns:
        float: F1 score (0.0 to 1.0) representing harmonic mean of precision and recall
        
    Note:
        - Precision: Fraction of predicted models that are correct
        - Recall: Fraction of true models that were predicted
        - F1: 2 * (precision * recall) / (precision + recall)
        - Returns 0.0 if no predictions or true models exist
    """
    pred_set = set(pred_list)
    true_set = set(true_list)
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    return (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )


def evaluate_f1_by_selection_count(pred_models_format, true_models_format):
    """
    Evaluates F1 scores stratified by the number of models in surgical scenarios.
    
    Computes separate F1 scores for single-model, two-model, and three-model
    surgical scenarios to understand performance across different complexity levels.
    This stratified analysis is crucial for surgical AI where different procedures
    may require different numbers of specialized models.

    Args:
        pred_models_format (list): List of predicted model sequences (pipe-separated strings)
        true_models_format (list): List of ground truth model sequences (pipe-separated strings)
        
    Returns:
        tuple: F1 scores for different model count scenarios:
            - avg_one_model_f1 (float): Average F1 for single-model scenarios
            - avg_two_model_f1 (float): Average F1 for two-model scenarios  
            - avg_three_model_f1 (float): Average F1 for three-model scenarios
            
    Note:
        - Insufficient predictions are padded with empty strings
        - Only considers the first N models for N-model scenarios
        - Samples with other model counts are ignored
        - Returns 0.0 for categories with no samples
    """
    one_model_scores = []
    two_model_scores = []
    three_model_scores = []

    for pred, true in zip(pred_models_format, true_models_format):
        # Split strings and remove extra spaces
        pred_models = [m.strip() for m in pred.split("|")]
        true_models = [m.strip() for m in true.split("|")]

        # If the number of predictions is insufficient, add empty strings (optional, adjust as needed)
        while len(pred_models) < len(true_models):
            pred_models.append("")

        # Classify and calculate based on the number of true models
        if len(true_models) == 1:
            # For the case of one model, take the first one
            score = f1_score_set(pred_models[:1], true_models[:1])
            one_model_scores.append(score)
        elif len(true_models) == 2:
            # For the case of two models, take the first two
            score = f1_score_set(pred_models[:2], true_models[:2])
            two_model_scores.append(score)
        elif len(true_models) == 3:
            # For the case of three models, take the first three
            score = f1_score_set(pred_models[:3], true_models[:3])
            three_model_scores.append(score)
        else:
            # If the number of true models in the sample is not 1, 2, or 3, handle as needed (ignored here)
            continue

    avg_one_model_f1 = (
        sum(one_model_scores) / len(one_model_scores) if one_model_scores else 0
    )
    avg_two_model_f1 = (
        sum(two_model_scores) / len(two_model_scores) if two_model_scores else 0
    )
    avg_three_model_f1 = (
        sum(three_model_scores) / len(three_model_scores) if three_model_scores else 0
    )
    return avg_one_model_f1, avg_two_model_f1, avg_three_model_f1


class TextQuestionLabelDataset(Dataset):
    """
    Custom PyTorch Dataset for surgical AI question-answer evaluation data.
    
    This dataset class handles CSV files containing surgical queries and their
    corresponding model selection labels. It provides efficient data loading
    for inference and evaluation pipelines with proper indexing and access methods.
    
    The dataset expects CSV files with 'Input' and 'Label' columns where:
    - Input: Contains surgical queries/questions
    - Label: Contains expected model selections and prompts
    
    Args:
        input_file (str): Path to CSV file containing evaluation data
        
    Attributes:
        data (pd.DataFrame): Loaded CSV data
        questions (list): List of input questions/queries
        labels (list): List of corresponding ground truth labels
    """
    def __init__(self, input_file):
        self.data = pd.read_csv(input_file)
        self.questions = self.data["Input"].tolist()
        self.labels = self.data["Label"].tolist()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Total number of question-answer pairs in the dataset
        """
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (question, label) pair for the specified index
        """
        return self.questions[idx], self.labels[idx]


# ----------------- Metric Extraction -----------------
def extract_values(file_path: str) -> dict:
    """
    Extracts evaluation metrics from saved evaluation result files.
    
    Parses evaluation.txt files to extract numerical metric values using
    regex patterns. This function supports different metric types and
    handles variations in file naming for model count-specific F1 scores.
    
    Args:
        file_path (str): Path to the evaluation.txt file to parse
        
    Returns:
        dict: Dictionary containing extracted metric values:
            - rouge1, rougeL: ROUGE scores
            - bleu1-bleu4: BLEU scores  
            - METEOR: METEOR scores
            - Matching_Accuracy: Model selection accuracy percentages
            - F1: F1 scores (context-dependent based on filename)
            
    Note:
        F1 score extraction adapts based on filename:
        - Files starting with "2": Two-model F1 scores
        - Files starting with "3": Three-model F1 scores  
        - Other files: Current/single-model F1 scores
    """
    # Read a evaluation.txt file and return a dictionary of metric values (which may have multiple values).
    with open(file_path, "r", encoding="utf-8") as f:
        txt = f.read()

    pat = {
        "rouge1": r"rouge1'\s*:\s*(?:np\.float64\()?([0-9]*\.?[0-9]+)",
        "rougeL": r"rougeL'\s*:\s*(?:np\.float64\()?([0-9]*\.?[0-9]+)",
        "bleu1": r"bleu1'\s*:\s*([0-9]*\.?[0-9]+)",
        "bleu2": r"bleu2'\s*:\s*([0-9]*\.?[0-9]+)",
        "bleu3": r"bleu3'\s*:\s*([0-9]*\.?[0-9]+)",
        "bleu4": r"bleu4'\s*:\s*([0-9]*\.?[0-9]+)",
        "METEOR": r"METEOR Score\s*:?\s*\[\s*(?:np\.float64\()?([0-9]*\.?[0-9]+)",
        "Matching_Accuracy": r"Matching Accuracy of(?:\s+the)?\s+\d+(?:st|nd|rd|th)\s+model:\s*([0-9]*\.?[0-9]+)%",
    }
    out = {k: [float(x) for x in re.findall(pat[k], txt)] for k in pat}

    fname = os.path.basename(file_path).lstrip()
    head = fname[0] if fname else ""
    if head == "2":
        f1_pat = r"F1 score (?:of|for) two models\s*:\s*([0-9]*\.?[0-9]+)"
    elif head == "3":
        f1_pat = r"F1 score (?:of|for) three models\s*:\s*([0-9]*\.?[0-9]+)"
    else:
        f1_pat = r"F1 score (?:of|for) current model\s*:\s*([0-9]*\.?[0-9]+)"
    out["F1"] = [float(x) for x in re.findall(f1_pat, txt)]

    return out


def mean(lst):
    """
    Calculates the arithmetic mean of a list of numbers.
    
    Provides safe mean calculation with handling for empty lists.
    Used for aggregating metric values across multiple evaluation files.
    
    Args:
        lst (list): List of numerical values
        
    Returns:
        float: Arithmetic mean of the list, or 0.0 if list is empty
    """
    return sum(lst) / len(lst) if lst else 0.0


def summarize_one_dir(root: str, files: list[str]):
    """
    Aggregates evaluation metrics from multiple result files in a directory.
    
    Processes all evaluation.txt files in a directory to compute average metrics
    across different test datasets. Creates comprehensive summary reports in both
    text and CSV formats for easy analysis and comparison.
    
    Args:
        root (str): Root directory path containing evaluation files
        files (list[str]): List of filenames in the directory
        
    Side Effects:
        Creates two output files in the root directory:
        - Average.txt: Human-readable summary with metric descriptions
        - Average.csv: Machine-readable CSV with metric values (percentages)
        
    Note:
        - Only processes files ending with "evaluation.txt"
        - Skips directories without evaluation files
        - Converts most metrics to percentages (×100) in CSV output
        - Matching accuracy remains as percentage in both outputs
        - Prints processing status for each directory
    """
    # Take all metrics from evaluation.txt files in the root directory and write the average to Average.txt and Average.csv.
    all_vals = defaultdict(list)

    eva_files = [f for f in files if f.lower().endswith("evaluation.txt")]
    if not eva_files:  # IF the directory has no evaluation.txt, skip it
        return

    for f in eva_files:
        fp = os.path.join(root, f)
        metrics = extract_values(fp)
        for k, v in metrics.items():
            all_vals[k].extend(v)

    avg = {k: mean(v) for k, v in all_vals.items()}

    # ---------- Write Average.txt ----------
    txt_fp = os.path.join(root, "Average.txt")
    with open(txt_fp, "w", encoding="utf-8") as w:
        w.write("Average rouge1: {:.4f}\n".format(avg.get("rouge1", 0)))
        w.write("Average rougeL: {:.4f}\n".format(avg.get("rougeL", 0)))
        w.write("Average bleu1 Score: {:.4f}\n".format(avg.get("bleu1", 0)))
        w.write("Average bleu2 Score: {:.4f}\n".format(avg.get("bleu2", 0)))
        w.write("Average bleu3 Score: {:.4f}\n".format(avg.get("bleu3", 0)))
        w.write("Average bleu4 Score: {:.4f}\n".format(avg.get("bleu4", 0)))
        w.write("Average METEOR Score: {:.4f}\n".format(avg.get("METEOR", 0)))
        w.write("Average F1 Score: {:.4f}\n".format(avg.get("F1", 0)))
        w.write(
            "Average Matching Accuracy: {:.4f}\n".format(
                avg.get("Matching_Accuracy", 0)
            )
        )

    # ---------- Write Average.csv ----------
    csv_fp = os.path.join(root, "Average.csv")
    headers = [
        "bleu1",
        "bleu2",
        "bleu3",
        "bleu4",
        "rouge1",
        "rougeL",
        "METEOR",
        "F1",
        "Matching_Accuracy",
    ]
    row = [
        avg.get("bleu1", 0) * 100,
        avg.get("bleu2", 0) * 100,
        avg.get("bleu3", 0) * 100,
        avg.get("bleu4", 0) * 100,
        avg.get("rouge1", 0) * 100,
        avg.get("rougeL", 0) * 100,
        avg.get("METEOR", 0) * 100,
        avg.get("F1", 0) * 100,
        avg.get("Matching_Accuracy", 0),
    ]
    with open(csv_fp, "w", encoding="utf-8") as w:
        w.write(",".join(headers) + "\n")
        w.write(",".join(f"{v:.2f}" for v in row) + "\n")

    print(f"[OK] {root}  ({len(eva_files)} files)")


if __name__ == "__main__":
    """
    Main execution block for DEFT-GaLore model inference and evaluation pipeline.
    
    This section orchestrates the complete evaluation workflow:
    1. Argument parsing and environment setup
    2. Model loading with 4-bit quantization for efficiency  
    3. Multi-file inference across different surgical AI datasets
    4. Comprehensive metric computation and result aggregation
    5. Automated summary generation for comparative analysis
    
    The pipeline evaluates surgical AI models across multiple dimensions:
    - Text generation quality (ROUGE, BLEU, METEOR)
    - Model selection accuracy (position-wise matching)
    - Scenario-based performance (1/2/3 model F1 scores)
    """
    args = parse_args()
    login(token=args.HF_TOKEN)  # Authenticate with Hugging Face for model access
    set_seed(args.seed)  # Set random seed for reproducibility

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ============ Model Loading with 4-bit Quantization ============
    # Configure BitsAndBytesConfig for memory-efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normalized Float 4 (better than standard FP4)
        bnb_4bit_use_double_quant=True,  # Uses secondary quantization for better precision
        bnb_4bit_compute_dtype=torch.float16,  # Keeps computation in FP16 for stability
    )
    
    # Load quantized model and configure tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.best_model_path, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.best_model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Special handling for Qwen models
    if args.model_type == "qwen":
        if tokenizer.eos_token != "<|endoftext|>":
            tokenizer.eos_token = "<|endoftext|>"
            tokenizer.special_tokens_map["eos_token"] = "<|endoftext|>"

    # Parse input file list
    input_files = [f.strip() for f in args.input_files.split(",") if f.strip()]

    # ============ Multi-File Evaluation Loop ============

    # ============ Multi-File Evaluation Loop ============
    for input_file in input_files:
        print(f"Processing file: {input_file}")
        logging.warning(f"Processing file: {input_file}")

        # -------- Data Loading and Preprocessing --------
        dataset = TextQuestionLabelDataset(input_file)
        test_dataset = [format_data(sample) for sample in dataset]
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        all_pred = []
        all_ans = []

        # -------- Inference Process --------
        # Generate model responses for all test samples
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                temp_pred = []
                temp_ans = []
                for sample in batch:
                    output = generate_answer(sample[0]["content"], model, tokenizer)
                    ans = sample[1]["content"]
                    temp_pred.append(output)
                    temp_ans.append(ans)
                all_pred.extend(temp_pred)
                all_ans.extend(temp_ans)

        # -------- Save Prediction Results --------
        # Create prediction output file with readable format
        base_filename = os.path.basename(input_file).replace(".csv", "")
        pred_output_file = os.path.join(args.output_dir, f"{base_filename}_pred.txt")
        with open(pred_output_file, "w") as f:
            for pred_text, ans_text in zip(all_pred, all_ans):
                f.write(f"pred: {pred_text}\n")
                f.write(f"ans: {ans_text}\n\n")
        print(f"Saved predictions to {pred_output_file}")

        # -------- Comprehensive Evaluation Pipeline --------
        # Prepare prompt-based evaluation data
        all_pred_prompts = all_pred
        all_ans_prompts = all_ans
        num_sentences = len(all_ans_prompts[0].split("|"))
        grouped_pred_prompts = group_by_sentence_position(
            all_pred_prompts, num_sentences
        )
        grouped_ans_prompts = group_by_sentence_position(all_ans_prompts, num_sentences)

        # Compute text generation metrics (ROUGE, BLEU, METEOR)
        rouge_results, bleu_scores, meteor_scores = compute_metrics(
            grouped_pred_prompts, grouped_ans_prompts
        )

        # -------- Model Selection Accuracy Evaluation --------
        # Extract and evaluate surgical AI model selections
        model_names = [
            "Segment-MRI",
            "Segment-Video",
            "Track-Instrument",
            "Surgical-VQA",
            "Overlaying",
        ]
        pred_models = [
            extract_model(pred, model_names=model_names).strip() for pred in all_pred
        ]
        true_models = [
            extract_model(ans, model_names=model_names).strip() for ans in all_ans
        ]
        
        # Calculate position-wise matching accuracy
        first_rate, second_rate, third_rate = match_rate_per_Cat(
            pred_models, true_models
        )
        true_models_list = true_models[0].split("|")
        model_num = len(true_models_list)

        # Calculate F1 scores stratified by model count scenarios
        avg_one_f1, avg_two_f1, avg_three_f1 = evaluate_f1_by_selection_count(
            pred_models, true_models
        )

        # -------- Save Comprehensive Evaluation Results --------

        eval_output_file = os.path.join(
            args.output_dir, f"{base_filename}_evaluation.txt"
        )
        with open(eval_output_file, "w") as f:
            f.write(f"Rouge Scores: {rouge_results}\n")
            f.write(f"BLEU Score: {bleu_scores}\n")
            f.write(f"METEOR Score: {meteor_scores}\n")
            if model_num > 0:
                f.write(f"Matching Accuracy of the 1st model: {first_rate:.2f}%\n")
                f.write(f"F1 score of current model: {avg_one_f1:.2f}\n")
            if model_num > 1:
                f.write(f"Matching Accuracy of the 2nd model: {second_rate:.2f}%\n")
                f.write(f"F1 score for two models: {avg_two_f1:.2f}\n")
            if model_num > 2:
                f.write(f"Matching Accuracy of the 3rd model: {third_rate:.2f}%\n")
                f.write(f"F1 score for three models: {avg_three_f1:.2f}\n")
        print(f"Saved evaluation results to {eval_output_file}")
        logging.warning(f"Saved evaluation results to {eval_output_file}")
    
    print("All files processed!")

    # ============ Final Result Aggregation ============
    # Generate summary statistics across all evaluation files
    for root, dirs, files in os.walk(args.output_dir):
        summarize_one_dir(root, files)
