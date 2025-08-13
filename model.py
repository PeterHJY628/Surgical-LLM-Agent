"""
DEFT-GaLore Model Implementation for Surgical AI Training

This module implements the core DEFT (Deterministic Energy-based Fourier Transform) GaLore 
optimization framework for memory-efficient fine-tuning of large language models in surgical 
AI applications. The implementation combines FFT-based gradient projection with low-rank 
adaptation techniques to enable training of billion-parameter models with reduced memory 
requirements.

Key Components:
==============

1. FFT Energy Projection:
   - fft_energy_projection(): Core FFT-based energy projection algorithm
   - Identifies most energetic frequency components for orthogonal basis construction
   - Enables deterministic low-rank gradient approximation

2. DEFT Projector Classes:
   - DEFTProjector: Main projection class for 2D gradient matrices
   - GaLoreProjectorTensor: Extended projector for high-dimensional tensors (>2D)
   - Supports multiple projection types: std, reverse_std, left, right, full

3. DEFT-GaLore Optimizer:
   - DEFT_GaLoreAdamW: Enhanced AdamW optimizer with gradient projection
   - Integrates DEFT projection into the optimization pipeline
   - Maintains separate projectors for different parameter groups
   - Supports both matrix and tensor gradient compression

4. Model Utilities:
   - load_model_and_tokenizer(): Pre-trained model loading with proper configuration
   - save_best_model(): Model checkpoint management with best model tracking
   - Special handling for LLaMA and Qwen model architectures

Technical Features:
==================
- Memory-efficient training through gradient low-rank projection
- FFT-based energy analysis for optimal subspace selection
- Tucker decomposition for high-dimensional tensor handling
- Automatic device mapping and mixed precision support
- Reproducible training with deterministic operations

Usage:
======
This module is designed for integration with the DEFT-GaLore training pipeline:
- Import optimizer: from model import DEFT_GaLoreAdamW
- Configure projector parameters: rank, update_proj_gap, proj_type
- Apply to attention layers: q_proj, k_proj, v_proj, o_proj

The implementation enables efficient fine-tuning of large language models for surgical 
AI applications while maintaining training effectiveness through careful gradient 
reconstruction and projection techniques.

Date: 2025
"""

# ---------------------------Import Packages------------------------------------------------------
import os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

import math
import warnings
from typing import Callable, Iterable, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn as nn
from transformers.utils.versions import require_version
from typing import Optional


# ---------------------------DEFT-Galore part-------------------------------------------
def fft_energy_projection(
    A: torch.Tensor, k: int, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Performs FFT-based energy projection to compute a low-rank orthogonal basis.
    This function applies Fast Fourier Transform to identify the most energetic 
    frequency components and constructs an orthogonal basis from them.
    
    Args:
        A (torch.Tensor): Input real/float matrix of shape (m, n)
        k (int): Target rank for the projection
        generator (Optional[torch.Generator]): Random generator for reproducibility
    
    Returns:
        torch.Tensor: Real orthogonal basis matrix of shape (m, k)
    """
    m, n = A.shape
    # (1) FFT (complex)
    Af = torch.fft.fft(A, dim=1)  # (m, n), complex
    # (2) Energy score
    score = Af.abs().pow(2).sum(dim=0)
    # (3) Top-k frequency index (reproducible)
    idx = torch.topk(score, k, largest=True).indices  # (k,)
    idx = idx[torch.argsort(idx)]  # keep the order
    # (4) Construct sketch Y and QR in complex domain
    Yc = Af[:, idx]  # (m, k), complex
    Qc, _ = torch.linalg.qr(Yc)  # complex QR
    # (5) Convert to real → then QR
    Yr = torch.view_as_real(Qc).reshape(m, -1)  # (m, 2k)
    Qr, _ = torch.linalg.qr(Yr)  # (m, 2k) real orthogonal
    return Qr[:, :k].contiguous()  # take the first k columns


class DEFTProjector:
    """
    DEFT (Distributed Energy-based Frequency Transform) Projector for gradient compression.
    This class implements different projection strategies for low-rank gradient approximation
    using FFT-based energy projection methods.
    
    Args:
        rank (int): The target rank for low-rank approximation
        verbose (bool): Whether to print verbose output during projection
        update_proj_gap (int): Number of iterations between orthogonal matrix updates
        scale (float): Scaling factor applied to projected gradients
        proj_type (str): Type of projection ('std', 'reverse_std', 'left', 'right', 'full')
    """
    def __init__(
        self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type="std"
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def project(self, full_rank_grad, iter):
        """
        Projects full-rank gradients to low-rank subspace using the specified projection type.
        The projection method adapts based on gradient shape and projection type configuration.
        
        Args:
            full_rank_grad (torch.Tensor): Full-rank gradient tensor to be projected
            iter (int): Current iteration number for determining when to update projection matrix
            
        Returns:
            torch.Tensor: Low-rank projected gradient tensor
        """
        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    )
                low_rank_grad = torch.matmul(
                    full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type)
                )
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    )
                low_rank_grad = torch.matmul(
                    self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad
                )
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="left"
                    )
                low_rank_grad = torch.matmul(
                    self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad
                )
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(
                        full_rank_grad, self.rank, type="right"
                    )
                low_rank_grad = torch.matmul(
                    full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type)
                )
        elif self.proj_type == "right":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                )
            low_rank_grad = torch.matmul(
                full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type)
            )
        elif self.proj_type == "left":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                )
            low_rank_grad = torch.matmul(
                self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad
            )
        elif self.proj_type == "full":
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="full"
                )
            low_rank_grad = torch.matmul(
                self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad
            ) @ self.ortho_matrix[1].t().to(full_rank_grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        """
        Projects low-rank gradients back to the original full-rank space.
        This is the inverse operation of the project method, restoring the gradient
        to its original dimensionality for parameter updates.
        
        Args:
            low_rank_grad (torch.Tensor): Low-rank gradient tensor to be projected back
            
        Returns:
            torch.Tensor: Full-rank gradient tensor scaled by the projection scale factor
        """
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
                )
            else:
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
                )
        elif self.proj_type == "reverse_std":
            if (
                low_rank_grad.shape[0] <= low_rank_grad.shape[1]
            ):  # note this is different from std
                full_rank_grad = torch.matmul(
                    self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
                )
            else:
                full_rank_grad = torch.matmul(
                    low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
                )
        elif self.proj_type == "right":
            full_rank_grad = torch.matmul(
                low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type)
            )
        elif self.proj_type == "left":
            full_rank_grad = torch.matmul(
                self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad
            )
        elif self.proj_type == "full":
            full_rank_grad = torch.matmul(
                self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad
            ) @ self.ortho_matrix[1].to(low_rank_grad.device.type)

        return full_rank_grad * self.scale

    def get_orthogonal_matrix(
        self,
        weights,
        rank: int,
        type: str,
        *,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Computes orthogonal projection matrices using FFT-based energy projection.
        Supports different projection types: left, right, or full (bidirectional) projection.
        
        Args:
            weights (torch.Tensor): Weight tensor to compute orthogonal matrix from
            rank (int): Target rank for the projection
            type (str): Projection type - 'left', 'right', or 'full'
            generator (Optional[torch.Generator]): Random generator for reproducibility
            
        Returns:
            torch.Tensor or List[torch.Tensor]: Orthogonal matrix/matrices for projection
            
        Raises:
            ValueError: If type is not one of 'left', 'right', or 'full'
        """

        mp = weights
        float_data = mp.data.dtype == torch.float
        orig_dtype, orig_device = mp.data.dtype, mp.data.device
        A = mp.data.float() if not float_data else mp.data  # (m,n)

        # ---- Left Projection ----
        Q_left = fft_energy_projection(A, rank, generator)

        # ---- Right Projection (if needed) ----
        if type in {"right", "full"}:
            # Do the same projection on A^T, then transpose
            Q_right = (
                fft_energy_projection(A.t(), rank, generator).t().contiguous()
            )  # (k,n)

        def _cast(x):
            return x if float_data else x.to(orig_device).type(orig_dtype)

        if type == "left":
            return _cast(Q_left)
        elif type == "right":
            return _cast(Q_right)
        elif type == "full":
            return [_cast(Q_left), _cast(Q_right)]
        else:
            raise ValueError("type must be 'left', 'right' or 'full'")


import torch
from tensorly.decomposition import tucker
from tensorly import tenalg


# The GaLoreProjectorTensor class implements gradient projection for tensors with dimension > 2
# using Tucker decomposition from the tensorly library for low-rank approximation.
# This enables memory-efficient training by compressing high-dimensional gradients.
class GaLoreProjectorTensor:
    """
    A tensor projector for the GaLore algorithm that handles high-dimensional tensors.
    Uses Tucker decomposition for orthogonal matrix computation and supports tensors
    with more than 2 dimensions, extending beyond standard matrix operations.

    Args:
        rank (int): The rank of the projector for low-rank approximation
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        update_proj_gap (int, optional): Number of iterations between orthogonal matrix updates. Defaults to 200.
        scale (float, optional): Scaling factor for projected gradients. Defaults to 1.0.
    """

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.transformed_low_rank = None

    def project(self, full_rank_grad, iter):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)
        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(
            self.ortho_matrix, self.transformed_low_rank
        )
        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank_all):
        """
        Computes orthogonal matrix using Tucker decomposition for tensor SVD.
        This method handles high-dimensional tensors by applying Tucker decomposition
        to obtain a factorized representation suitable for low-rank approximation.

        Args:
            weights (torch.Tensor): The weight tensor to decompose (can be >2D)
            rank_all (int): The desired rank for each mode of the decomposition

        Returns:
            tuple: Tucker tensor containing core and factor matrices for orthogonal projection
        """
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
        tucker_tensor = tucker(matrix, rank=rank_all)
        return tucker_tensor

    def transform(self, tensor, x):
        """
        Transforms input tensor to low-rank representation using Tucker factors.
        Applies multi-mode tensor product with transposed factor matrices to project
        the tensor into a compressed subspace.

        Args:
            tensor (tuple): Tucker tensor containing core and factor matrices
            x (torch.Tensor): Input tensor to be transformed

        Returns:
            torch.Tensor: Transformed tensor in the low-rank subspace
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        """
        Inverse transforms tensor from low-rank back to original space using Tucker factors.
        Applies multi-mode tensor product with factor matrices (without transpose) to 
        reconstruct the tensor in its original dimensionality.

        Args:
            tensor (tuple): Tucker tensor containing core and factor matrices
            x (torch.Tensor): Low-rank tensor to be inverse transformed

        Returns:
            torch.Tensor: Reconstructed tensor in the original space
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors)


class DEFT_GaLoreAdamW(Optimizer):
    """
    DEFT-Enhanced GaLore AdamW Optimizer with gradient projection capabilities.
    
    This optimizer combines the Adam algorithm with weight decay fix and gradient projection
    techniques from GaLore (Gradient Low-Rank Projection). It supports both standard matrix
    projections and tensor projections for high-dimensional parameters, enabling memory-efficient
    training of large models through gradient compression.

    Key Features:
    - Integrates DEFT (Distributed Energy-based Frequency Transform) projection
    - Supports both 2D matrix and high-dimensional tensor gradient projections  
    - Maintains separate projectors for different parameter groups
    - Includes bias correction and weight decay regularization

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step with optional gradient projection.
        
        This method implements the core DEFT-GaLore optimization algorithm:
        1. Applies gradient projection (if rank is specified in parameter group)
        2. Computes Adam momentum updates with bias correction
        3. Projects gradients back to original space (for GaLore parameters)
        4. Applies parameter updates with weight decay
        
        The projection step reduces memory usage by operating in a low-rank subspace,
        while maintaining training effectiveness through careful reconstruction.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
            
        Returns:
            Optional[float]: The loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                if "dim" not in group:
                    group["dim"] = 2

                # Projection
                if "rank" in group:

                    if "projector" not in state:
                        if group["dim"] <= 2:
                            state["projector"] = DEFTProjector(
                                group["rank"],
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"],
                                proj_type=group["proj_type"],
                            )
                        else:
                            state["projector"] = GaLoreProjectorTensor(
                                group["rank"],
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"],
                                proj_type=group["proj_type"],
                            )
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


def load_model_and_tokenizer(model_name):
    """
    Loads and configures a pre-trained language model and its corresponding tokenizer.
    
    This function handles the initialization of transformer models with proper tokenizer
    configuration, including padding token setup and special handling for specific model types
    like Qwen models. It automatically distributes the model across available devices.
    
    Args:
        model_name (str): Name or path of the pre-trained model to load
        
    Returns:
        tuple: A tuple containing:
            - model (AutoModelForCausalLM): The loaded and configured language model
            - tokenizer (AutoTokenizer): The corresponding tokenizer with proper configuration
            
    Note:
        - Sets padding token to EOS token for proper sequence handling
        - Handles special token configuration for Qwen models
        - Resizes token embeddings if new tokens are added
        - Configures device mapping for multi-GPU setups
    """

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.config.pad_token_id = tokenizer.eos_token_id

    if model_name == "Qwen/Qwen2.5-1.5B-Instruct":
        if tokenizer.eos_token != "<|endoftext|>":
            tokenizer.eos_token = "<|endoftext|>"
            tokenizer.special_tokens_map["eos_token"] = "<|endoftext|>"

        # if a new token is added to the tokenizer, it is necessary to extend the model's embedding
        num_added = tokenizer.add_special_tokens({})
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# Function to save the best model
def save_best_model(model, tokenizer, epoch, best_loss, current_loss, save_path):
    """
    Saves model checkpoints and tracks the best performing model during training.
    
    This function implements a dual saving strategy:
    1. Saves current epoch model with epoch-specific naming
    2. Maintains a separate copy of the best model based on validation loss
    
    The best model is only updated when validation loss improves, ensuring that
    the final saved model represents the optimal training checkpoint.
    
    Args:
        model: The neural network model to save
        tokenizer: The tokenizer associated with the model
        epoch (int): Current training epoch number
        best_loss (float): Best validation loss achieved so far
        current_loss (float): Current epoch's validation loss
        save_path (str): Base directory path for saving models
        
    Returns:
        float: Updated best loss value (either unchanged or current_loss if improved)
        
    Side Effects:
        - Creates epoch-specific model directory and saves model/tokenizer
        - Updates best model directory if current loss is better
        - Prints progress messages to console
    """
    new_save_path = f"{save_path}_{epoch}"
    os.makedirs(new_save_path, exist_ok=True)
    model.save_pretrained(new_save_path)
    tokenizer.save_pretrained(new_save_path)
    print(
        f"Current model saved at epoch {epoch} with validation loss: {current_loss:.4f} in directory: {new_save_path}"
    )
    if current_loss < best_loss:
        best_loss = current_loss
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(
            f"Best model saved at epoch {epoch} with validation loss: {best_loss:.4f}"
        )
    return best_loss
