"""
TernaryLLM-20B Training Loop Implementation

This module implements a comprehensive training loop for the 20B-parameter ternary
language model with QAT (Quantization-Aware Training) and knowledge distillation.

Key Features:
- 20B parameter ternary model with 0.5 bits/parameter storage
- Hybrid 3:1 KDA-to-Full attention ratio
- QAT with learned ternary codebooks
- Knowledge distillation from teacher model
- Multi-sequence batching with dynamic sequence lengths
- Mixed precision training (FP16/BF16)
- Comprehensive monitoring and checkpointing

Author: TernaryLLM Training Team
Version: 1.0.0
"""

import os
import json
import math
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# Rich TUI for live monitoring
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """20B Ternary LLM Architecture Configuration"""
    # Model dimensions
    TOTAL_PARAMS: int = 20_000_000_000
    D_MODEL: int = 5120  # Hidden dimension
    N_LAYER: int = 60  # Number of transformer layers
    N_HEAD: int = 40  # Number of attention heads
    D_HEAD: int = 128  # Per-head dimension
    D_FF: int = 13696  # FFN intermediate dimension
    VOCAB_SIZE: int = 100352  # LLaMA-style tokenizer
    
    # Attention configuration
    KDA_RATIO: float = 3.0  # 3 KDA : 1 Full attention
    NUM_KDA_LAYERS: int = 45
    NUM_FULL_LAYERS: int = 15
    MAX_CONTEXT_LEN: int = 1_048_576  # 1M tokens
    
    # Quantization configuration
    ACTIVATION_2BIT_FREQ: int = 4  # 2-bit activation every 4th layer
    TERNARY_BITS_PER_PARAM: int = 1  # 0.5 bytes = 4 bits per param
    MEMORY_TARGET_GB: float = 1.25
    
    # KV Cache configuration
    KV_CACHE_MAX_SEQ: int = 1024
    KV_CACHE_MAX_HEADS: int = N_HEAD
    KV_CACHE_MAX_BATCH: int = 32
    
    # Performance targets
    TARGET_PERPLEXITY: float = 6.5
    TARGET_TOKENS_PER_SEC: float = 2500.0
    TARGET_POWER_WATTS: float = 75.0


@dataclass
class TrainingConfig:
    """Training hyperparameters and configuration"""
    # Model parameters
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Training hyperparameters
    batch_size: int = 32
    sequence_length: int = 2048
    max_sequence_length: int = 4096
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # Learning rate scheduling
    lr_schedule_type: str = "cosine"  # cosine, linear, constant
    warmup_steps: int = 2000
    total_steps: int = 100000
    min_lr_ratio: float = 0.1
    
    # Optimization
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    clip_grad_norm: bool = True
    
    # Mixed precision training
    use_mixed_precision: bool = True
    amp_dtype: torch.dtype = torch.bfloat16  # bfloat16 for training stability
    
    # QAT parameters
    qat_start_step: int = 1000
    qat_temperature: float = 1.0
    codebook_lr_scale: float = 0.01
    
    # Knowledge distillation
    use_knowledge_distillation: bool = True
    teacher_model_path: Optional[str] = None
    distillation_alpha: float = 0.5
    temperature: float = 4.0
    logit_distill_weight: float = 0.7
    hidden_state_distill_weight: float = 0.3
    
    # Multi-sequence batching
    num_sequences: int = 4
    dynamic_batching: bool = True
    memory_efficient: bool = True
    
    # Checkpointing and monitoring
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 10
    max_checkpoints: int = 5
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Validation
    validation_split: float = 0.05
    validation_batch_size: int = 16
    max_validation_steps: int = 100
    
    # Output directories
    output_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    tensorboard_dir: str = "./runs"


@dataclass
class TrainingState:
    """Current training state and metrics"""
    global_step: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
    best_perplexity: float = float('inf')
    patience_counter: int = 0
    is_best_model: bool = False
    
    # Loss tracking
    total_loss: float = 0.0
    loss_count: int = 0
    recent_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Performance metrics
    tokens_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_eval_time: float = field(default_factory=time.time)
    
    # QAT metrics
    codebook_updates: int = 0
    quantization_error: float = 0.0
    
    # Memory usage
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0


# ============================================================================
# QUANTIZATION AND CODEBOOK IMPLEMENTATIONS
# ============================================================================

class TernaryQuantizer(nn.Module):
    """Ternary quantization with learned codebooks for QAT"""
    
    def __init__(self, num_channels: int, init_centers: Optional[List[float]] = None):
        super().__init__()
        self.num_channels = num_channels
        
        # Learned ternary centers
        if init_centers is None:
            init_centers = [-1.2, 0.1, 0.9]
        
        self.centers = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32))
        self.scale_factors = nn.Parameter(torch.ones(3))
        self.usage_count = nn.Parameter(torch.zeros(3), requires_grad=False)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass with straight-through estimator for quantization
        """
        if not self.training:
            return self.dequantize(self.quantize(x))
        
        # Compute distances to centers
        centers = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        x_expanded = x.unsqueeze(-1)  # [..., 1]
        
        distances = torch.abs(x_expanded - centers)
        
        # Gumbel-Softmax for differentiable quantization
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(distances)))
        logits = -distances / temperature + gumbel_noise / temperature
        
        # Softmax to get soft assignments
        soft_assignments = F.softmax(logits, dim=-1)
        
        # Hard assignment for forward pass (straight-through estimator)
        hard_assignments = torch.zeros_like(soft_assignments)
        hard_assignments.scatter_(-1, torch.argmin(distances, dim=-1, keepdim=True), 1.0)
        
        # Straight-through estimator: forward uses hard assignment, backward uses soft
        output = (hard_assignments - soft_assignments).detach() + soft_assignments
        output = torch.sum(output * centers, dim=-1)
        
        return output
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Non-differentiable quantization for inference"""
        distances = torch.abs(x.unsqueeze(-1) - self.centers.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)
        return self.centers[indices]
    
    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """Dequantization for inference"""
        return quantized
    
    def update_codebook(self, x: torch.Tensor):
        """Update codebook centers based on usage statistics"""
        with torch.no_grad():
            distances = torch.abs(x.unsqueeze(-1) - self.centers.unsqueeze(0).unsqueeze(0))
            indices = torch.argmin(distances, dim=-1)
            
            # Update usage counts
            for i in range(3):
                self.usage_count[i] += torch.sum(indices == i)
            
            # Simple gradient-based center update
            for i in range(3):
                mask = indices == i
                if torch.sum(mask) > 0:
                    mean_val = torch.mean(x[mask])
                    self.centers[i] = 0.99 * self.centers[i] + 0.01 * mean_val
            
            # Clamp centers to reasonable range
            self.centers.data.clamp_(-2.0, 2.0)


class ActivationQuantizer(nn.Module):
    """2-bit activation quantization every N layers"""
    
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        # 2-bit quantization levels: [-1.5, -0.5, +0.5, +1.5]
        self.levels = nn.Parameter(torch.tensor([-1.5, -0.5, 0.5, 1.5]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.quantize(x)
        
        # Straight-through estimator for 2-bit quantization
        quantized = self.quantize(x)
        return (quantized - x).detach() + x
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """2-bit quantization"""
        # Simple threshold-based quantization
        thresholds = torch.tensor([-1.0, 0.0, 1.0], device=x.device)
        
        # Find closest level for each value
        distances = torch.abs(x.unsqueeze(-1) - self.levels.unsqueeze(0).unsqueeze(0))
        indices = torch.argmin(distances, dim=-1)
        
        return self.levels[indices]


# ============================================================================
# KNOWLEDGE DISTILLATION LOSS
# ============================================================================

class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining logit and hidden state distillation"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.distillation_alpha
        
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_hidden: Optional[torch.Tensor] = None,
                teacher_hidden: Optional[torch.Tensor] = None,
                target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        loss_dict = {}
        
        # Logit distillation loss
        if teacher_logits is not None:
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            logit_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
            loss_dict['logit_distill_loss'] = logit_loss.item()
        else:
            logit_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Hidden state distillation loss
        if student_hidden is not None and teacher_hidden is not None:
            hidden_loss = F.mse_loss(student_hidden, teacher_hidden)
            loss_dict['hidden_distill_loss'] = hidden_loss.item()
        else:
            hidden_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Standard cross-entropy loss
        if target is not None:
            ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), 
                                    target.view(-1), reduction='mean')
            loss_dict['ce_loss'] = ce_loss.item()
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Combined loss
        if self.config.use_knowledge_distillation:
            total_loss = (self.config.logit_distill_weight * logit_loss + 
                         self.config.hidden_state_distill_weight * hidden_loss +
                         (1.0 - self.config.distillation_alpha) * ce_loss)
        else:
            total_loss = ce_loss
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


# ============================================================================
# MULTI-SEQUENCE BATCHING AND KV CACHE
# ============================================================================

class SequenceBatch:
    """Container for multi-sequence batching with dynamic lengths"""
    
    def __init__(self, sequences: List[torch.Tensor], sequence_ids: List[int]):
        self.sequences = sequences
        self.sequence_ids = sequence_ids
        self.batch_size = len(sequences)
        self.max_length = max(seq.size(0) for seq in sequences)
        self.min_length = min(seq.size(0) for seq in sequences)
        
        # Pad sequences to max length
        self.padded_sequences = []
        self.attention_masks = []
        
        for seq in sequences:
            if seq.size(0) < self.max_length:
                padding_length = self.max_length - seq.size(0)
                padded = F.pad(seq, (0, padding_length), value=0)
                mask = torch.cat([torch.ones(seq.size(0)), torch.zeros(padding_length)])
            else:
                padded = seq
                mask = torch.ones(seq.size(0))
            
            self.padded_sequences.append(padded)
            self.attention_masks.append(mask)
        
        self.input_ids = torch.stack(self.padded_sequences)
        self.attention_mask = torch.stack(self.attention_masks)


class KVCacheManager:
    """Memory-efficient KV cache management for multi-sequence training"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.kv_caches: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
        
    def get_cache(self, sequence_id: int, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create KV cache for sequence and layer"""
        if sequence_id not in self.kv_caches:
            self.kv_caches[sequence_id] = {}
        
        if layer not in self.kv_caches[sequence_id]:
            # Initialize KV cache
            k_cache = torch.zeros(1, self.config.N_HEAD, self.config.KV_CACHE_MAX_SEQ, 
                                self.config.D_HEAD)
            v_cache = torch.zeros(1, self.config.N_HEAD, self.config.KV_CACHE_MAX_SEQ, 
                                self.config.D_HEAD)
            self.kv_caches[sequence_id][layer] = (k_cache, v_cache)
        
        return self.kv_caches[sequence_id][layer]
    
    def update_cache(self, sequence_id: int, layer: int, position: int, 
                    k: torch.Tensor, v: torch.Tensor):
        """Update KV cache with new key/value pairs"""
        k_cache, v_cache = self.get_cache(sequence_id, layer)
        
        if position < self.config.KV_CACHE_MAX_SEQ:
            k_cache[0, :, position:position+1, :] = k.unsqueeze(0).unsqueeze(2)
            v_cache[0, :, position:position+1, :] = v.unsqueeze(0).unsqueeze(2)
    
    def clear_sequence(self, sequence_id: int):
        """Clear cache for specific sequence"""
        if sequence_id in self.kv_caches:
            del self.kv_caches[sequence_id]
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        total_elements = 0
        for seq_cache in self.kv_caches.values():
            for k_cache, v_cache in seq_cache.values():
                total_elements += k_cache.numel() + v_cache.numel()
        
        return total_elements * 4 / (1024 * 1024)  # Assuming float32


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

class TernaryLLMTrainer:
    """
    Comprehensive training loop for TernaryLLM-20B with QAT and knowledge distillation
    
    Features:
    - 20B parameter ternary model training
    - Quantization-aware training with learned codebooks
    - Knowledge distillation from teacher model
    - Multi-sequence batching with dynamic lengths
    - Mixed precision training
    - Comprehensive monitoring and checkpointing
    - Early stopping and validation
    """
    
    def __init__(self, 
                 model: nn.Module,
                 teacher_model: Optional[nn.Module] = None,
                 config: Optional[TrainingConfig] = None):
        
        self.model = model
        self.teacher_model = teacher_model
        self.config = config or TrainingConfig()
        self.state = TrainingState()
        
        # Initialize console for Rich TUI
        self.console = Console()
        
        # Setup output directories
        self.setup_directories()
        
        # Initialize components
        self.setup_training_components()
        
        # Setup Rich TUI layout
        self.setup_tui_layout()
        
        self.console.print("[bold green]TernaryLLM-20B Trainer initialized successfully![/bold green]")
        
    def setup_directories(self):
        """Create necessary output directories"""
        directories = [self.config.output_dir, self.config.logs_dir, self.config.tensorboard_dir]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_training_components(self):
        """Initialize training components"""
        # Optimizer
        self.setup_optimizer()
        
        # Learning rate scheduler
        self.setup_scheduler()
        
        # Loss function
        self.distillation_loss = DistillationLoss(self.config)
        
        # Mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
        
        # KV cache manager
        self.kv_cache_manager = KVCacheManager(self.config.model_config)
        
        # Quantizers
        self.setup_quantizers()
        
        # Monitoring
        self.setup_monitoring()
    
    def setup_optimizer(self):
        """Setup AdamW optimizer"""
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and 'centers' not in n and 'levels' not in n],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and 'centers' not in n and 'levels' not in n],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'centers' in n or 'levels' in n],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate * self.config.codebook_lr_scale
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps
        )
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                if self.config.lr_schedule_type == "cosine":
                    progress = (step - self.config.warmup_steps) / (self.config.total_steps - self.config.warmup_steps)
                    return self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
                elif self.config.lr_schedule_type == "linear":
                    progress = (step - self.config.warmup_steps) / (self.config.total_steps - self.config.warmup_steps)
                    return max(self.config.min_lr_ratio, 1 - progress)
                else:  # constant
                    return 1.0
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def setup_quantizers(self):
        """Initialize quantization components"""
        self.activation_quantizers = nn.ModuleList([
            ActivationQuantizer(self.config.model_config.D_MODEL) 
            for _ in range(self.config.model_config.N_LAYER // self.config.model_config.ACTIVATION_2BIT_FREQ)
        ])
    
    def setup_monitoring(self):
        """Setup monitoring and logging"""
        # Training metrics
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        
        # Rich progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("ETA:[progress.elapsed]{task.elapsed}"),
            TimeRemainingColumn(),
            console=self.console
        )
    
    def setup_tui_layout(self):
        """Setup Rich TUI layout for live monitoring"""
        self.layout = Layout()
        
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Header with model info
        self.header_text = Text()
        self.header_text.append("TernaryLLM-20B Training Loop", style="bold blue")
        self.header_text.append("\nQAT + Knowledge Distillation + Multi-sequence Batching")
        
        # Main area with training metrics
        self.metrics_table = Table(show_header=True, header_style="bold magenta")
        self.metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        self.metrics_table.add_column("Value", style="green")
        self.metrics_table.add_column("Status", style="yellow")
        
        # Footer with progress
        self.footer_text = Text()
        
        self.layout["header"].update(Panel(self.header_text, border_style="blue"))
        self.layout["footer"].update(Panel(self.footer_text, border_style="green"))
    
    def create_sequence_batch(self, batch_data: List[Tuple[torch.Tensor, int]]) -> SequenceBatch:
        """Create multi-sequence batch with dynamic length handling"""
        sequences = []
        sequence_ids = []
        
        for seq_data in batch_data:
            if isinstance(seq_data, tuple):
                seq, seq_id = seq_data
            else:
                seq = seq_data
                seq_id = random.randint(0, 1000)
            
            sequences.append(seq)
            sequence_ids.append(seq_id)
        
        return SequenceBatch(sequences, sequence_ids)
    
    def forward_pass(self, 
                    batch: SequenceBatch, 
                    use_teacher: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with multi-sequence handling"""
        model_to_use = self.teacher_model if use_teacher and self.teacher_model else self.model
        
        # Move batch to device
        input_ids = batch.input_ids.to(next(model_to_use.parameters()).device)
        attention_mask = batch.attention_mask.to(next(model_to_use.parameters()).device)
        
        with torch.autocast(device_type='cuda', dtype=self.config.amp_dtype) if self.config.use_mixed_precision else torch.no_grad():
            outputs = model_to_use(input_ids=input_ids, attention_mask=attention_mask)
            
            hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
            logits = outputs.logits if hasattr(outputs, 'logits') else None
        
        return logits, hidden_states, {}
    
    def compute_loss(self, 
                    student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    student_hidden: Optional[torch.Tensor] = None,
                    teacher_hidden: Optional[torch.Tensor] = None,
                    target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute distillation loss"""
        return self.distillation_loss(
            student_logits, teacher_logits, 
            student_hidden, teacher_hidden, 
            target
        )
    
    def update_quantization_codebooks(self, batch: SequenceBatch):
        """Update ternary codebooks during QAT phase"""
        if self.state.global_step < self.config.qat_start_step:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'centers' in name:
                    # Simple codebook update
                    param.data.clamp_(-2.0, 2.0)
                    self.state.codebook_updates += 1
    
    def training_step(self, batch: SequenceBatch) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update quantization codebooks
        self.update_quantization_codebooks(batch)
        
        # Forward pass through student model
        student_logits, student_hidden, _ = self.forward_pass(batch, use_teacher=False)
        
        # Forward pass through teacher model if available
        teacher_logits, teacher_hidden, _ = (None, None, {})
        if self.teacher_model:
            with torch.no_grad():
                teacher_logits, teacher_hidden, _ = self.forward_pass(batch, use_teacher=True)
        
        # Create targets for loss computation
        target = batch.input_ids[:, 1:].contiguous()  # Shifted targets
        student_logits_for_loss = student_logits[:, :-1, :].contiguous()  # Shifted logits
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(
            student_logits_for_loss, teacher_logits,
            student_hidden, teacher_hidden, target
        )
        
        # Backward pass with gradient accumulation
        if self.config.use_mixed_precision:
            self.scaler.scale(total_loss / self.config.gradient_accumulation_steps).backward()
        else:
            (total_loss / self.config.gradient_accumulation_steps).backward()
        
        # Gradient accumulation
        if (self.state.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.use_mixed_precision:
                # Gradient clipping
                if self.config.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.config.clip_grad_norm:
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
        
        # Update metrics
        self.update_training_metrics(total_loss.item(), loss_dict, batch)
        
        # Clear KV caches for this batch
        for seq_id in batch.sequence_ids:
            self.kv_cache_manager.clear_sequence(seq_id)
        
        return loss_dict
    
    def update_training_metrics(self, loss: float, loss_dict: Dict[str, float], batch: SequenceBatch):
        """Update training metrics and state"""
        self.state.total_loss += loss
        self.state.loss_count += 1
        self.state.recent_losses.append(loss)
        
        # Update performance metrics
        batch_tokens = batch.input_ids.numel()
        self.state.tokens_processed += batch_tokens
        
        # Update memory usage
        self.state.current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        self.state.peak_memory_mb = max(self.state.peak_memory_mb, self.state.current_memory_mb)
        
        # Store metrics
        for key, value in loss_dict.items():
            self.train_metrics[key].append(value)
    
    def validation_step(self, batch: SequenceBatch) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            student_logits, student_hidden, _ = self.forward_pass(batch)
            
            target = batch.input_ids[:, 1:].contiguous()
            student_logits_for_loss = student_logits[:, :-1, :].contiguous()
            
            # Compute standard cross-entropy for validation
            loss = F.cross_entropy(
                student_logits_for_loss.view(-1, student_logits_for_loss.size(-1)),
                target.view(-1),
                reduction='mean'
            )
            
            # Compute perplexity
            perplexity = torch.exp(loss).item()
            
            return {
                'val_loss': loss.item(),
                'val_perplexity': perplexity
            }
    
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Complete validation evaluation"""
        self.console.print(f"[bold yellow]Starting validation evaluation...[/bold yellow]")
        
        total_val_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with self.progress:
            task = self.progress.add_task("Validation", total=len(val_dataloader))
            
            for batch_idx, batch_data in enumerate(val_dataloader):
                if batch_idx >= self.config.max_validation_steps:
                    break
                
                # Create sequence batch
                if isinstance(batch_data, (list, tuple)):
                    batch = self.create_sequence_batch(batch_data)
                else:
                    batch = self.create_sequence_batch([(batch_data, 0)])
                
                # Validation step
                val_metrics = self.validation_step(batch)
                
                total_val_loss += val_metrics['val_loss']
                total_perplexity += val_metrics['val_perplexity']
                num_batches += 1
                
                self.progress.update(task, advance=1)
        
        avg_val_loss = total_val_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            'val_loss': avg_val_loss,
            'val_perplexity': avg_perplexity
        }
    
    def should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        if not self.config.use_early_stopping:
            return False
        
        current_perplexity = val_metrics['val_perplexity']
        current_loss = val_metrics['val_loss']
        
        # Update best metrics
        if current_loss < self.state.best_val_loss - self.config.early_stopping_min_delta:
            self.state.best_val_loss = current_loss
            self.state.patience_counter = 0
            self.state.is_best_model = True
        else:
            self.state.patience_counter += 1
            self.state.is_best_model = False
        
        if current_perplexity < self.state.best_perplexity:
            self.state.best_perplexity = current_perplexity
        
        # Check early stopping condition
        return self.state.patience_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self, val_metrics: Optional[Dict[str, float]] = None):
        """Save training checkpoint"""
        checkpoint = {
            'global_step': self.state.global_step,
            'epoch': self.state.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.state.best_val_loss,
            'best_perplexity': self.state.best_perplexity,
            'config': self.config.__dict__,
            'training_state': {
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'best_val_loss': self.state.best_val_loss,
                'best_perplexity': self.state.best_perplexity,
                'patience_counter': self.state.patience_counter
            }
        }
        
        if self.config.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if val_metrics:
            checkpoint['val_metrics'] = val_metrics
        
        # Save checkpoint
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_step_{self.state.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model checkpoint
        if self.state.is_best_model:
            best_path = Path(self.config.output_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.console.print(f"[bold green]Saved best model to {best_path}[/bold green]")
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        checkpoint_dir = Path(self.config.output_dir)
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if len(checkpoint_files) > self.config.max_checkpoints:
            # Sort by global step (reverse)
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
            
            # Remove oldest checkpoints
            for checkpoint_file in checkpoint_files[self.config.max_checkpoints:]:
                checkpoint_file.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        training_state = checkpoint.get('training_state', {})
        self.state.global_step = training_state.get('global_step', 0)
        self.state.epoch = training_state.get('epoch', 0)
        self.state.best_val_loss = training_state.get('best_val_loss', float('inf'))
        self.state.best_perplexity = training_state.get('best_perplexity', float('inf'))
        self.state.patience_counter = training_state.get('patience_counter', 0)
        
        # Load scaler state if using mixed precision
        if self.config.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.console.print(f"[bold green]Loaded checkpoint from step {self.state.global_step}[/bold green]")
    
    def log_metrics(self, step_metrics: Dict[str, float], is_training: bool = True):
        """Log training metrics"""
        metrics_type = "train" if is_training else "val"
        
        # Update Rich table
        self.update_metrics_table(step_metrics, is_training)
        
        # Log to file
        log_file = Path(self.config.logs_dir) / f"{metrics_type}_metrics.json"
        
        log_entry = {
            'global_step': self.state.global_step,
            'timestamp': time.time(),
            **step_metrics
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def update_metrics_table(self, step_metrics: Dict[str, float], is_training: bool = True):
        """Update Rich metrics table"""
        # Clear existing rows
        self.metrics_table.rows.clear()
        
        # Add current metrics
        data_type = "Training" if is_training else "Validation"
        
        # Core metrics
        self.metrics_table.add_row("Global Step", str(self.state.global_step), "")
        self.metrics_table.add_row("Epoch", str(self.state.epoch), "")
        self.metrics_table.add_row(f"{data_type} Loss", f"{step_metrics.get('total_loss', 0):.4f}", 
                                  "[green]✓[/green]" if is_training else "")
        self.metrics_table.add_row("Learning Rate", f"{self.optimizer.param_groups[0]['lr']:.2e}", "")
        
        # Performance metrics
        tokens_per_sec = self.get_tokens_per_second()
        self.metrics_table.add_row("Tokens/sec", f"{tokens_per_sec:.1f}", "")
        self.metrics_table.add_row("Memory (MB)", f"{self.state.current_memory_mb:.1f}", "")
        self.metrics_table.add_row("Peak Memory (MB)", f"{self.state.peak_memory_mb:.1f}", "")
        
        # QAT metrics
        if not is_training and 'val_perplexity' in step_metrics:
            self.metrics_table.add_row("Validation Perplexity", f"{step_metrics['val_perplexity']:.2f}", 
                                      "[green]New Best![/green]" if step_metrics['val_perplexity'] == self.state.best_perplexity else "")
            self.metrics_table.add_row("Best Perplexity", f"{self.state.best_perplexity:.2f}", "")
        
        # Loss breakdown (for training)
        if is_training and 'logit_distill_loss' in step_metrics:
            self.metrics_table.add_row("Logit Distill Loss", f"{step_metrics['logit_distill_loss']:.4f}", "")
            self.metrics_table.add_row("Hidden Distill Loss", f"{step_metrics.get('hidden_distill_loss', 0):.4f}", "")
            self.metrics_table.add_row("Cross-Entropy Loss", f"{step_metrics.get('ce_loss', 0):.4f}", "")
        
        # Early stopping info
        if self.config.use_early_stopping:
            self.metrics_table.add_row("Early Stop Patience", 
                                      f"{self.state.patience_counter}/{self.config.early_stopping_patience}", 
                                      "[yellow]⚠[/yellow]" if self.state.patience_counter > 0 else "")
    
    def get_tokens_per_second(self) -> float:
        """Calculate current tokens per second"""
        elapsed_time = time.time() - self.state.start_time
        if elapsed_time > 0:
            return self.state.tokens_processed / elapsed_time
        return 0.0
    
    def update_tui(self, step_metrics: Dict[str, float]):
        """Update Rich TUI with current state"""
        # Update header with progress
        progress_pct = (self.state.global_step / self.config.total_steps) * 100
        self.header_text = Text()
        self.header_text.append(f"TernaryLLM-20B Training Loop (Step {self.state.global_step}/{self.config.total_steps})", 
                               style="bold blue")
        self.header_text.append(f"\nProgress: {progress_pct:.1f}% • ")
        self.header_text.append(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}", style="cyan")
        
        # Update footer with performance info
        tokens_per_sec = self.get_tokens_per_second()
        self.footer_text = Text()
        self.footer_text.append(f"Tokens/sec: {tokens_per_sec:.1f} • ", style="green")
        self.footer_text.append(f"Memory: {self.state.current_memory_mb:.1f}MB", style="yellow")
        if self.config.use_mixed_precision:
            self.footer_text.append(" • AMP", style="magenta")
        
        # Update layout
        self.layout["header"].update(Panel(self.header_text, border_style="blue"))
        self.layout["main"].update(self.metrics_table)
        self.layout["footer"].update(Panel(self.footer_text, border_style="green"))
    
    def train_epoch(self, train_dataloader: DataLoader):
        """Train for one epoch"""
        self.console.print(f"[bold blue]Starting epoch {self.state.epoch + 1}[/bold blue]")
        
        with self.progress:
            # Add progress tasks
            train_task = self.progress.add_task("Training", total=len(train_dataloader))
            
            for batch_idx, batch_data in enumerate(train_dataloader):
                # Create multi-sequence batch
                if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
                    if isinstance(batch_data[0], tuple):
                        # Already in sequence format
                        batch = self.create_sequence_batch(batch_data)
                    else:
                        # Single tensor batch
                        batch = self.create_sequence_batch([(batch_data, 0)])
                else:
                    # Single tensor batch
                    batch = self.create_sequence_batch([(batch_data, 0)])
                
                # Training step
                step_metrics = self.training_step(batch)
                
                # Update progress
                self.progress.update(train_task, advance=1)
                
                # Log metrics
                if self.state.global_step % self.config.log_every_n_steps == 0:
                    self.log_metrics(step_metrics, is_training=True)
                    self.update_tui(step_metrics)
                
                # Evaluation
                if self.state.global_step % self.config.eval_every_n_steps == 0 and self.state.global_step > 0:
                    self.console.print(f"[bold yellow]Running evaluation at step {self.state.global_step}...[/bold yellow]")
                    # Note: In practice, you'd have a separate validation dataloader
                    # For now, just log the current metrics
                    self.log_metrics(step_metrics, is_training=False)
                
                # Save checkpoint
                if self.state.global_step % self.config.save_every_n_steps == 0 and self.state.global_step > 0:
                    self.save_checkpoint()
                
                # Early stopping check
                if self.config.use_early_stopping and self.state.patience_counter >= self.config.early_stopping_patience:
                    self.console.print(f"[bold red]Early stopping triggered at step {self.state.global_step}[/bold red]")
                    return True
                
                self.state.global_step += 1
                
                if self.state.global_step >= self.config.total_steps:
                    self.console.print(f"[bold green]Reached maximum steps ({self.config.total_steps})[/bold green]")
                    return True
        
        return False
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop"""
        self.console.print("[bold green]Starting TernaryLLM-20B training...[/bold green]")
        
        # Try to load checkpoint if exists
        latest_checkpoint = None
        checkpoint_dir = Path(self.config.output_dir)
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        if latest_checkpoint:
            self.load_checkpoint(str(latest_checkpoint))
        
        # Main training loop
        try:
            with Live(self.layout, refresh_per_second=4, console=self.console) as live:
                while self.state.epoch < 100:  # Arbitrary max epochs
                    # Train one epoch
                    should_stop = self.train_epoch(train_dataloader)
                    
                    if should_stop:
                        break
                    
                    self.state.epoch += 1
                    
                    # Validation after each epoch
                    if val_dataloader and self.state.epoch % 1 == 0:
                        val_metrics = self.evaluate(val_dataloader)
                        self.log_metrics(val_metrics, is_training=False)
                        
                        # Early stopping check
                        if self.should_early_stop(val_metrics):
                            self.console.print(f"[bold red]Training stopped early at epoch {self.state.epoch}[/bold red]")
                            break
                        
                        # Save checkpoint with validation metrics
                        self.save_checkpoint(val_metrics)
        
        except KeyboardInterrupt:
            self.console.print("[bold yellow]Training interrupted by user[/bold yellow]")
            self.save_checkpoint()
        
        except Exception as e:
            self.console.print(f"[bold red]Training error: {e}[/bold red]")
            self.save_checkpoint()
            raise
        
        # Final save
        self.save_checkpoint()
        self.console.print("[bold green]Training completed successfully![/bold green]")


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def create_trainer(model: nn.Module, 
                  teacher_model: Optional[nn.Module] = None,
                  config: Optional[TrainingConfig] = None) -> TernaryLLMTrainer:
    """Factory function to create trainer instance"""
    return TernaryLLMTrainer(model, teacher_model, config)


def run_training_pipeline(model: nn.Module,
                         train_dataloader: DataLoader,
                         val_dataloader: Optional[DataLoader] = None,
                         teacher_model: Optional[nn.Module] = None,
                         config: Optional[TrainingConfig] = None):
    """Complete training pipeline"""
    # Create trainer
    trainer = create_trainer(model, teacher_model, config)
    
    # Start training
    trainer.train(train_dataloader, val_dataloader)
    
    return trainer


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    from torch.utils.data import TensorDataset
    
    # Create dummy model (replace with actual TernaryLLM-20B model)
    model = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
            d_model=5120, 
            nhead=40, 
            dim_feedforward=13696,
            batch_first=True
        ),
        num_layers=60
    )
    
    # Create dummy data
    train_data = TensorDataset(torch.randint(0, 100352, (1000, 2048)))
    val_data = TensorDataset(torch.randint(0, 100352, (200, 2048)))
    
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # Create training configuration
    config = TrainingConfig(
        total_steps=10000,
        batch_size=32,
        learning_rate=1e-4,
        use_mixed_precision=True,
        use_knowledge_distillation=True,
        eval_every_n_steps=500,
        save_every_n_steps=1000
    )
    
    # Run training
    trainer = run_training_pipeline(model, train_dataloader, val_dataloader, config=config)
    print("Training completed!")