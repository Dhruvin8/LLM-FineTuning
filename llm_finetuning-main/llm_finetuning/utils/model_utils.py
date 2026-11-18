"""
Model utilities for LLM fine-tuning with LoRA.
"""

import torch
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    model_name: str,
    max_length: int = 512,
    use_cache: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> tuple:
    """
    Setup model and tokenizer for fine-tuning.
    
    Args:
        model_name: Name of the model to load
        max_length: Maximum sequence length
        use_cache: Whether to use KV cache
        load_in_8bit: Whether to load model in 8-bit
        load_in_4bit: Whether to load model in 4-bit
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Setup quantization config if needed
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    # Configure model
    model.config.use_cache = use_cache
    model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info(f"Model loaded successfully. Parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def setup_lora_model(
    model,
    lora_config: Dict[str, Any]
) -> Any:
    """
    Setup LoRA configuration and apply it to the model.
    
    Args:
        model: Base model to apply LoRA to
        lora_config: LoRA configuration dictionary
        
    Returns:
        Model with LoRA applied
    """
    logger.info("Setting up LoRA configuration")
    
    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def get_model_info(model) -> Dict[str, Any]:
    """
    Get information about the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary containing model information
    """
    info = {
        "num_parameters": model.num_parameters(),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
    }
    
    # Add LoRA info if available
    if hasattr(model, "peft_config"):
        info["lora_config"] = model.peft_config
        info["lora_enabled"] = True
    else:
        info["lora_enabled"] = False
    
    return info


def save_model_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    checkpoint_name: str = "checkpoint"
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        checkpoint_name: Name of the checkpoint
    """
    import os
    
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(checkpoint_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_path)
    
    logger.info(f"Model checkpoint saved to: {checkpoint_path}")


def load_model_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None
) -> tuple:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import PeftModel
    
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Load LoRA weights if available
    try:
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info("LoRA weights loaded successfully")
    except:
        model = base_model
        logger.info("No LoRA weights found, using base model")
    
    return model, tokenizer
