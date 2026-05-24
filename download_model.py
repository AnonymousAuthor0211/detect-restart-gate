#!/usr/bin/env python3
"""
Download models from Hugging Face and save to base_llms folder
"""

import os
import argparse
import logging
from pathlib import Path


# Try to suppress torchvision import issues
try:
    import torchvision
except (ImportError, RuntimeError):
    pass

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if accelerate is available for device_map
try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

# Try importing transformers, fallback to huggingface_hub if there are import issues
try:
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except (ImportError, RuntimeError, ModuleNotFoundError) as e:
    logger.warning(f"Transformers import failed: {e}. Will use huggingface_hub fallback if available.")
    HAS_TRANSFORMERS = False

# Fallback: use huggingface_hub for direct file download
try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

def download_model(model_name: str, output_dir: str):
    """
    Download a model from Hugging Face and save it locally.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-Math-7B-Instruct")
        output_dir: Local directory to save the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Output directory: {output_path.absolute()}")
    
    try:
        # If transformers has import issues, use huggingface_hub fallback
        if not HAS_TRANSFORMERS and HAS_HF_HUB:
            logger.info("Using huggingface_hub snapshot_download (bypassing model loading)...")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(output_path),
                local_dir_use_symlinks=False
            )
            logger.info("Model files downloaded successfully")
            return
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)
        logger.info("Tokenizer downloaded and saved")
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        
        # Prepare common kwargs
        load_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        
        # Only use device_map if accelerate is available
        if HAS_ACCELERATE:
            load_kwargs["device_map"] = "auto"
        else:
            logger.info("Accelerate not available, loading without device_map")
        
        # Try AutoModelForCausalLM first (for standard language models)
        try:
            # Try with eager attention first
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    attn_implementation="eager",  # Use eager attention for V100 compatibility
                    **load_kwargs
                )
                logger.info("Loaded as AutoModelForCausalLM with eager attention")
            except (TypeError, ValueError) as attn_e:
                # If attn_implementation fails, try without it
                if "attn_implementation" in str(attn_e) or "Unrecognized" in str(attn_e):
                    logger.info("Eager attention not supported, trying without attn_implementation...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **load_kwargs
                    )
                    logger.info("Loaded as AutoModelForCausalLM")
                else:
                    raise
        except (ValueError, TypeError, RuntimeError, ModuleNotFoundError) as e:
            # Check error message and exception chain for torchvision/import issues
            error_str = str(e)
            error_chain = []
            current_exc = e
            while current_exc:
                error_chain.append(str(current_exc))
                if hasattr(current_exc, '__cause__') and current_exc.__cause__:
                    current_exc = current_exc.__cause__
                elif hasattr(current_exc, '__context__') and current_exc.__context__:
                    current_exc = current_exc.__context__
                else:
                    break
            
            # Check if this is a torchvision/import related error
            is_import_error = any(keyword in err for err in error_chain for keyword in [
                "Unrecognized configuration class", 
                "for this kind of AutoModel", 
                "Could not import module",
                "Could not find",
                "torchvision::nms",
                "operator torchvision::nms does not exist"
            ])
            
            if is_import_error:
                # If transformers import fails, try huggingface_hub fallback
                if HAS_HF_HUB:
                    logger.info("Transformers loading failed due to import/torchvision issues, using huggingface_hub fallback...")
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=str(output_path),
                        local_dir_use_symlinks=False
                    )
                    logger.info("Model files downloaded successfully via fallback")
                    return
                else:
                    raise RuntimeError("Transformers failed to load model and huggingface_hub is not available. Please install: pip install huggingface_hub")
            else:
                # Try AutoModel as fallback for other errors
                logger.info("Causal model loading failed, trying AutoModel...")
                try:
                    model = AutoModel.from_pretrained(
                        model_name,
                        attn_implementation="eager",
                        **load_kwargs
                    )
                    logger.info("Loaded as AutoModel with eager attention")
                except (TypeError, ValueError, RuntimeError) as auto_e:
                    # If AutoModel also fails with import errors, use huggingface_hub
                    auto_error_str = str(auto_e)
                    if HAS_HF_HUB and any(keyword in auto_error_str for keyword in [
                        "Could not import", "Could not find", "torchvision"
                    ]):
                        logger.info("AutoModel also failed, using huggingface_hub fallback...")
                        snapshot_download(
                            repo_id=model_name,
                            local_dir=str(output_path),
                            local_dir_use_symlinks=False
                        )
                        logger.info("Model files downloaded successfully via fallback")
                        return
                    # If attn_implementation is not supported, load without it
                    try:
                        model = AutoModel.from_pretrained(
                            model_name,
                            **load_kwargs
                        )
                        logger.info("Loaded as AutoModel")
                    except Exception:
                        raise
        
        # Only save if we loaded a model object (not if we used fallback)
        if 'model' in locals():
            model.save_pretrained(output_path)
            logger.info("Model downloaded and saved")
        
        logger.info(f"Model successfully downloaded to: {output_path.absolute()}")
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        # Last resort: try huggingface_hub if available
        error_str = str(e)
        error_chain = []
        current_exc = e
        while current_exc:
            error_chain.append(str(current_exc))
            if hasattr(current_exc, '__cause__') and current_exc.__cause__:
                current_exc = current_exc.__cause__
            elif hasattr(current_exc, '__context__') and current_exc.__context__:
                current_exc = current_exc.__context__
            else:
                break
        
        should_fallback = HAS_HF_HUB and any(
            keyword in err for err in error_chain for keyword in [
                "torchvision", "Could not import", "Could not find",
                "operator torchvision::nms", "ModuleNotFoundError"
            ]
        )
        
        if should_fallback:
            logger.info("Attempting fallback to huggingface_hub snapshot_download...")
            try:
                snapshot_download(
                    repo_id=model_name,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False
                )
                logger.info("Model files downloaded successfully via fallback")
                return
            except Exception as fallback_e:
                logger.error(f"Fallback also failed: {fallback_e}")
        raise

def get_local_model_name(hf_model_name: str) -> str:
    """
    Convert Hugging Face model name to local directory name.
    
    Args:
        hf_model_name: Hugging Face model identifier
        
    Returns:
        Local directory name
    """
    # Remove the organization prefix (e.g., "Qwen/")
    if "/" in hf_model_name:
        return hf_model_name.split("/")[-1]
    return hf_model_name

def parse_args():
    parser = argparse.ArgumentParser(description="Download Hugging Face models for the standalone DRG/SC pipeline")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Hugging Face model id to download. Repeat for multiple models.",
    )
    parser.add_argument(
        "--models_file",
        type=Path,
        default=None,
        help="Optional text file with one Hugging Face model id per line.",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).parent / "base_llms",
        help="Base output directory for downloaded models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = list(args.model)
    if args.models_file is not None:
        for line in args.models_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                models.append(line)
    if not models:
        models = ["mistralai/Ministral-3-14B-Reasoning-2512"]

    base_llms_dir = args.base_dir
    base_llms_dir.mkdir(parents=True, exist_ok=True)

    for idx, model_name in enumerate(models, start=1):
        local_name = get_local_model_name(model_name)
        output_dir = base_llms_dir / local_name

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model {model_name} ({idx}/{len(models)})")
        logger.info(f"{'='*60}\n")

        download_model(model_name, str(output_dir))

        logger.info(f"\nCompleted download of {model_name}\n")

    logger.info(f"All models downloaded successfully to: {base_llms_dir.absolute()}")
