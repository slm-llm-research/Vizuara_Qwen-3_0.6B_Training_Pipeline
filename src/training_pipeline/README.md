# Qwen3-0.6B Training Pipeline for AI/Human Text Detection

This directory contains a generic training pipeline for fine-tuning Qwen3-0.6B on AI/Human text detection datasets.

## Table of Contents

- [Files](#files)
- [File Status](#file-status)
- [Quick Start](#quick-start)
  - [Option 1: Using the Standalone Script](#option-1-using-the-standalone-script-recommended)
  - [Option 2: Using the Jupyter Notebook](#option-2-using-the-jupyter-notebook)
- [Dataset Information](#dataset-information)
- [Key Features](#key-features)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Modifying Hyperparameters](#modifying-hyperparameters)
- [Running in Different Environments](#running-in-different-environments)
- [Requirements](#requirements)
- [WandB Configuration](#wandb-configuration)
- [Training Time Estimates](#training-time-estimates)
- [Choosing Between Notebook and Script](#choosing-between-notebook-and-script)
- [Troubleshooting](#troubleshooting)
- [Notebook Best Practices](#notebook-best-practices)
- [Next Steps After Training](#next-steps-after-training)
- [Additional Resources](#additional-resources)

## Files

- `config.yaml` - Configuration file with all training parameters
- `train_qwen.py` - **Class-based refactored script with CLI (RECOMMENDED)**
- `train_qwen_standalone.py` - Original standalone script (proven, simple)
- `train_qwen_generic.ipynb` - Jupyter notebook version (partially complete)
- `requirements.txt` - Python package dependencies
- `README.md` - This file
- `QUICKSTART.md` - Fast-start guide
- `ARCHITECTURE.md` - Architecture comparison and details
- `PIPELINE_SUMMARY.md` - Overview and summary

## File Status

| File | Status | Architecture | Lines | Completeness | Recommended For |
|------|--------|--------------|-------|--------------|-----------------|
| `train_qwen.py` | ✅ Complete | **Class-Based** | ~1200 | 100% | **Production, CLI** |
| `train_qwen_standalone.py` | ✅ Complete | Procedural | 934 | 100% | Quick Start, Simple |
| `train_qwen_generic.ipynb` | ⚠️ Partial | Interactive | 300 | ~40% | Learning |

**New!** `train_qwen.py` is a completely refactored version with:
- Clean class-based architecture
- Full CLI argument support
- Modular components (easy to test/extend)
- Same functionality as standalone script
- Professional code structure

See `ARCHITECTURE.md` for detailed comparison.

## Quick Start

### Option 1: Using the Refactored Script with CLI (RECOMMENDED)

**Status**: Complete and production-ready

**Architecture**: Class-based with full CLI support

**Best for**: Production training, RunPod, flexibility, no code editing needed

```bash
# Navigate to directory
cd /Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Vizuara_Qwen-3_0.6B_Training_Pipeline/src/training_pipeline

# Install dependencies
pip install -r requirements.txt

# Basic usage - select dataset via CLI
python3 train_qwen.py --dataset 10k

# With custom hyperparameters - NO CODE EDITING NEEDED!
python3 train_qwen.py --dataset 1M --epochs 5 --batch-size 16 --learning-rate 2e-4

# Disable WandB
python3 train_qwen.py --dataset 100k --no-wandb

# Resume from checkpoint
python3 train_qwen.py --dataset 10k --resume ./output/qwen3-0.6b-10k/checkpoint-1000

# Run in background
nohup python3 train_qwen.py --dataset 1M > training.log 2>&1 &
```

**CLI Arguments**:
- `--dataset`: Dataset size (10k, 100k, 1M, 2M) - **REQUIRED**
- `--epochs`: Number of epochs (overrides config)
- `--batch-size`: Batch size per device (overrides config)
- `--learning-rate`: Learning rate (overrides config)
- `--lora-rank`: LoRA rank (overrides config)
- `--no-wandb`: Disable WandB logging
- `--resume`: Path to checkpoint to resume from
- `--config`: Custom config file path

**What it includes**:
- ✅ Clean class-based architecture (12 modular methods)
- ✅ Full CLI argument parsing
- ✅ Complete training pipeline (data loading to model saving)
- ✅ WandB integration with pre-configured API key
- ✅ Automatic class weight calculation
- ✅ Pre and post-training evaluation
- ✅ Progress tracking and detailed logging
- ✅ Smart caching for faster subsequent runs
- ✅ Comprehensive error handling
- ✅ Easy to test and extend
- ✅ Can be imported as a module

**Advantages**:
- 🚀 No code editing required - all parameters via CLI
- 🧩 Modular design - can test/reuse individual components
- 🔧 Flexible - override any parameter without editing config
- 📦 Importable - use in other Python scripts
- 🎯 Professional code structure

### Option 2: Using the Original Standalone Script (Simple Alternative)

**Status**: Complete and proven

**Architecture**: Procedural (single main function)

**Best for**: Quick start without learning class structure, simplicity

```bash
# Navigate to directory
cd /Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Vizuara_Qwen-3_0.6B_Training_Pipeline/src/training_pipeline

# Install dependencies
pip install -r requirements.txt

# Edit the script to select your dataset (line 52)
nano train_qwen_standalone.py
# Change: SELECTED_DATASET = "10k"  # Options: "10k", "100k", "1M", "2M"

# Run the training
python3 train_qwen_standalone.py

# Optional: Run in background
nohup python3 train_qwen_standalone.py > training.log 2>&1 &
```

**What it includes**:
- ✅ Complete training pipeline
- ✅ All functionality from refactored version
- ✅ Proven and tested
- ⚠️ Must edit source code to change settings
- ⚠️ Monolithic structure (800+ lines in one function)

### Option 3: Using the Jupyter Notebook

**Status**: Partially complete (12 cells, covers setup and configuration only)

**Best for**: Interactive exploration, step-by-step debugging, learning

**Current Coverage**:
- Cell 0-1: Introduction and installation
- Cell 2-3: Library imports and setup
- Cell 4-6: Dataset selection
- Cell 7-8: Configuration loading
- Cell 9-11: Hyperparameter configuration

**Missing Components** (would need to be added):
- WandB initialization
- Dataset loading from HuggingFace
- Tokenization with caching
- Model loading and quantization
- LoRA/PEFT application
- Training loop
- Pre and post-training evaluation
- Model saving

#### Running the Notebook (Current State)

**Prerequisites**:
```bash
# Install Jupyter if not already installed
pip install jupyter notebook
# Or for JupyterLab
pip install jupyterlab

# Install dependencies
pip install -r requirements.txt
```

**Steps**:

1. **Start Jupyter**:
   ```bash
   cd /Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Vizuara_Qwen-3_0.6B_Training_Pipeline/src/training_pipeline
   
   # Option A: Jupyter Notebook
   jupyter notebook
   
   # Option B: JupyterLab (recommended)
   jupyter lab
   ```

2. **Open the notebook**:
   - In the browser window that opens, click on `train_qwen_generic.ipynb`

3. **Run cells sequentially**:
   - **Cell 0-1**: Read introduction and installation instructions
   - **Cell 2**: Uncomment and run pip install commands if needed
   - **Cell 3**: Import libraries (verify CUDA availability)
   - **Cell 4-5**: Review dataset options
   - **Cell 6**: **IMPORTANT - Select your dataset**:
     ```python
     SELECTED_DATASET = "10k"  # Change to "10k", "100k", "1M", or "2M"
     ```
   - **Cell 7-8**: Load and configure settings
   - **Cell 9-10**: Review hyperparameters
   - **Cell 11**: (Optional) Modify hyperparameters:
     ```python
     # Example modifications:
     config['training']['num_train_epochs'] = 5
     config['training']['learning_rate'] = 2e-4
     config['peft']['lora_r'] = 32
     ```

4. **Current Limitation**: 
   - The notebook currently ends after hyperparameter configuration
   - To complete training, you have two options:
     - **A) Use the standalone script** (recommended): The notebook has done the setup, now run `python3 train_qwen_standalone.py`
     - **B) Request notebook completion**: Ask to add the remaining cells for full training

#### Completing the Notebook

If you need the full notebook implementation, the following cells need to be added:

- **Cell 12-13**: WandB initialization
- **Cell 14-15**: Utility functions
- **Cell 16-18**: Dataset loading from HuggingFace
- **Cell 19-20**: Label mappings and class weights
- **Cell 21-22**: Tokenizer loading
- **Cell 23-24**: Dataset tokenization with caching
- **Cell 25-26**: Model loading and quantization
- **Cell 27-28**: LoRA/PEFT application
- **Cell 29-30**: Data collator and metrics
- **Cell 31-32**: Custom weighted trainer
- **Cell 33-34**: Training arguments
- **Cell 35-36**: Trainer initialization
- **Cell 37-38**: Pre-training evaluation
- **Cell 39-40**: Training execution
- **Cell 41-42**: Post-training evaluation
- **Cell 43-44**: Model saving
- **Cell 45**: Summary and WandB finish

**To request completion**, let me know and I can add all remaining cells to make it fully functional.

## Dataset Information

| Dataset | HuggingFace Name | Size | Balance | Class Weights |
|---------|------------------|------|---------|---------------|
| 10k | codefactory4791/raid_aligned_10k | ~10K | 50/50 | [1.0, 1.0] |
| 100k | codefactory4791/raid_aligned_100k | ~100K | 50/50 | [1.0, 1.0] |
| 1M | codefactory4791/raid_aligned_1000k | ~1M | AI: 52.17%, Human: 47.83% | [0.9167, 1.0909] |
| 2M | codefactory4791/raid_aligned_2000k | ~2M | AI: 54.57%, Human: 45.43% | [0.8324, 1.2009] |

**Class Weights:** `[AI_Generated, Human_Written]`

## Key Features

- **Automatic Class Weighting**: Class weights are automatically applied based on the selected dataset's label distribution
- **WandB Integration**: All experiments are logged to Weights & Biases project "Vizuara AI Human Content Detection Qwen 3 0.6B Finetuning"
- **Optimized for A100**: Configuration is optimized for RunPod A100 GPUs
- **PEFT/LoRA**: Uses LoRA for efficient fine-tuning
- **Caching**: Tokenized datasets are cached for faster subsequent runs

## Output Structure

```
output/
  qwen3-0.6b-10k/          # Model checkpoints and results
  qwen3-0.6b-100k/
  qwen3-0.6b-1m/
  qwen3-0.6b-2m/

logs/
  qwen3-0.6b-10k/          # Training logs
  ...

tokenized_cache/
  10k/                     # Cached tokenized datasets
  ...
```

## Configuration

All configuration is managed through `config.yaml`. Key parameters:

### Training
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 32
- `gradient_accumulation_steps`: 8 (effective batch size: 256)
- `learning_rate`: 1e-4
- `lr_scheduler_type`: cosine

### LoRA
- `lora_r`: 16
- `lora_alpha`: 32
- `lora_dropout`: 0.05

### Quantization
- `load_in_4bit`: true (4-bit quantization for memory efficiency)

## Modifying Hyperparameters

### In the Notebook
Edit the configuration in Cell 11:

```python
# Example: Train for 5 epochs
config['training']['num_train_epochs'] = 5

# Example: Use higher learning rate
config['training']['learning_rate'] = 2e-4

# Example: Increase LoRA rank
config['peft']['lora_r'] = 32
```

### In the Config File
Edit `config.yaml` directly before running the script/notebook.

## WandB Configuration

The pipeline automatically logs to Weights & Biases:

- **Project**: Vizuara AI Human Content Detection Qwen 3 0.6B Finetuning
- **API Key**: Pre-configured in the notebook (939a604fbdaf6ec2e540b7d12232eb267ec5bab2)
- **Tags**: Automatically includes dataset size (e.g., "dataset-10k")

## Training Time Estimates

| Dataset | Training Time (A100) | Disk Space |
|---------|----------------------|------------|
| 10k | ~30 minutes | ~5 GB |
| 100k | ~3 hours | ~10 GB |
| 1M | ~12 hours | ~50 GB |
| 2M | ~24 hours | ~100 GB |

*Estimates are approximate and may vary based on GPU availability and caching.*

## Running in Different Environments

### Local Machine
```bash
# Install Jupyter
pip install jupyter notebook

# Navigate to directory
cd /Users/mle/Documents/MLEngineering/Vizuara_LLMResearch/Vizuara_Qwen-3_0.6B_Training_Pipeline/src/training_pipeline

# Start Jupyter
jupyter notebook
```

### RunPod (Recommended for Training)

**For Standalone Script**:
```bash
# Upload the training_pipeline folder to RunPod
# SSH into RunPod instance
cd /workspace/training_pipeline

# Install dependencies
pip install -r requirements.txt

# Run training
python3 train_qwen_standalone.py
```

**For Notebook on RunPod**:
```bash
# RunPod usually has Jupyter pre-installed
# Access via the Jupyter Lab URL provided by RunPod
# Upload train_qwen_generic.ipynb
# Run cells sequentially
```

### Google Colab

1. Upload the notebook to Google Colab
2. Upload `config.yaml` to Colab session
3. Install dependencies in first cell:
   ```python
   !pip install -q transformers accelerate datasets evaluate peft bitsandbytes
   !pip install -q scikit-learn pandas numpy PyYAML wandb
   ```
4. Run cells sequentially

### Jupyter Lab vs Jupyter Notebook

**Jupyter Lab** (Recommended):
- Modern interface with file browser
- Better for managing multiple files
- `jupyter lab`

**Jupyter Notebook** (Classic):
- Simpler interface
- `jupyter notebook`

Both work with the notebook file.

## Requirements

### Core Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install -U transformers accelerate datasets evaluate peft bitsandbytes
pip install -U scikit-learn pandas numpy PyYAML
pip install -U wandb
```

### Jupyter-Specific Requirements
```bash
pip install jupyter notebook  # For classic notebook
# OR
pip install jupyterlab  # For JupyterLab (recommended)

# Optional: for better notebook experience
pip install ipywidgets
```

## Choosing Between Notebook and Script

| Criterion | Standalone Script | Jupyter Notebook |
|-----------|------------------|------------------|
| **Completeness** | 100% Complete | ~40% Complete |
| **Production Ready** | Yes | No (needs completion) |
| **Best For** | RunPod, background jobs | Interactive exploration |
| **Ease of Use** | Single command | Step-by-step control |
| **Debugging** | Console logs | Cell-by-cell execution |
| **Long Training** | Excellent (can run in background) | Not recommended |
| **Documentation** | Self-contained | Inline explanations |
| **Modification** | Edit Python file | Edit cells interactively |
| **Recommendation** | **Use this for training** | Use for learning/debugging |

**Bottom Line**: For actual training runs, especially on RunPod, use `train_qwen_standalone.py`. The notebook is great for understanding the pipeline step-by-step but is incomplete.

## Troubleshooting

### General Issues

#### Out of Memory
```yaml
# In config.yaml, reduce batch size:
training:
  per_device_train_batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size
```

#### Slow Training
- Check that `dataloader_num_workers` is appropriate for your system
- Ensure cached datasets are being used (check `tokenized_cache/` directory)
- Consider using a smaller dataset for testing first (start with 10k)
- First run is slower due to downloading and tokenization

#### WandB Issues
- Verify API key is correct: `939a604fbdaf6ec2e540b7d12232eb267ec5bab2`
- Check internet connectivity on RunPod
- Set `wandb.enabled: false` in config.yaml to disable
- Manual login: `wandb login 939a604fbdaf6ec2e540b7d12232eb267ec5bab2`

### Notebook-Specific Issues

#### Jupyter Not Starting
```bash
# Reinstall Jupyter
pip install --upgrade jupyter notebook

# Or use JupyterLab
pip install --upgrade jupyterlab
jupyter lab
```

#### Kernel Dies During Execution
- Out of memory - reduce batch size in Cell 11
- CUDA out of memory - restart kernel and reduce `per_device_train_batch_size`
- Check GPU memory: `nvidia-smi`

#### Cannot Find config.yaml
- Ensure `config.yaml` is in the same directory as the notebook
- Check current working directory in notebook:
  ```python
  import os
  print(os.getcwd())
  ```

#### ModuleNotFoundError
```bash
# In notebook cell, install missing package:
!pip install <package-name>

# Or restart kernel after installing via terminal
```

#### Notebook Cells Not Running
- Make sure you're running cells in order (Cell 1, 2, 3, etc.)
- Don't skip cells - they depend on previous cells
- Restart kernel and run all cells: `Kernel` → `Restart & Run All`

#### WandB in Notebook
The notebook has WandB API key pre-configured. If you see login prompt:
```python
# In a notebook cell:
import wandb
wandb.login(key="939a604fbdaf6ec2e540b7d12232eb267ec5bab2")
```

### Standalone Script Issues

#### Script Exits Immediately
- Check for Python errors: `python3 train_qwen_standalone.py 2>&1 | tee error.log`
- Verify dataset selection at line 52
- Ensure config.yaml exists in same directory

#### Background Job Not Running
```bash
# Check if process is running
ps aux | grep train_qwen

# View output
tail -f training.log

# Kill process if needed
pkill -f train_qwen_standalone.py
```

#### Cannot Resume Training
- Set `resume_from_checkpoint` in config.yaml:
  ```yaml
  misc:
    resume_from_checkpoint: "./output/qwen3-0.6b-10k/checkpoint-1000"
  ```

## Notebook Best Practices

### When Using the Notebook (Current Partial Version)

1. **Start Small**: Use the 10k dataset first to validate setup
2. **Run Sequentially**: Don't skip cells, run them in order
3. **Check Each Output**: Verify each cell's output before proceeding
4. **Save Frequently**: Use `File` → `Save and Checkpoint`
5. **Monitor Resources**: Keep an eye on GPU memory with `nvidia-smi`

### Recommended Workflow

**For Initial Setup and Testing** (Use Notebook):
1. Open `train_qwen_generic.ipynb`
2. Run through Cells 1-11 to configure everything
3. Verify dataset selection and hyperparameters
4. Close notebook

**For Actual Training** (Switch to Script):
1. Your configuration from the notebook is stored in variables
2. Run `python3 train_qwen_standalone.py` with same settings
3. Monitor via WandB dashboard
4. Script handles the complete training automatically

### If You Want to Complete the Notebook

The notebook currently covers setup and configuration but needs additional cells for the full training pipeline. If you need a complete notebook:

**Request**: Ask to add the remaining ~30 cells that include:
- WandB initialization and login
- Dataset loading from HuggingFace
- Tokenization with smart caching
- Model loading with 4-bit quantization
- LoRA/PEFT configuration and application
- Custom weighted trainer implementation
- Training execution with progress bars
- Pre and post-training evaluation
- Confusion matrices and classification reports
- Model and checkpoint saving
- WandB logging and visualization

**Estimated Addition**: ~600 lines of code in ~30 additional cells

**Alternative**: The standalone script already has all this functionality implemented and tested.

## Next Steps After Training

### 1. Review Training Metrics

**WandB Dashboard**:
- Project: "Vizuara AI Human Content Detection Qwen 3 0.6B Finetuning"
- View loss curves, accuracy trends
- Compare different runs
- Download metrics as CSV

**Local Files**:
```bash
# View metrics files
cat output/qwen3-0.6b-10k/pre_training_metrics.txt
cat output/qwen3-0.6b-10k/post_training_metrics.txt

# View predictions
head -20 output/qwen3-0.6b-10k/predictions.csv
```

### 2. Evaluate Model Performance

The script automatically evaluates on the test set. Review:
- Accuracy and balanced accuracy
- Precision, recall, F1 score
- Confusion matrix
- Per-class performance

### 3. Load and Use the Trained Model

**For LoRA Models**:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3-0.6B",
    num_labels=2
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model, 
    "./output/qwen3-0.6b-10k"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./output/qwen3-0.6b-10k")

# Inference
text = "This is a sample text to classify"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()

# 0 = AI_Generated, 1 = Human_Written
result = "AI_Generated" if prediction == 0 else "Human_Written"
print(f"Prediction: {result}")
```

**For Full Model (if PEFT disabled)**:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./output/qwen3-0.6b-10k")
tokenizer = AutoTokenizer.from_pretrained("./output/qwen3-0.6b-10k")

# Same inference code as above
```

### 4. Deploy Model

Options for deployment:
- FastAPI server
- Gradio interface
- HuggingFace Spaces
- Custom REST API

### 5. Iterate and Improve

Based on results:
- Try different hyperparameters (epochs, learning rate, LoRA rank)
- Use larger dataset (scale from 10k → 100k → 1M → 2M)
- Experiment with different class weights
- Fine-tune on domain-specific data

## Notebook vs Script - Quick Reference

### Use the Notebook When:
- Learning how the pipeline works
- Debugging specific components
- Experimenting with small code changes
- Teaching or presenting the methodology
- Need cell-by-cell execution control

### Use the Standalone Script When:
- Running actual training (especially long runs)
- Training on RunPod or cloud GPUs
- Need to run in background
- Production or automated workflows
- Want complete, tested implementation

## Additional Resources

- **QUICKSTART.md**: Fast-start guide with minimal steps
- **PIPELINE_SUMMARY.md**: Detailed overview of the implementation
- **config.yaml**: Full configuration reference with comments
- **WandB Dashboard**: Real-time training monitoring

## Support

For issues or questions:
1. Check this README first
2. Review QUICKSTART.md for common tasks
3. Check troubleshooting sections above
4. Verify config.yaml settings
5. Review WandB dashboard for training issues

## Summary

**Quick Decision Guide**:
- **Want CLI and flexibility?** → Use `train_qwen.py` ✅ **RECOMMENDED**
- **Want simple quick start?** → Use `train_qwen_standalone.py`
- **Want to learn/explore?** → Use `train_qwen_generic.ipynb` (partial)
- **Need to change parameters often?** → Use `train_qwen.py` (CLI arguments)
- **Best for RunPod?** → `train_qwen.py` (no code editing)
- **Best for production?** → `train_qwen.py` (professional structure)
- **Best for learning?** → `train_qwen_generic.ipynb` then `train_qwen.py`

**Script Comparison**:

| Feature | train_qwen.py | train_qwen_standalone.py | train_qwen_generic.ipynb |
|---------|---------------|--------------------------|--------------------------|
| **CLI Support** | ✅ Full | ❌ None | ❌ N/A |
| **Code Editing** | ❌ Not needed | ✅ Required | ✅ In cells |
| **Architecture** | Class-Based | Procedural | Interactive |
| **Modularity** | ✅ High | ⚠️ Low | ✅ High |
| **Testability** | ✅ Easy | ⚠️ Hard | ✅ Easy |
| **Completeness** | 100% | 100% | 40% |
| **Learning Curve** | Medium | Low | Low |
| **Production Ready** | ✅ Yes | ✅ Yes | ❌ No |

See **ARCHITECTURE.md** for detailed comparison.

