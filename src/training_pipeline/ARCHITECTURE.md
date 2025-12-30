# Training Pipeline Architecture

## Available Scripts

| Script | Structure | Status | Lines | Best For |
|--------|-----------|--------|-------|----------|
| `train_qwen.py` | **Class-Based (Refactored)** | ✅ Complete | ~1200 | **Production, Flexibility** |
| `train_qwen_standalone.py` | Procedural (Original) | ✅ Complete | 934 | Quick Start, Simple |
| `train_qwen_generic.ipynb` | Notebook | ⚠️ Partial | 300 | Learning, Interactive |

## train_qwen.py (Recommended - Class-Based Architecture)

### Architecture

```python
QwenTrainingPipeline (Main Class)
├── __init__()              # Initialize with parameters
├── load_config()           # Load and validate configuration
├── initialize_wandb()      # Setup WandB logging
├── load_datasets()         # Load from HuggingFace
├── create_label_mappings() # Create label mappings and weights
├── setup_tokenizer()       # Load and configure tokenizer
├── tokenize_datasets()     # Tokenize with caching
├── setup_model()           # Load model with quantization
├── apply_lora()            # Apply LoRA/PEFT
├── create_trainer()        # Initialize trainer
├── run_pre_evaluation()   # Pre-training baseline
├── train()                 # Execute training
├── run_post_evaluation()  # Post-training evaluation
├── save_artifacts()        # Save model and config
├── print_summary()         # Print results
├── cleanup()               # Cleanup WandB
└── run()                   # Orchestrate all stages
```

### Features

✅ **Modular Design**: Each stage is a separate method
✅ **Easy Testing**: Can test individual components
✅ **CLI Support**: Full argument parsing with argparse
✅ **Flexible**: Override any parameter from command line
✅ **Reusable**: Can import and use class in other scripts
✅ **Clean**: Separation of concerns
✅ **Documented**: Comprehensive docstrings
✅ **Error Handling**: Try-catch with cleanup
✅ **Type Hints**: Better code clarity

### Usage Examples

**Basic Usage**:
```bash
python train_qwen.py --dataset 10k
```

**Custom Hyperparameters**:
```bash
python train_qwen.py --dataset 1M --epochs 5 --batch-size 16 --learning-rate 2e-4
```

**Disable WandB**:
```bash
python train_qwen.py --dataset 100k --no-wandb
```

**Resume Training**:
```bash
python train_qwen.py --dataset 10k --resume ./output/qwen3-0.6b-10k/checkpoint-1000
```

**Full Customization**:
```bash
python train_qwen.py --dataset 2M \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --lora-rank 32 \
  --config custom_config.yaml
```

### CLI Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--dataset` | str | Yes | - | Dataset size (10k, 100k, 1M, 2M) |
| `--config` | str | No | config.yaml | Path to config file |
| `--epochs` | int | No | None | Number of epochs (overrides config) |
| `--batch-size` | int | No | None | Batch size per device |
| `--learning-rate` | float | No | None | Learning rate |
| `--lora-rank` | int | No | None | LoRA rank |
| `--no-wandb` | flag | No | False | Disable WandB logging |
| `--resume` | str | No | None | Checkpoint path to resume from |

### Advantages Over Original

1. **Modularity**: Can call individual methods
   ```python
   from train_qwen import QwenTrainingPipeline
   
   pipeline = QwenTrainingPipeline(dataset="10k")
   pipeline.load_config()
   pipeline.load_datasets()
   # ... customize as needed
   ```

2. **Testing**: Easy to test components
   ```python
   def test_tokenization():
       pipeline = QwenTrainingPipeline(dataset="10k")
       pipeline.load_config()
       pipeline.setup_tokenizer()
       assert pipeline.tokenizer is not None
   ```

3. **Flexibility**: Override any stage
   ```python
   pipeline = QwenTrainingPipeline(dataset="10k")
   pipeline.load_config()
   # Override config
   pipeline.config['training']['num_train_epochs'] = 10
   pipeline.run()
   ```

4. **CLI Power**: No code editing needed
   ```bash
   # Quick experiment with different settings
   python train_qwen.py --dataset 10k --epochs 1  # Fast test
   python train_qwen.py --dataset 10k --epochs 5  # Full run
   ```

## train_qwen_standalone.py (Original - Procedural)

### Architecture

```python
main()  # 800+ lines
├── Configuration loading
├── WandB initialization
├── Dataset loading
├── Tokenization
├── Model loading
├── LoRA application
├── Trainer creation
├── Training execution
├── Evaluation
└── Saving
```

### Features

✅ **Simple**: Single function, easy to follow linearly
✅ **Complete**: All functionality in one place
✅ **Proven**: Tested and working
⚠️ **Monolithic**: Hard to modify individual parts
⚠️ **No CLI**: Must edit source code to change settings

### Usage

```bash
# Edit line 52 to change dataset
nano train_qwen_standalone.py
# Change: SELECTED_DATASET = "10k"

python train_qwen_standalone.py
```

### When to Use

- Quick start without learning class structure
- Don't need CLI arguments
- Following a tutorial step-by-step
- Prefer linear code flow

## train_qwen_generic.ipynb (Notebook - Interactive)

### Architecture

12 cells covering:
- Setup and imports
- Dataset selection
- Configuration loading
- Hyperparameter display

**Status**: Incomplete (~40% done)

### When to Use

- Learning the pipeline interactively
- Debugging step-by-step
- Teaching or presenting
- Experimenting with small changes

## Comparison

### Code Organization

**Class-Based (train_qwen.py)**:
```python
class QwenTrainingPipeline:
    def load_config(self):
        # ~30 lines
    
    def load_datasets(self):
        # ~40 lines
    
    def train(self):
        # ~50 lines
```
✅ Clean separation, easy to navigate

**Procedural (train_qwen_standalone.py)**:
```python
def main():
    # Load config (40 lines)
    # Load datasets (80 lines)
    # Train (100 lines)
    # ... 800+ lines total
```
⚠️ Everything in one function

### Reusability

**Class-Based**:
```python
# Can import and customize
from train_qwen import QwenTrainingPipeline

pipeline = QwenTrainingPipeline(dataset="10k", epochs=1)
pipeline.run()

# Or run specific stages
pipeline.load_config()
pipeline.load_datasets()
# ... custom logic
pipeline.train()
```

**Procedural**:
```python
# Must copy-paste or edit source
# No easy way to reuse parts
```

### Testing

**Class-Based**:
```python
def test_config_loading():
    pipeline = QwenTrainingPipeline(dataset="10k")
    pipeline.load_config()
    assert pipeline.config is not None
    assert pipeline.dataset_name == "codefactory4791/raid_aligned_10k"

def test_tokenization():
    pipeline = QwenTrainingPipeline(dataset="10k")
    pipeline.load_config()
    pipeline.setup_tokenizer()
    assert pipeline.tokenizer is not None
```

**Procedural**:
```python
# Difficult to test individual components
# Would need to refactor or mock heavily
```

### CLI Flexibility

**Class-Based**:
```bash
# Change any parameter without editing code
python train_qwen.py --dataset 10k --epochs 5
python train_qwen.py --dataset 10k --epochs 3 --batch-size 16
python train_qwen.py --dataset 1M --learning-rate 2e-4 --lora-rank 32
```

**Procedural**:
```bash
# Must edit source code
nano train_qwen_standalone.py  # Edit SELECTED_DATASET
python train_qwen_standalone.py
```

## Recommendations

### For Production Training

**Use**: `train_qwen.py` (Class-Based)

**Reasons**:
- CLI arguments for flexibility
- Easy to modify and extend
- Better error handling
- Can be imported and customized
- Professional code structure

**Example**:
```bash
python train_qwen.py --dataset 1M --epochs 3 --batch-size 32
```

### For Quick Start / Learning

**Use**: `train_qwen_standalone.py` (Procedural)

**Reasons**:
- Simpler to understand initially
- Linear flow
- Less abstraction
- Proven and tested

**Example**:
```bash
# Edit source to select dataset
nano train_qwen_standalone.py
python train_qwen_standalone.py
```

### For Interactive Exploration

**Use**: `train_qwen_generic.ipynb` (Notebook)

**Reasons**:
- Cell-by-cell execution
- Inline documentation
- Visual feedback
- Good for learning

**Note**: Currently incomplete, missing training stages

## Migration Path

### From Standalone to Class-Based

**Before** (train_qwen_standalone.py):
```bash
# Edit line 52
SELECTED_DATASET = "1M"
python train_qwen_standalone.py
```

**After** (train_qwen.py):
```bash
python train_qwen.py --dataset 1M
```

### Customization Example

**Before**:
```python
# Edit inside main()
config['training']['num_train_epochs'] = 5
config['training']['learning_rate'] = 2e-4
```

**After**:
```bash
python train_qwen.py --dataset 10k --epochs 5 --learning-rate 2e-4
```

## Summary

| Criterion | train_qwen.py | train_qwen_standalone.py | train_qwen_generic.ipynb |
|-----------|---------------|-------------------------|--------------------------|
| **Structure** | Class-Based | Procedural | Interactive Cells |
| **Modularity** | ✅ Excellent | ⚠️ Monolithic | ✅ Good |
| **Testability** | ✅ Easy | ⚠️ Difficult | ✅ Easy |
| **CLI Support** | ✅ Full | ❌ None | ❌ N/A |
| **Reusability** | ✅ High | ⚠️ Low | ⚠️ Medium |
| **Learning Curve** | Medium | Low | Low |
| **Flexibility** | ✅ High | ⚠️ Low | ✅ High |
| **Production Ready** | ✅ Yes | ✅ Yes | ❌ No |
| **Completeness** | 100% | 100% | 40% |
| **Recommended For** | Production | Quick Start | Learning |

**Bottom Line**: 
- **Use `train_qwen.py` for actual training** - it's professional, flexible, and has CLI support
- **Use `train_qwen_standalone.py` for quick tests** - simpler but less flexible
- **Use `train_qwen_generic.ipynb` for learning** - interactive but incomplete

