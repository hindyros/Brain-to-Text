# Brain-to-Text Competition Guide

Complete guide for the Kaggle Brain-to-Text '25 competition codebase.

## Table of Contents

1. [Dataset Download](#dataset-download)
2. [Model Architectures](#model-architectures)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Experiment Ideas](#experiment-ideas)

---

## Dataset Download

### What to Download

**Essential**: `t15_copyTask_neuralData/hdf5_data_final/` folder (~10-12 GB)
- Contains all neural data files (45 sessions)
- Required for training

**Optional**: `t15_pretrained_rnn_baseline/` (~1-2 GB)
- Only if you want to fine-tune from baseline

### Download Methods

**Kaggle** (Recommended):
1. Join competition: https://www.kaggle.com/competitions/brain-to-text-25
2. Accept rules
3. Data tab → Expand `t15_copyTask_neuralData` → `hdf5_data_final`
4. Download `hdf5_data_final` folder (NOT "Download All")

**Dryad** (Alternative):
- https://doi.org/10.5061/dryad.dncjsxm85
- Download and extract, use `hdf5_data_final` folder

### Setup

```bash
# Set environment variable
export BRAIN_TO_TEXT_DATA_DIR=/path/to/hdf5_data_final

# Verify download
python test_data_loading.py
```

### Expected Structure

```
hdf5_data_final/
├── t15.2023.08.13/
│   ├── data_train.hdf5
│   ├── data_val.hdf5
│   └── data_test.hdf5
└── ... (45 sessions)
```

---

## Model Architectures

### CTC Models (Two-Stage: Neural → Phonemes → Text)

**Pipeline**: Neural Features → Encoder → Phoneme Probabilities → CTC Decode → Phonemes → Language Model → Text

**How it works**:
1. Encoder predicts phoneme probabilities at each timestep
2. CTC loss handles alignment automatically
3. Decoding converts probabilities to phoneme sequence
4. Language model converts phonemes to text

**Models Available**:
- `RecurrentModel`: RNN, LSTM, or GRU
- `TransformerEncModel`: Transformer encoder
- `ConformerModel`: CNN + Transformer hybrid

**Advantages**: Modular, interpretable, works well with language models  
**Disadvantages**: Two-stage, requires language model

### E2E Models (Direct: Neural → Text)

**Pipeline**: Neural Features → Encoder → Latent Rep → Text Decoder → Text

**How it works**:
1. Encoder processes neural features → latent representation
2. Text decoder generates text autoregressively (character by character)
3. No intermediate phoneme step

**Models Available**:
- `EndToEndModel`: Transformer encoder-decoder
- `NeuralEncoder`: Convolutional + Transformer
- `TextDecoder`: Autoregressive decoder

**Advantages**: Direct, end-to-end optimization, simpler  
**Disadvantages**: Data hungry, slower training, less interpretable

### Which to Use?

- **CTC**: Want to use strong language models, need interpretability, limited data
- **E2E**: Want simplicity, end-to-end optimization, sufficient data

---

## Quick Start

### Training CTC Model

```bash
python training/train_ctc.py \
    --config config/ctc_config.yaml \
    --model-type LSTM \
    --data-dir /path/to/hdf5_data_final
```

### Training E2E Model

```bash
python training/train_e2e.py \
    --config config/e2e_config.yaml \
    --data-dir /path/to/hdf5_data_final
```

### Generate Submission

```bash
# CTC
python inference/generate_submission.py \
    --model-path checkpoints/ctc/best_checkpoint.pth \
    --config config/ctc_config.yaml \
    --test-data-dir /path/to/hdf5_data_final \
    --output submission.csv \
    --model-type ctc \
    --ctc-arch LSTM

# E2E
python inference/generate_submission.py \
    --model-path checkpoints/e2e/best_checkpoint.pth \
    --config config/e2e_config.yaml \
    --test-data-dir /path/to/hdf5_data_final \
    --output submission.csv \
    --model-type e2e
```

---

## Configuration

Config files (`config/*.yaml`) specify:
- Model architecture parameters
- Training hyperparameters (LR, batch size, epochs)
- Data augmentation settings
- Checkpoint paths

Edit these files to experiment with different settings.

---

## Experiment Ideas

See `Todo.txt` for comprehensive experiment list. Key areas:

**Architecture**:
- Different model types (RNN, LSTM, GRU, Transformer, Conformer)
- Vary depths, widths, attention heads
- Hybrid architectures

**Data Augmentation**:
- Temporal masking (vary percentages)
- Electrode dropout
- Gaussian noise
- Mixup/CutMix

**Training**:
- Different optimizers (Adam, AdamW, SGD)
- Learning rate schedules
- Loss functions (CTC variants, transducer)
- Regularization (dropout, weight decay)

**Language Models** (for CTC):
- N-gram models (1, 3, 5-gram)
- Neural LMs (GPT-2, BERT)
- Ensemble multiple LMs

**Advanced**:
- Transfer learning from T12 dataset
- Model ensemble
- Test-time augmentation
- Beam search decoding

---

## References

- Competition: https://www.kaggle.com/competitions/brain-to-text-25
- Dataset: https://doi.org/10.5061/dryad.dncjsxm85
- Baseline WER: 6.70% (target: < 5%)

