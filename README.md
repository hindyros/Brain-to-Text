# Brain-to-Text Competition

Modular codebase for the Kaggle Brain-to-Text '25 competition. Decodes speech from neural activity using CTC-based and end-to-end approaches.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (see docs/GUIDE.md)
export BRAIN_TO_TEXT_DATA_DIR=/path/to/hdf5_data_final

# Train a model
python training/train_ctc.py \
    --config config/ctc_config.yaml \
    --model-type LSTM \
    --data-dir $BRAIN_TO_TEXT_DATA_DIR

# Generate submission
python inference/generate_submission.py \
    --model-path checkpoints/ctc/best_checkpoint.pth \
    --config config/ctc_config.yaml \
    --test-data-dir $BRAIN_TO_TEXT_DATA_DIR \
    --output submission.csv \
    --model-type ctc \
    --ctc-arch LSTM
```

## Project Structure

```
Brain-to-Text/
├── data/           # Dataset loading and tokenization
├── models/         # CTC and E2E model architectures
├── training/       # Training scripts
├── inference/      # Submission generation
├── config/         # Configuration files
├── utils/          # Evaluation utilities
├── docs/           # Documentation
└── Todo.txt        # Experiment ideas
```

## Documentation

See **[docs/GUIDE.md](docs/GUIDE.md)** for:
- Dataset download instructions
- Model architecture explanations (CTC vs E2E)
- Detailed usage examples
- Configuration guide
- Experiment ideas

## Key Features

- **CTC Models**: RNN/LSTM/GRU/Transformer/Conformer for phoneme prediction
- **E2E Models**: Direct neural-to-text generation
- **Data Augmentation**: Temporal masking, electrode dropout, noise
- **Modular Design**: Self-contained, reusable components

## Competition Info

- **Baseline WER**: 6.70%
- **Target**: < 5% WER
- **Dataset**: 10,948 sentences, 45 sessions, 20 months
- **Competition**: [Kaggle Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25)
