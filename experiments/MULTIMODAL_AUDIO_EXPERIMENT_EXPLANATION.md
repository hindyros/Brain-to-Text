# Multimodal Audio Experiment: Detailed Explanation

## Overview

This document explains the rationale, design, and potential risks of our multimodal brain-to-text decoding experiment that uses audio features as training-time supervision.

---

## 1. The Baseline Approach 

### Current Pipeline
- **Input**: Neural activity signals from 512 channels, sampled at 20ms intervals
- **Model Architecture**: LSTM encoder → CTC decoder → Phoneme predictions
- **Training**: Supervised learning from neural data to phoneme labels
- **Inference**: Neural data only → Phoneme sequence predictions

### Goal
Decode phoneme sequences directly from neural activity, enabling brain-computer interfaces (BCIs) for speech restoration.

---

## 2. The Proposed Change: Adding Audio During Training

### What We're Adding
- **Audio Features**: Mel-spectrograms and/or Wav2Vec2Phoneme features generated from text labels using Text-to-Speech (TTS)
- **Training**: Neural + Audio features → Phoneme predictions
- **Evaluation**: Neural-only inputs (no audio) → Phoneme predictions

### Key Design Principle
Audio is used **only during training** as auxiliary supervision. At test time, all models (including multimodal ones) are evaluated with **neural-only inputs**, simulating the real-world BCI scenario where audio is unavailable.

---

## 3. The Core Idea: Knowledge Distillation

### Concept
This follows a **knowledge distillation** or **teacher-student learning** paradigm:

- **Teacher Signal**: Audio features contain rich acoustic-phonetic information (formants, prosody, timing, phoneme boundaries)
- **Student**: The neural encoder learns to extract similar information from neural signals alone
- **Learning Process**: By aligning neural patterns with audio patterns during training, the neural encoder learns better representations

### Why This Might Work

1. **Complementary Information**: 
   - Neural signals encode motor planning and execution
   - Audio signals encode acoustic-phonetic realization
   - These are related but not identical - audio can guide the neural encoder to learn better phoneme-aligned representations

2. **Temporal Alignment Cues**:
   - Audio provides explicit timing information about phoneme boundaries
   - This can help the CTC decoder learn better alignments between neural activity and phoneme sequences

3. **Regularization**:
   - Audio acts as a form of regularization, preventing the model from learning spurious patterns in neural data
   - Forces the model to learn representations that align with known acoustic-phonetic structure

---

## 4. Architecture Design

### Cross-Modal Attention Mechanism

The multimodal models use **cross-modal attention** with a specific design:

```
Query: Neural features (primary input)
Key/Value: Audio features (auxiliary context)
```

### Why This Design?

1. **Neural as Query**: Forces neural features to be the primary input
   - The model must process neural features first through the neural encoder
   - Neural features determine what information to attend to in audio

2. **Audio as Key/Value**: Provides context but doesn't replace neural features
   - Audio guides attention to relevant neural patterns
   - The model learns: "When I see this neural pattern, it corresponds to this audio pattern"

3. **Feature Fusion**: Combines neural and attended features
   - Final representation: `[neural_features, attended_features]`
   - Neural features remain the foundation

### Forward Pass Flow

1. **Neural Encoder**: Processes neural signals → `neural_out` (B, T, hidden_dim)
2. **Audio Encoder**: Processes audio features → `audio_out` (B, T, hidden_dim)
3. **Cross-Attention**: `neural_out` queries `audio_out` → `attended_neural`
4. **Fusion**: Concatenate `[neural_out, attended_neural]` → fused features
5. **Output**: Fused features → phoneme logits

### Test-Time Behavior

At test time, audio features are set to **zero**:
- Audio encoder processes zero features → produces zero/background representations
- Cross-attention still operates, but attends to "silence"
- Model must rely on `neural_out` (the improved neural representations learned during training)

---

## 5. The Critical Concern: Won't This Make the Model Weaker?

### The Valid Concern

**Question**: If we provide audio (which contains the target sentence), isn't the model just "cheating"? Won't it learn to ignore neural features and rely on audio, making it weaker at test time?

**Answer**: This is a **valid and important concern**. The experiment is specifically designed to test whether this happens.

### The Risk

If the model learns to **ignore neural features** and rely primarily on audio during training:

- **Training**: Low loss (model uses audio to predict phonemes)
- **Test Time**: Poor performance (no audio available, neural encoder hasn't learned useful features)
- **Result**: Multimodal model performs **worse** than baseline

### How We Mitigate This Risk

1. **Architecture Design**:
   - Neural features are queries (primary input)
   - Audio features are keys/values (auxiliary context)
   - This forces neural features to be processed first

2. **Evaluation Strategy**:
   - All models tested with neural-only inputs
   - If multimodal models fail, we know audio was harmful
   - Direct comparison: Baseline (neural-only) vs. Multimodal (trained with audio, tested without)

3. **Loss Function**:
   - CTC loss is computed on the final phoneme predictions
   - Model must predict correct phonemes, but can't ignore neural features entirely (they're the query)

### What If It Doesn't Work?

If the baseline outperforms multimodal models, this would:
- **Validate the concern**: Audio supervision was harmful/distracting
- **Suggest improvements**: Need better architecture or training strategy
- **Inform future work**: Alternative approaches needed (see Section 7)

---

## 6. What This Experiment Tests

### Three Possible Outcomes

#### Outcome 1: Multimodal > Baseline
**Interpretation**: Audio supervision helped the neural encoder learn better features

**What this means**:
- Knowledge distillation worked
- Neural encoder learned better phoneme-aligned representations
- Cross-modal alignment was successful
- **Implication**: Real audio recordings (if available) would likely help even more

#### Outcome 2: Multimodal ≈ Baseline
**Interpretation**: Audio supervision didn't help (or hurt)

**Possible reasons**:
- Neural signals already contain sufficient information
- Domain mismatch too severe (TTS audio doesn't match attempted speech)
- Model architecture insufficient to exploit multimodal information
- Dataset too small to learn cross-modal relationships

**Implication**: Need to investigate which factor is limiting

#### Outcome 3: Baseline > Multimodal
**Interpretation**: Audio supervision was harmful/distracting

**What this means**:
- Model learned to rely on audio during training
- Neural encoder didn't learn useful features
- Audio features were distracting rather than helpful
- **Implication**: Validates the concern - need alternative approach

---

## 7. Experimental Design

### Models Tested

1. **Baseline**: Neural-only LSTM + CTC (trained and evaluated with neural data only)
2. **Mel-only**: Neural + Mel-spectrogram features
3. **Wav2Vec2-only**: Neural + Wav2Vec2Phoneme features
4. **Combined**: Neural + Mel + Wav2Vec2Phoneme features

### Controlled Variables

- Same neural data
- Same text labels
- Same train/val/test splits
- Same evaluation protocol (neural-only inputs)
- Same training hyperparameters
- Same random seeds

### Independent Variable

- Presence/absence of audio features **during training only**

### Dependent Variables

- Phoneme Error Rate (PER) on neural-only test inputs
- Training loss convergence
- Model parameter count

### Dataset

- 3 sessions (~300-500 sentences total)
- TTS-generated audio from text labels
- Neural recordings include pre/post-speech silence

---

## 8. Limitations and Caveats

### Domain Mismatch

- **Neural signals**: Come from a person *attempting* to say a sentence
- **TTS audio**: Synthetic speech from gTTS (different speaker, different production)
- **Risk**: Model may learn to match TTS patterns that don't correspond to neural activity
- **Mitigation**: This is a pilot study - if it works despite mismatch, real audio would work better

### Temporal Alignment

- **Neural recordings**: Include full trial duration (~20s) with pre/post-speech silence
- **TTS audio**: Contains only speech (~3-4s), no silence padding
- **Current approach**: Pad audio with silence to match neural length
- **Issue**: Speech portions may not align temporally (person may start speaking at different times)
- **Mitigation**: Cross-modal attention should learn to handle this misalignment

### Small Sample Size

- Limited to 3 sessions (~300-500 sentences)
- Results may not generalize across sessions or participants
- **Mitigation**: This is a proof-of-concept study

---

## 9. Alternative Designs (Not Implemented)

To further mitigate the risk of over-reliance on audio, future work could explore:

### Auxiliary Loss
Add a loss term that penalizes the model if it performs poorly with zero audio during training:
```
L_total = L_ctc + λ * L_neural_only
```
This explicitly encourages the model to work well without audio.

### Progressive Masking
Gradually reduce audio signal strength during training:
- Early epochs: Full audio signal
- Later epochs: Gradually mask/reduce audio
- Final epochs: Zero audio (neural-only)

This forces the model to gradually transition from relying on audio to relying on neural features.

### Adversarial Training
Train a discriminator to detect when the model is relying too heavily on audio:
- Discriminator tries to predict whether audio was used
- Model tries to fool discriminator while maintaining performance
- Forces model to learn neural-only representations

### Contrastive Learning
Use audio to create positive/negative pairs:
- Positive pairs: Neural + corresponding audio
- Negative pairs: Neural + mismatched audio
- Learn representations that align neural and audio for matching pairs

---

## 10. Scientific Validity

### Strengths

- ✅ Controlled experiment (same data, splits, hyperparameters)
- ✅ Realistic evaluation scenario (neural-only inference)
- ✅ Clear hypothesis and null hypothesis
- ✅ Multiple ablation conditions (Mel, Wav2Vec2, Combined)
- ✅ Direct comparison with baseline

### Weaknesses

- ⚠️ Domain mismatch (TTS vs. attempted speech)
- ⚠️ Temporal alignment uncertainty
- ⚠️ Small sample size (3 sessions)
- ⚠️ No ground-truth audio to validate alignment

### Interpretation Guidelines

**If multimodal models outperform baseline**:
- **Conservative interpretation**: Audio supervision helps despite domain mismatch
- **Optimistic interpretation**: Real audio would help even more
- **Caveat**: May be due to regularization effects, not true multimodal learning

**If multimodal models perform similarly**:
- **Interpretation**: Either (a) neural signals sufficient, (b) domain mismatch too severe, or (c) model needs improvement
- **Next steps**: Investigate with real audio, better alignment, or larger models

**If baseline outperforms multimodal**:
- **Interpretation**: Audio features are harmful or distracting
- **Possible reasons**: Domain mismatch, overfitting to TTS, or architecture issues
- **Action**: Need alternative approach or better training strategy

---

## 11. Conclusion

### Does This Experiment Make Sense?

**YES, with important caveats**:

- The experiment tests a valid hypothesis: Can audio supervision improve neural-only decoding?
- The design is sound: controlled comparison with realistic evaluation
- The limitations are acknowledged: domain mismatch, alignment issues
- The interpretation must be cautious: Results may not generalize to real audio

### This is a Valid Pilot Study

The experiment can inform whether to pursue:
1. Real audio recordings (if available)
2. Better alignment strategies
3. More sophisticated multimodal architectures
4. Larger-scale experiments

### Key Takeaway

**The experiment makes sense as a proof-of-concept**, but results should be interpreted in light of the TTS domain mismatch limitation. If multimodal models outperform the baseline despite this limitation, it suggests that real audio recordings would likely provide even greater benefits.

---

## 12. Summary for Quick Reference

**Question**: Does training with audio features improve neural-only decoding performance?

**Approach**: 
- Train models with neural + audio features
- Evaluate all models with neural-only inputs
- Compare multimodal models to neural-only baseline

**Key Design**:
- Cross-modal attention: Neural queries audio
- Audio acts as teacher signal during training
- Test-time evaluation: Neural-only (realistic BCI scenario)

**Risk**: Model might learn to ignore neural features and rely on audio

**Mitigation**: 
- Architecture forces neural as primary input
- Evaluation tests neural-only performance
- Direct comparison with baseline

**What We Learn**:
- If multimodal > baseline: Audio supervision helps
- If multimodal ≈ baseline: Audio didn't help
- If baseline > multimodal: Audio was harmful (validates concern)

---

**End of Document**

