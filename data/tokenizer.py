"""
Tokenization utilities for Brain-to-Text competition.

Provides:
- Phoneme tokenizer (for CTC-based models)
- Character tokenizer (for end-to-end text generation)
- BPE tokenizer support (future)
"""

import string
import torch


# Phoneme vocabulary (from competition baseline)
PHONEME_VOCAB = [
    'BLANK',    # 0 = CTC blank symbol
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    ' | ',    # 39 = silence/word boundary token
]

BLANK_ID = 0


class PhonemeTokenizer:
    """Tokenizer for phoneme sequences (used with CTC loss)."""
    
    def __init__(self):
        self.vocab = PHONEME_VOCAB
        self.blank_id = BLANK_ID
        self.vocab_size = len(self.vocab)
        
        # Create mapping
        self.id_to_phoneme = {i: phoneme for i, phoneme in enumerate(self.vocab)}
        self.phoneme_to_id = {phoneme: i for i, phoneme in enumerate(self.vocab)}
    
    def decode(self, indices, collapse_repeats=True, remove_blank=True):
        """
        Decode phoneme indices to string.
        
        Args:
            indices: List or tensor of phoneme indices
            collapse_repeats: Whether to collapse consecutive repeats (CTC decoding)
            remove_blank: Whether to remove blank tokens
            
        Returns:
            Decoded phoneme string
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        if collapse_repeats:
            # CTC-style: remove consecutive duplicates
            indices = [indices[0]] + [indices[i] for i in range(1, len(indices)) 
                                     if indices[i] != indices[i-1]]
        
        phonemes = []
        for idx in indices:
            if remove_blank and idx == self.blank_id:
                continue
            if idx < len(self.id_to_phoneme):
                phonemes.append(self.id_to_phoneme[idx])
        
        return ' '.join(phonemes)


class CharTokenizer:
    """
    Character-level tokenizer for end-to-end text generation.
    
    Vocabulary:
    0: PAD
    1: BOS (begin of sequence)
    2: EOS (end of sequence)
    3-28: a-z
    29: space
    30+: punctuation
    """
    
    def __init__(self):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        
        # Build vocabulary
        self.chars = ['<PAD>', '<BOS>', '<EOS>']
        self.chars += list(string.ascii_lowercase)  # a-z
        self.chars += [' ']  # space
        self.chars += list("'.,!?-")  # basic punctuation
        
        self.char2id = {c: i for i, c in enumerate(self.chars)}
        self.id2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    
    def encode(self, text):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            Tensor of token IDs
        """
        text = text.lower()
        ids = [self.bos_id]
        for c in text:
            if c in self.char2id:
                ids.append(self.char2id[c])
            else:
                # Unknown chars → space
                ids.append(self.char2id[' '])
        ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)
    
    def decode(self, ids):
        """
        Decode token IDs to text.
        
        Args:
            ids: List or tensor of token IDs
            
        Returns:
            Decoded text string
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
        
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if i > 2:  # Skip PAD, BOS, EOS
                chars.append(self.id2char.get(i, ' '))
        
        return ''.join(chars)

