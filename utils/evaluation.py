"""
Evaluation utilities for Brain-to-Text competition.

Provides functions for computing WER, PER, and other metrics.
"""

import jiwer
import numpy as np
from typing import List, Dict


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
        
    Returns:
        WER as float
    """
    return jiwer.wer(references, predictions)


def compute_per(predictions: List[str], references: List[str]) -> float:
    """
    Compute Phoneme Error Rate (using WER metric on phonemes).
    
    Args:
        predictions: List of predicted phoneme sequences (space-separated)
        references: List of reference phoneme sequences (space-separated)
        
    Returns:
        PER as float
    """
    return jiwer.wer(references, predictions)


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
        
    Returns:
        CER as float
    """
    return jiwer.cer(references, predictions)


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all metrics.
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
        
    Returns:
        Dictionary with WER, CER, and other metrics
    """
    metrics = {
        'wer': compute_wer(predictions, references),
        'cer': compute_cer(predictions, references),
    }
    
    # Additional jiwer metrics
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    try:
        measures = jiwer.compute_measures(
            references, 
            predictions, 
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
        metrics.update({
            'hits': measures['hits'],
            'substitutions': measures['substitutions'],
            'deletions': measures['deletions'],
            'insertions': measures['insertions'],
        })
    except:
        pass
    
    return metrics


def normalize_text_for_eval(text: str) -> str:
    """
    Normalize text for evaluation (remove punctuation except apostrophe).
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    import re
    text = text.lower()
    text = text.replace("'", "'")
    
    # Remove punctuation except apostrophe
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

