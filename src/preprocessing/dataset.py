import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from .stft import STFTProcessor

class SpeechDataset(Dataset):
    """Dataset for speech enhancement training."""
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        sample_rate: int = 16000,
        segment_length: Optional[int] = None,
        normalize: bool = True,
        stft_params: Optional[dict] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            clean_dir: Directory containing clean audio files
            noisy_dir: Directory containing noisy audio files
            sample_rate: Target sample rate
            segment_length: Length of audio segments (samples), if None use full file
            normalize: Whether to normalize audio to [-1, 1]
            stft_params: Parameters for STFT transformation
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.normalize = normalize
        
        # Setup STFT processor
        self.stft = STFTProcessor(**(stft_params or {}))
        
        # Get file pairs
        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        self.noisy_files = sorted(list(self.noisy_dir.glob("*.wav")))
        
        if len(self.clean_files) == 0 or len(self.noisy_files) == 0:
            raise ValueError("No WAV files found in the specified directories")
            
        if len(self.clean_files) != len(self.noisy_files):
            raise ValueError("Number of clean and noisy files must match")
            
    def __len__(self) -> int:
        return len(self.clean_files)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of clean and noisy spectrograms.
        
        Returns:
            Tuple containing:
            - noisy magnitude
            - noisy phase
            - clean magnitude
            - clean phase
        """
        # Load audio files
        clean_audio, sr = torchaudio.load(self.clean_files[idx])
        noisy_audio, sr = torchaudio.load(self.noisy_files[idx])
        
        # Ensure mono
        if clean_audio.size(0) > 1:
            clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
        if noisy_audio.size(0) > 1:
            noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            clean_audio = resampler(clean_audio)
            noisy_audio = resampler(noisy_audio)
            
        # Normalize if requested
        if self.normalize:
            clean_audio = clean_audio / torch.max(torch.abs(clean_audio))
            noisy_audio = noisy_audio / torch.max(torch.abs(noisy_audio))
            
        # Random segment if specified
        if self.segment_length is not None:
            if clean_audio.size(1) > self.segment_length:
                start = torch.randint(0, clean_audio.size(1) - self.segment_length, (1,))
                clean_audio = clean_audio[:, start:start + self.segment_length]
                noisy_audio = noisy_audio[:, start:start + self.segment_length]
            else:
                # Pad if audio is shorter than segment length
                clean_audio = torch.nn.functional.pad(
                    clean_audio, (0, self.segment_length - clean_audio.size(1))
                )
                noisy_audio = torch.nn.functional.pad(
                    noisy_audio, (0, self.segment_length - noisy_audio.size(1))
                )
                
        # Apply STFT
        noisy_mag, noisy_phase = self.stft.stft(noisy_audio)
        clean_mag, clean_phase = self.stft.stft(clean_audio)
        
        return noisy_mag, noisy_phase, clean_mag, clean_phase