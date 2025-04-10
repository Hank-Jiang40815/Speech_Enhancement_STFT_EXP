import torch
import torchaudio
import numpy as np
from typing import Tuple

class STFTProcessor:
    """STFT-based audio processing class for speech enhancement."""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        sample_rate: int = 16000,
        center: bool = True,
    ):
        """
        Initialize STFT processor.
        
        Args:
            n_fft (int): Number of FFT points
            hop_length (int): Number of samples between successive frames
            win_length (int): Window length for STFT
            sample_rate (int): Audio sample rate
            center (bool): Whether to pad input on both sides
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.center = center
        self.window = torch.hann_window(win_length)
        
    def stft(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform STFT on input audio.
        
        Args:
            audio (torch.Tensor): Input audio tensor [B, T]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Magnitude and phase spectrograms
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Compute STFT
        complex_spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True
        )
        
        # Split into magnitude and phase
        magnitude = complex_spec.abs()
        phase = complex_spec.angle()
        
        return magnitude, phase
    
    def istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Perform inverse STFT.
        
        Args:
            magnitude (torch.Tensor): Magnitude spectrogram
            phase (torch.Tensor): Phase spectrogram
            
        Returns:
            torch.Tensor: Reconstructed audio
        """
        # Combine magnitude and phase
        complex_spec = torch.polar(magnitude, phase)
        
        # Inverse STFT
        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center
        )
        
        return audio