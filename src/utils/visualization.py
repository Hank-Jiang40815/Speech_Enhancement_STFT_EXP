import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import librosa
import librosa.display
from typing import Optional, Tuple

class AudioVisualizer:
    """Tools for visualizing audio signals and spectrograms."""
    
    def __init__(self, output_dir: str = "experiments/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def plot_spectrogram(
        self,
        magnitude: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        title: str = "Spectrogram",
        save_name: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot magnitude spectrogram and optionally phase.
        
        Args:
            magnitude: Magnitude spectrogram [batch, freq, time]
            phase: Optional phase spectrogram [batch, freq, time]
            title: Plot title
            save_name: If provided, save plot with this name
            
        Returns:
            Figure and axes objects
        """
        if magnitude.dim() == 3:
            magnitude = magnitude[0]  # Take first item if batched
        if phase is not None and phase.dim() == 3:
            phase = phase[0]
            
        magnitude = magnitude.cpu().numpy()
        
        if phase is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            im = librosa.display.specshow(
                librosa.amplitude_to_db(magnitude, ref=np.max),
                y_axis='log',
                x_axis='time',
                ax=ax
            )
            plt.colorbar(im, ax=ax, format='%+2.0f dB')
            ax.set_title(title)
        else:
            phase = phase.cpu().numpy()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot magnitude
            im1 = librosa.display.specshow(
                librosa.amplitude_to_db(magnitude, ref=np.max),
                y_axis='log',
                x_axis='time',
                ax=ax1
            )
            plt.colorbar(im1, ax=ax1, format='%+2.0f dB')
            ax1.set_title(f"{title} - Magnitude")
            
            # Plot phase
            im2 = librosa.display.specshow(
                phase,
                y_axis='log',
                x_axis='time',
                ax=ax2
            )
            plt.colorbar(im2, ax=ax2)
            ax2.set_title(f"{title} - Phase")
            
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}_{self.timestamp}.png"
            plt.savefig(save_path)
            
        return fig, ax if phase is None else (ax1, ax2)
        
    def plot_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        title: str = "Waveform",
        save_name: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot audio waveform.
        
        Args:
            waveform: Audio tensor [channels, samples]
            sample_rate: Audio sample rate
            title: Plot title
            save_name: If provided, save plot with this name
            
        Returns:
            Figure and axes objects
        """
        if waveform.dim() == 2:
            waveform = waveform[0]  # Take first channel if multi-channel
            
        waveform = waveform.cpu().numpy()
        duration = len(waveform) / sample_rate
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.linspace(0, duration, len(waveform)), waveform)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        
        if save_name:
            save_path = self.output_dir / f"{save_name}_{self.timestamp}.png"
            plt.savefig(save_path)
            
        return fig, ax
        
    def compare_spectrograms(
        self,
        original_mag: torch.Tensor,
        enhanced_mag: torch.Tensor,
        title: str = "Spectrogram Comparison",
        save_name: Optional[str] = None
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Compare original and enhanced spectrograms.
        
        Args:
            original_mag: Original magnitude spectrogram
            enhanced_mag: Enhanced magnitude spectrogram
            title: Plot title
            save_name: If provided, save plot with this name
            
        Returns:
            Figure and axes objects
        """
        if original_mag.dim() == 3:
            original_mag = original_mag[0]
        if enhanced_mag.dim() == 3:
            enhanced_mag = enhanced_mag[0]
            
        original_mag = original_mag.cpu().numpy()
        enhanced_mag = enhanced_mag.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot original
        im1 = librosa.display.specshow(
            librosa.amplitude_to_db(original_mag, ref=np.max),
            y_axis='log',
            x_axis='time',
            ax=ax1
        )
        plt.colorbar(im1, ax=ax1, format='%+2.0f dB')
        ax1.set_title(f"{title} - Original")
        
        # Plot enhanced
        im2 = librosa.display.specshow(
            librosa.amplitude_to_db(enhanced_mag, ref=np.max),
            y_axis='log',
            x_axis='time',
            ax=ax2
        )
        plt.colorbar(im2, ax=ax2, format='%+2.0f dB')
        ax2.set_title(f"{title} - Enhanced")
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}_{self.timestamp}.png"
            plt.savefig(save_path)
            
        return fig, (ax1, ax2)