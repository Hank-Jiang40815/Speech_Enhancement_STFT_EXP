import os
import sys
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.data_loader import DataManager
from src.utils.logger import ExperimentLogger
from src.utils.visualization import AudioVisualizer

def main():
    # Initialize logger
    logger = ExperimentLogger("data_pipeline_demo")
    
    # Load configuration
    config = {
        'data': {
            'clean_dir': 'data/clean',
            'noisy_dir': 'data/noisy',
            'sample_rate': 16000,
            'segment_length': 16000  # 1 second segments
        },
        'preprocessing': {
            'n_fft': 512,
            'hop_length': 128,
            'win_length': 512
        }
    }
    
    # Initialize data manager
    data_manager = DataManager(config, logger)
    
    # Get data loaders
    train_loader, val_loader = data_manager.get_dataloaders(
        batch_size=4,
        val_split=0.2,
        num_workers=2
    )
    
    # Initialize visualizer
    visualizer = AudioVisualizer()
    
    # Get a batch of data and visualize
    print("Loading a batch of data...")
    noisy_mag, noisy_phase, clean_mag, clean_phase = next(iter(train_loader))
    
    print(f"Batch shapes:")
    print(f"Noisy magnitude: {noisy_mag.shape}")
    print(f"Noisy phase: {noisy_phase.shape}")
    print(f"Clean magnitude: {clean_mag.shape}")
    print(f"Clean phase: {clean_phase.shape}")
    
    # Plot and save spectrograms
    print("\nGenerating visualizations...")
    
    # Compare noisy and clean spectrograms
    visualizer.compare_spectrograms(
        noisy_mag[0],
        clean_mag[0],
        title="Noisy vs Clean Spectrogram",
        save_name="spectrogram_comparison"
    )
    
    # Plot phase spectrograms
    visualizer.plot_spectrogram(
        noisy_mag[0],
        noisy_phase[0],
        title="Noisy Audio",
        save_name="noisy_spectrograms"
    )
    
    print("\nDataset information:")
    print(f"Total number of samples: {len(data_manager.dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Save dataset information
    data_manager.save_dataset_info("experiments")
    
    print("\nDemo completed! Check the experiments/visualizations directory for output plots.")

if __name__ == "__main__":
    main()