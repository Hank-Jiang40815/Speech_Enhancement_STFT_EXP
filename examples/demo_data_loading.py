import os
import sys
import torch
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.data_loader import DataManager
from src.utils.visualization import AudioVisualizer
from src.utils.logger import ExperimentLogger

def demo_data_pipeline():
    # Create necessary directories
    os.makedirs("experiments/visualizations", exist_ok=True)
    os.makedirs("data/clean", exist_ok=True)
    os.makedirs("data/noisy", exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger("data_pipeline_demo")
    logger.log_experiment_config({
        "description": "Demonstrating data pipeline functionality",
        "timestamp": logger.timestamp
    })
    
    # Load or create default config
    config = {
        "data": {
            "sample_rate": 16000,
            "clean_dir": "data/clean",
            "noisy_dir": "data/noisy",
            "val_split": 0.1,
            "segment_length": 16000  # 1 second segments
        },
        "preprocessing": {
            "n_fft": 512,
            "hop_length": 128,
            "win_length": 512
        },
        "training": {
            "batch_size": 4
        }
    }
    
    # Initialize components
    data_manager = DataManager(config, logger)
    visualizer = AudioVisualizer()
    
    try:
        # Get dataloaders
        train_loader, val_loader = data_manager.get_dataloaders(
            batch_size=config['training']['batch_size'],
            val_split=config['data']['val_split']
        )
        
        # Process one batch
        for batch_idx, (noisy_mag, noisy_phase, clean_mag, clean_phase) in enumerate(train_loader):
            if batch_idx == 0:
                # Log shapes
                logger.log_experiment_config({
                    "batch_shapes": {
                        "noisy_magnitude": list(noisy_mag.shape),
                        "noisy_phase": list(noisy_phase.shape),
                        "clean_magnitude": list(clean_mag.shape),
                        "clean_phase": list(clean_phase.shape)
                    }
                })
                
                # Create visualizations
                visualizer.plot_spectrogram(
                    noisy_mag, 
                    noisy_phase,
                    title="Noisy Speech Example",
                    save_name="demo_noisy"
                )
                
                visualizer.plot_spectrogram(
                    clean_mag,
                    clean_phase,
                    title="Clean Speech Example",
                    save_name="demo_clean"
                )
                
                # Compare spectrograms
                visualizer.compare_spectrograms(
                    noisy_mag,
                    clean_mag,
                    title="Noisy vs Clean Comparison",
                    save_name="demo_comparison"
                )
                break
                
        print("Demo completed successfully!")
        print("Check the following locations for outputs:")
        print("- Visualizations: experiments/visualizations/")
        print("- Logs: experiments/experiments.log")
        print("- Dataset info: experiments/data_info/")
        
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        logger.logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    demo_data_pipeline()