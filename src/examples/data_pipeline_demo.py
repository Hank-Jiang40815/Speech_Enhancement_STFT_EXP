import torch
import json
from pathlib import Path
from ..utils.data_loader import DataManager
from ..utils.visualization import AudioVisualizer
from ..utils.logger import ExperimentLogger

def main():
    # Initialize logger
    logger = ExperimentLogger("data_pipeline_demo")
    
    # Load configuration
    with open("experiments/configs/default_config.json", 'r') as f:
        config = json.load(f)
        
    # Initialize data manager
    data_manager = DataManager(config, logger)
    
    # Get dataloaders
    train_loader, val_loader = data_manager.get_dataloaders(
        batch_size=config['training']['batch_size'],
        val_split=config['data']['val_split']
    )
    
    # Initialize visualizer
    visualizer = AudioVisualizer()
    
    # Get a batch of data and visualize
    for batch_idx, (noisy_mag, noisy_phase, clean_mag, clean_phase) in enumerate(train_loader):
        if batch_idx == 0:  # Only process first batch
            # Log batch information
            batch_info = {
                "noisy_magnitude_shape": list(noisy_mag.shape),
                "noisy_phase_shape": list(noisy_phase.shape),
                "clean_magnitude_shape": list(clean_mag.shape),
                "clean_phase_shape": list(clean_phase.shape)
            }
            logger.log_experiment_config({"batch_info": batch_info})
            
            # Visualize spectrograms
            visualizer.plot_spectrogram(
                noisy_mag,
                noisy_phase,
                title="Noisy Speech",
                save_name="noisy_spectrogram"
            )
            
            visualizer.plot_spectrogram(
                clean_mag,
                clean_phase,
                title="Clean Speech",
                save_name="clean_spectrogram"
            )
            
            # Compare spectrograms
            visualizer.compare_spectrograms(
                noisy_mag,
                clean_mag,
                title="Noisy vs Clean Magnitude Spectrograms",
                save_name="spectrogram_comparison"
            )
            
            break
            
    # Save dataset information
    data_manager.save_dataset_info("experiments/data_info")
    
    print("Data pipeline demo completed. Check experiments/visualizations for plots.")
    
if __name__ == "__main__":
    main()