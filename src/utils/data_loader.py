import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
from ..preprocessing.dataset import SpeechDataset
from .logger import ExperimentLogger

class DataManager:
    """Manages data loading and splitting for speech enhancement."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[ExperimentLogger] = None
    ):
        """
        Initialize data manager.
        
        Args:
            config: Configuration dictionary containing data parameters
            logger: Optional experiment logger for tracking
        """
        self.config = config
        self.logger = logger
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create dataset
        self.dataset = SpeechDataset(
            clean_dir=config['data']['clean_dir'],
            noisy_dir=config['data']['noisy_dir'],
            sample_rate=config['data']['sample_rate'],
            segment_length=config['data'].get('segment_length'),
            stft_params=config['preprocessing']
        )
        
        # Log dataset information
        if self.logger:
            dataset_info = {
                "timestamp": self.timestamp,
                "total_files": len(self.dataset),
                "sample_rate": config['data']['sample_rate'],
                "clean_dir": config['data']['clean_dir'],
                "noisy_dir": config['data']['noisy_dir'],
                "stft_params": config['preprocessing']
            }
            self.logger.log_experiment_config({"dataset": dataset_info})
        
    def get_dataloaders(
        self,
        batch_size: int,
        val_split: float = 0.1,
        num_workers: int = 4,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation dataloaders.
        
        Args:
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            num_workers: Number of worker processes for data loading
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Calculate split sizes
        val_size = int(len(self.dataset) * val_split)
        train_size = len(self.dataset) - val_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Log split information
        if self.logger:
            split_info = {
                "train_size": train_size,
                "val_size": val_size,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "seed": seed
            }
            self.logger.log_experiment_config({"data_split": split_info})
        
        return train_loader, val_loader
        
    def save_dataset_info(self, output_dir: str):
        """
        Save dataset information to a JSON file.
        
        Args:
            output_dir: Directory to save the information
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        info = {
            "timestamp": self.timestamp,
            "total_files": len(self.dataset),
            "clean_files": [str(f) for f in self.dataset.clean_files],
            "noisy_files": [str(f) for f in self.dataset.noisy_files],
            "sample_rate": self.config['data']['sample_rate'],
            "stft_params": self.config['preprocessing']
        }
        
        info_file = output_path / f"dataset_info_{self.timestamp}.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
            
        if self.logger:
            self.logger.log_experiment_config({"dataset_info_file": str(info_file)})