import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any

class ExperimentLogger:
    """Logger for tracking experiment progress and results."""
    
    def __init__(self, exp_name: str, log_dir: str = "experiments"):
        """
        Initialize experiment logger.
        
        Args:
            exp_name (str): Name of the experiment
            log_dir (str): Directory to store logs
        """
        self.exp_name = exp_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = f"{exp_name}_{self.timestamp}"
        
        # Configure file handler
        log_file = self.log_dir / "experiments.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.exp_id)
        
    def log_experiment_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.logger.info(f"Starting experiment: {self.exp_id}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Update REPORT.md
        self._update_report(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training/validation metrics."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")
    
    def log_experiment_results(self, results: Dict[str, Any]):
        """Log final experiment results."""
        self.logger.info(f"Experiment {self.exp_id} completed")
        self.logger.info(f"Results: {json.dumps(results, indent=2)}")
        
        # Update REPORT.md with results
        self._update_report(results, is_result=True)
    
    def _update_report(self, data: Dict[str, Any], is_result: bool = False):
        """Update REPORT.md with experiment information."""
        report_path = Path("REPORT.md")
        if not report_path.exists():
            return
            
        current_content = report_path.read_text().split("\n")
        
        # Find the Latest Experiment section and update it
        for i, line in enumerate(current_content):
            if line.startswith("### Latest Experiment"):
                current_content.insert(i + 1, f"- Date: {self.timestamp}")
                current_content.insert(i + 2, f"- Version: {self.exp_name}")
                if is_result:
                    result_summary = "\n".join([f"  - {k}: {v}" for k, v in data.items()])
                    current_content.insert(i + 3, f"- Results:\n{result_summary}")
                else:
                    config_summary = "\n".join([f"  - {k}: {v}" for k, v in data.items()])
                    current_content.insert(i + 3, f"- Configuration:\n{config_summary}")
                break
                
        report_path.write_text("\n".join(current_content))