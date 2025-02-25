# src/utils.py
import logging
import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

class FinancialLogger:
    def __init__(self, name: str, log_dir: str = "logs/financial"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"financial_analysis_{timestamp}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(f"❌ {message}")

    def warning(self, message: str):
        self.logger.warning(f"⚠️ {message}")

    def success(self, message: str):
        self.logger.info(f"✅ {message}")

class FinancialModelCheckpointer:
    def __init__(self, base_dir: str = "checkpoints/financial"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None,
        filename: Optional[str] = None
    ):
        """Save model checkpoint with financial metrics"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"financial_model_{timestamp}.pt"
            
        save_path = self.base_dir / filename
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, save_path)
        return str(save_path)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: str = "best_financial_model.pt"
    ):
        """Load financial model checkpoint"""
        checkpoint_path = self.base_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint.get('epoch'), checkpoint.get('metrics', {})

class FinancialMetricsTracker:
    def __init__(self, save_dir: str = "metrics/financial"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            'financial_accuracy': [],
            'concept_understanding': [],
            'fraud_detection_rate': [],
            'analysis_quality': []
        }
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")

    def update(self, metric_name: str, value: float, step: int):
        """Update financial metrics"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })

    def save_metrics(self):
        """Save financial metrics"""
        metrics_file = self.save_dir / f"financial_metrics_{self.current_session}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get latest values for all metrics"""
        return {
            metric: entries[-1]['value'] if entries else None
            for metric, entries in self.metrics.items()
        }

def setup_financial_device():
    """Setup compute device optimized for financial analysis"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    return device

def format_financial_output(text: str) -> str:
    """Format financial analysis output"""
    sections = {
        "Key Findings": [],
        "Risk Factors": [],
        "Recommendations": []
    }
    
    current_section = "Key Findings"
    
    for line in text.split('\n'):
        line = line.strip()
        if line.lower().startswith('risk'):
            current_section = "Risk Factors"
        elif line.lower().startswith('recommend'):
            current_section = "Recommendations"
        elif line:
            sections[current_section].append(line)
    
    formatted_output = ""
    for section, points in sections.items():
        if points:
            formatted_output += f"\n{section}:\n"
            formatted_output += "\n".join(f"- {point}" for point in points)
            formatted_output += "\n"
    
    return formatted_output
