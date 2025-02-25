import torch
from transformers import Trainer, TrainingArguments
from typing import Dict
import wandb
from src.utils import FinancialLogger

class FinancialTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        train_dataset,
        eval_dataset,
        logger: FinancialLogger
    ):
        self.model = model
        self.config = config
        self.logger = logger
        
        # Initialize wandb
        wandb.init(project="financial-shenanigans-qlora")
        
        # Setup training arguments
        self.training_args = TrainingArguments(
            output_dir=config['training']['output_dir'],
            num_train_epochs=config['training']['num_epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            warmup_ratio=config['training']['warmup_ratio'],
            logging_steps=config['training']['logging_steps'],
            eval_steps=config['training']['eval_steps'],
            save_steps=config['training']['save_steps'],
            max_grad_norm=config['training']['max_grad_norm'],
            evaluation_strategy="steps",
            logging_strategy="steps",
            save_strategy="steps",
            fp16=True,
            optim="paged_adamw_32bit"
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

    def train(self):
        try:
            self.logger.info("Starting training...")
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model(self.config['training']['output_dir'])
            
            # Log metrics
            metrics = train_result.metrics
            self.logger.info(f"Training metrics: {metrics}")
            wandb.log(metrics)
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
