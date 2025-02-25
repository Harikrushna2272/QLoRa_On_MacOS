import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.model import FinancialQLoRAModel
from src.data_processor import FinancialDataProcessor
from src.trainer import FinancialTrainer
from src.utils import FinancialLogger
import torch

def main():
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Initialize logger
    logger = FinancialLogger("financial_training")
    
    try:
        # Initialize model
        logger.info("Initializing model...")
        model_handler = FinancialQLoRAModel(config)
        model, tokenizer = model_handler.get_model_and_tokenizer()
        
        # Prepare data
        logger.info("Preparing dataset...")
        data_processor = FinancialDataProcessor(config)
        train_dataset, eval_dataset = data_processor.prepare_dataset(tokenizer)
        
        # Initialize trainer
        logger.info("Setting up trainer...")
        trainer = FinancialTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.success("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
