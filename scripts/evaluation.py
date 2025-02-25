import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from pathlib import Path
import json
from datetime import datetime
from src.model import FinancialQLoRAModel
from src.metrics import FinancialMetrics
from src.utils import FinancialLogger

class ModelEvaluator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.logger = FinancialLogger("evaluation")
        self.metrics = FinancialMetrics()
        self.results_dir = Path("results/evaluations")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load trained model"""
        model_handler = FinancialQLoRAModel(self.config)
        return model_handler.get_model_and_tokenizer()

    def evaluate_model(self, test_data: list):
        """Evaluate model on test dataset"""
        model, tokenizer = self.load_model()
        predictions = []
        references = []
        
        for sample in test_data:
            # Generate prediction
            inputs = tokenizer(
                sample['input'],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=200)
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(pred_text)
                references.append(sample['output'])

        # Calculate metrics
        metrics = self.metrics.evaluate_predictions(predictions, references)
        
        # Calculate analysis quality
        quality_scores = [
            self.metrics.calculate_analysis_quality(pred)
            for pred in predictions
        ]
        metrics['avg_analysis_quality'] = np.mean(quality_scores)
        
        return metrics, predictions

    def save_evaluation_results(self, metrics: Dict, predictions: List):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"eval_results_{timestamp}.json"
        
        results = {
            'metrics': metrics,
            'predictions': predictions,
            'timestamp': timestamp,
            'model_config': self.config['model']
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Evaluation results saved to {results_file}")
        return results_file

def main():
    evaluator = ModelEvaluator()
    
    # Load test data
    test_data = [
        {
            'input': 'Test financial text...',
            'output': 'Expected analysis...'
        }
        # Add more test samples
    ]
    
    try:
        # Run evaluation
        metrics, predictions = evaluator.evaluate_model(test_data)
        
        # Save results
        results_file = evaluator.save_evaluation_results(metrics, predictions)
        
        # Print summary
        print("\nEvaluation Results:")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        print(f"Average Confidence: {metrics['avg_confidence']:.3f}")
        print(f"Analysis Quality: {metrics['avg_analysis_quality']:.3f}")
        
    except Exception as e:
        evaluator.logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
