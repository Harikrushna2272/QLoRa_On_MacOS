import PyPDF2
from datasets import Dataset
import pandas as pd
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer

class FinancialDataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.chunk_size = config['data']['chunk_size']
        self.overlap = config['data']['overlap']

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def create_financial_samples(self, text: str) -> List[Dict]:
        words = text.split()
        samples = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            
            sample = {
                "instruction": "Analyze this financial text for potential shenanigans and fraudulent activities:",
                "input": chunk,
                "output": "Here's an analysis of the financial text and potential irregularities..."
            }
            
            samples.append(sample)
            
        return samples

    def prepare_dataset(
        self,
        tokenizer: PreTrainedTokenizer
    ) -> Tuple[Dataset, Dataset]:
        # Extract text from PDF
        text = self.extract_text_from_pdf(self.config['data']['pdf_path'])
        
        # Create samples
        samples = self.create_financial_samples(text)
        df = pd.DataFrame(samples)
        
        # Create dataset
        dataset = Dataset.from_pandas(df)
        
        # Tokenize
        def tokenize_function(examples):
            prompt = f"""### Instruction: {examples['instruction']}
            ### Input: {examples['input']}
            ### Response: {examples['output']}"""
            
            return tokenizer(
                prompt,
                truncation=True,
                max_length=self.config['model']['max_length'],
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names
        )

        # Split dataset
        split = tokenized_dataset.train_test_split(
            test_size=1-self.config['data']['train_test_split']
        )
        
        return split['train'], split['test']
