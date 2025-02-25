# scripts/inference.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict, Optional, Generator
import time
from src.utils import (
    FinancialLogger,
    setup_financial_device,
    format_financial_output
)

class FinancialAnalysisEngine:
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-2-8b",
        adapter_path: str = "checkpoints/financial_model",
    ):
        self.logger = FinancialLogger(__name__)
        self.device = setup_financial_device()
        
        # Initialize model components
        self.tokenizer = self._setup_tokenizer(base_model_name)
        self.model = self._setup_model(base_model_name, adapter_path)
        
        # Financial analysis templates
        self.templates = self._load_analysis_templates()
        
    def _setup_tokenizer(self, base_model_name: str):
        """Initialize tokenizer"""
        return AutoTokenizer.from_pretrained(
            base_model_name,
            use_auth_token=True,
            padding_side="right",
            pad_token="</s>"
        )

    def _setup_model(self, base_model_name: str, adapter_path: str):
        """Initialize quantized model with financial adapters"""
        try:
            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                use_auth_token=True,
                torch_dtype=torch.float16
            )
            
            # Load financial adapters
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16
            )
            
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def _load_analysis_templates(self) -> Dict[str, str]:
        """Load financial analysis templates"""
        return {
            "financial_shenanigans": (
                "Analyze the following financial text for potential shenanigans "
                "and irregularities. Focus on:\n"
                "1. Revenue recognition issues\n"
                "2. Expense manipulation\n"
                "3. Cash flow discrepancies\n"
                "4. Balance sheet anomalies\n\n"
                "Text: {text}\n\n"
                "Analysis:"
            ),
            "fraud_detection": (
                "Examine the following financial information for potential "
                "fraud indicators:\n{text}\n\n"
                "Identify red flags and explain their significance:"
            ),
            "risk_assessment": (
                "Assess the financial risks in the following text:\n{text}\n\n"
                "Provide a detailed risk analysis covering:\n"
                "- Market risks\n"
                "- Operational risks\n"
                "- Compliance risks\n"
                "- Financial statement risks"
            )
        }

    def analyze_financial_document(
        self,
        document: str,
        analysis_type: str = "financial_shenanigans",
        max_length: int = 1024
    ) -> Dict:
        """Analyze financial document with specific focus"""
        try:
            # Prepare chunks
            chunks = self._prepare_document_chunks(document, max_length)
            analyses = []
            
            # Analyze each chunk
            for chunk in chunks:
                template = self.templates[analysis_type]
                prompt = template.format(text=chunk)
                
                response = self._generate_analysis(
                    prompt,
                    max_length=max_length
                )
                
                analyses.append(response)
            
            # Synthesize findings
            final_analysis = self._synthesize_analyses(analyses)
            
            return {
                "detailed_analyses": analyses,
                "summary": format_financial_output(final_analysis),
                "risk_score": self._calculate_risk_score(final_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _generate_analysis(
        self,
        prompt: str,
        max_length: int = 1024,
        temperature: float = 0.3
    ) -> str:
        """Generate financial analysis"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _prepare_document_chunks(
        self,
        document: str,
        max_length: int
    ) -> List[str]:
        """Prepare document chunks for analysis"""
        tokens = self.tokenizer.encode(document)
        chunks = []
        
        for i in range(0, len(tokens), max_length - 100):
            chunk_tokens = tokens[i:i + max_length - 100]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    def _synthesize_analyses(self, analyses: List[str]) -> str:
        """Synthesize multiple analyses into coherent findings"""
        synthesis_prompt = (
            "Synthesize the following financial analyses into a "
            "comprehensive report:\n\n" +
            "\n---\n".join(analyses) +
            "\n\nProvide a consolidated analysis focusing on:"
            "\n1. Key findings"
            "\n2. Risk factors"
            "\n3. Recommendations"
        )
        
        return self._generate_analysis(synthesis_prompt)

    def _calculate_risk_score(self, analysis: str) -> float:
        """Calculate risk score based on analysis content"""
        risk_indicators = [
            "fraud", "manipulation", "irregular", "suspicious",
            "misleading", "inconsistent", "unusual", "aggressive"
        ]
        
        risk_score = 0
        analysis_lower = analysis.lower()
        
        for indicator in risk_indicators:
            risk_score += analysis_lower.count(indicator) * 0.1
            
        return min(max(risk_score, 0), 1)

    def analyze_with_streaming(
        self,
        document: str,
        analysis_type: str = "financial_shenanigans"
    ) -> Generator[str, None, None]:
        """Stream analysis results"""
        template = self.templates[analysis_type]
        prompt = template.format(text=document)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        for output in self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=1024,
            temperature=0.3,
            streaming=True
        ):
            yield self.tokenizer.decode([output], skip_special_tokens=True)

def main():
    """Example usage"""
    engine = FinancialAnalysisEngine()
    
    # Example financial text
    financial_text = """
    The company reported a 45% increase in revenues while operating 
    cash flows decreased by 15%. Accounts receivable aging showed 
    unusual patterns with a significant portion beyond 180 days. 
    The notes to financial statements revealed several one-time 
    gains classified as recurring operating income.
    """
    
    # Regular analysis
    print("Analyzing financial document...")
    analysis = engine.analyze_financial_document(
        financial_text,
        analysis_type="financial_shenanigans"
    )
    
    print("\nAnalysis Summary:")
    print(analysis["summary"])
    print(f"\nRisk Score: {analysis['risk_score']:.2f}")
    
    # Streaming analysis
    print("\nStreaming Analysis:")
    for token in engine.analyze_with_streaming(financial_text):
        print(token, end="", flush=True)

if __name__ == "__main__":
    main()
