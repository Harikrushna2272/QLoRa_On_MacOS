import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from typing import Dict

class FinancialQLoRAModel:
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['model']['name']
        self.device = torch.device(config['model']['device'])
        
        # Initialize tokenizer and model
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()

    def _setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=True,
            padding_side="right",
            pad_token="</s>"
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _setup_model(self):
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['model']['load_in_4bit'],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=True,
            torch_dtype=torch.float16
        )

        # Prepare for training
        model = prepare_model_for_kbit_training(model)

        # Add LoRA adapters
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )
        
        model = get_peft_model(model, lora_config)
        return model

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer
