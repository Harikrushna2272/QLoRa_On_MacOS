import os
import torch
import fitz
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer
import logging
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)


@dataclass
class ScriptArguments:
    pdf_path: Optional[str] = field(default="financial_shenenigm.pdf")
    per_device_train_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    use_4bit: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=3)
    output_dir: str = field(default="./financial_results")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        if not text.strip():
            raise ValueError("Extracted text is empty. Check the PDF content.")
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")


def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into overlapping chunks for training."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]


def prepare_financial_dataset(pdf_path):
    """Process PDF into a dataset for LLaMA fine-tuning."""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=512, overlap=50)

    processed_texts = [
        f"Below is a section from a financial document. Analyze and explain this information:\n\n{chunk}\n\nAnalysis:"
        for chunk in chunks
    ]

    return Dataset.from_dict({"text": processed_texts})


def create_and_prepare_model(args):
    os.makedirs(args.output_dir, exist_ok=True)

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=True,
        trust_remote_code=True  # ✅ Fix
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def train_model(args):
    model, peft_config, tokenizer = create_and_prepare_model(args)

    dataset = prepare_financial_dataset(args.pdf_path)

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=True
    )

    trainer.train()
    return args.output_dir


def run_inference(model_path, text):
    """Run inference on given text using the saved model."""
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(  # ✅ Fix
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

        prompt = f"Below is a section from a financial document. Analyze and explain this information:\n\n{text}\n\nAnalysis:"
        return pipe(prompt)[0]['generated_text'].split("Analysis:")[-1].strip()

    except Exception as e:
        return f"Error during inference: {e}"

def save_model(trainer, args):
    """Save the model, tokenizer and merge weights if specified"""
    print("Saving model...")
    final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
    merged_model_dir = os.path.join(args.output_dir, "merged_model")
    
    # Save the LoRA adaptations
    trainer.model.save_pretrained(final_checkpoint_dir)
    trainer.tokenizer.save_pretrained(final_checkpoint_dir)
    
    # Merge weights and save full model
    try:
        print("Merging model weights...")
        # Load the LoRA model
        model = AutoPeftModelForCausalLM.from_pretrained(
            final_checkpoint_dir,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Merge LoRA weights with base model
        merged_model = model.merge_and_unload()
        
        # Save the merged model
        merged_model.save_pretrained(merged_model_dir)
        trainer.tokenizer.save_pretrained(merged_model_dir)
        print(f"Merged model saved to {merged_model_dir}")
        
    except Exception as e:
        print(f"Error merging model weights: {e}")
        
    return final_checkpoint_dir, merged_model_dir

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    login(os.getenv("HF_TOKEN", "your_token_here"))

    model_path = train_model(script_args)

    checkpoint_dir, merged_dir = save_model(trainer, args)

    test_text = "The company reported a 25% increase in revenue to $1.2 billion."
    print("\nModel Analysis:", run_inference(model_path, test_text))
