## Financial Fraud Detection with QLoRA on MacOS for FInu-tunning (LLMs)
🚀 QLoRA (Quantized Low-Rank Adaptation) enables efficient fine-tuning of large language models (LLMs) on Financial Shenanigans. This repository provides a streamlined setup specifically for MacOS users for fiancial fraud detection in financial reports.

## 📌 Features
✅ Fine-tune LLMs on MacOS using QLoRA
✅ Optimized for Apple M1/M2/M3 chips (Metal backend support)
✅ Uses 4-bit quantization for memory efficiency
✅ Compatible with PyTorch, Hugging Face Transformers, and PEFT
✅ Step-by-step implementation with minimal resource usage

## ⚙️ Installation
1️⃣ Set Up Virtual Environment
python3 -m venv qlora-macos
source qlora-macos/bin/activate

## 2️⃣ Install Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft bitsandbytes datasets accelerate
