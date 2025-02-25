from setuptools import setup, find_packages

setup(
    name="financial_qlora",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.3",
        "bitsandbytes>=0.41.1",
        "peft>=0.4.0",
        "trl>=0.7.2",
        "PyPDF2>=3.0.0",
        "pandas>=1.5.3",
        "lightning>=2.0.0",
        "wandb>=0.15.8",
        "scikit-learn>=1.0.2",
        "numpy>=1.21.0",
        "yaml>=6.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="QLoRA fine-tuning for financial shenanigans detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
