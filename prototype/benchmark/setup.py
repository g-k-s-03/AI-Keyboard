from setuptools import setup, find_packages

setup(
    name="slm-eval",
    version="0.2.0",
    description="CLI benchmark tool for on-device SLM evaluation - AOSSIE AI Keyboard",
    author="Govind",
    packages=find_packages(),
    package_data={
        "slm_eval": ["datasets/*.json"],
    },
    install_requires=[
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "accelerate>=0.26.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
        "pandas>=2.0.0",
        "sacrebleu>=2.3.0",
        "rouge-score>=0.1.2",
        "click>=8.0.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": ["slm-eval=slm_eval.cli:main"],
    },
    python_requires=">=3.9",
)
