"""
Setup script for Neural Retriever-Reranker RAG Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="neural-retriever-reranker-rag",
    version="1.0.0",
    author="Teri Rumble, Zbyněk Gazdík, Javad Zarrin, Jagdeep Ahluwalia",
    author_email="teri.rumble@gmail.com",
    description="Neural Retriever-Reranker Pipelines for RAG over Knowledge Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-retriever-reranker-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'gpu': ['faiss-gpu>=1.7.0'],
        'dev': ['pytest>=6.2.0', 'black>=22.0.0', 'flake8>=4.0.0', 'mypy>=0.910'],
        'notebooks': ['jupyter>=1.0.0', 'ipykernel>=6.0.0'],
        'viz': ['matplotlib>=3.5.0', 'seaborn>=0.11.0']
    },
    entry_points={
        'console_scripts': [
            'rag-evaluate=scripts.run_evaluation:main',
            'rag-build-index=scripts.build_indices:main',
            'rag-download-data=scripts.download_data:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.txt', '*.md'],
    },
    keywords=[
        'retrieval-augmented-generation', 'rag', 'information-retrieval', 
        'knowledge-graphs', 'cross-encoders', 'faiss', 'sentence-transformers',
        'e-commerce', 'natural-language-processing', 'machine-learning'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/neural-retriever-reranker-rag/issues',
        'Source': 'https://github.com/yourusername/neural-retriever-reranker-rag',
        'Documentation': 'https://github.com/yourusername/neural-retriever-reranker-rag/docs',
    },
)