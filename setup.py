from setuptools import setup, find_packages

setup(
    name="schema-search",
    version="0.1.0",
    description="Lightweight database metadata search with embeddings and graph-based discovery",
    author="",
    packages=find_packages(),
    install_requires=[
        "sqlalchemy>=1.4.0",
        "sentence-transformers>=2.2.0",
        "networkx>=2.8.0",
        "rank-bm25>=0.2.2",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "anthropic>=0.40.0",
        "python-dotenv>=1.0.0",
        "rapidfuzz>=3.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "python-dotenv>=1.0.0",
        ],
        "postgres": [
            "psycopg2-binary>=2.9.0",
        ],
        "mysql": [
            "pymysql>=1.0.0",
        ],
    },
    python_requires=">=3.8",
)
