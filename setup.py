from setuptools import setup, find_packages

setup(
    name="schema-search",
    version="0.1.7",
    description="Natural language database schema search with graph-aware semantic retrieval",
    author="Adib Hasan",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://adibhasan.com/blog/schema-search/",
    packages=find_packages(),
    install_requires=[
        "sqlalchemy>=1.4.0",
        "sentence-transformers>=2.2.0",
        "networkx>=2.8.0",
        "bm25s>=0.2.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "openai>=1.0.0",
        "rapidfuzz>=3.0.0",
    ],
    extras_require={
        "mcp": [
            "fastmcp>=2.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "python-dotenv>=1.0.0",
            "psutil>=5.9.0",
            "datasets>=2.0.0",
        ],
        "postgres": [
            "psycopg2-binary>=2.9.0",
        ],
        "mysql": [
            "pymysql>=1.0.0",
        ],
        "snowflake": [
            "snowflake-sqlalchemy>=1.4.0",
            "snowflake-connector-python>=3.0.0",
        ],
        "bigquery": [
            "sqlalchemy-bigquery>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "schema-search=schema_search.mcp_server:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
