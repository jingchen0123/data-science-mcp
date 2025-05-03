from setuptools import setup, find_packages

setup(
    name="data_science_mcp",
    version="0.1.0",
    description="Data Science MCP Server",
    author="Jackie You",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "mcp>=1.2.0",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "data_science_mcp=data_science_mcp.server:main",
        ],
    },
)