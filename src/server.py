"""
Main server file for the Data Science MCP.
This file initializes the MCP server and registers all tools and prompts.
"""

from mcp.server.fastmcp import FastMCP, Context, Image
import os
from pathlib import Path
import inspect  # Move this import up with other imports

# Import modules
import tools.data_loading as data_loading
import tools.exploration as exploration
import tools.visualization as visualization
import tools.processing as processing
import tools.code_generation as code_generation
import tools.statistical_tests as statistical_tests
import tools.document_generation as document_generation
import prompts.templates as templates

# Create an MCP server
mcp = FastMCP("Data_Science_Agent")

# Define data directory in user's home directory
HOME_DIR = Path.home()
#DATA_DIR = HOME_DIR / ".data_science_mcp"
DATA_DIR = Path("/Users/jingchenyou/Downloads/data-science-mcp/data")

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize all modules
data_loading.initialize(mcp, DATA_DIR)
exploration.initialize(mcp, DATA_DIR)
visualization.initialize(mcp, DATA_DIR)
processing.initialize(mcp, DATA_DIR)
code_generation.initialize(mcp, DATA_DIR)
statistical_tests.initialize(mcp, DATA_DIR)
document_generation.initialize(mcp, DATA_DIR)
templates.initialize(mcp)

# Patch tools from each module
for module_name, module in [
    ("data_loading", data_loading), 
    ("exploration", exploration),
    ("visualization", visualization), 
    ("processing", processing),
    ("code_generation", code_generation),
    ("statistical_tests", statistical_tests)  # Add this module to the patching list
]:
    for name, obj in inspect.getmembers(module):
        if callable(obj) and not name.startswith('_'):
            # Check if it's a tool function registered with MCP
            if hasattr(obj, '__self__') and isinstance(obj.__self__, FastMCP):
                # Replace the original function with the patched version
                setattr(module, name, document_generation.patch_tool(obj, module_name))

# Create example data
def create_example_data():
    """Create sample data files for demo purposes"""
    sample_file = DATA_DIR / "example.csv"
    if not sample_file.exists():
        with open(sample_file, "w") as f:
            f.write("age,income,education_years,satisfaction\n")
            f.write("28,65000,16,7.2\n")
            f.write("35,72000,18,8.1\n")
            f.write("42,58000,12,6.5\n")
            f.write("25,48000,16,6.8\n")
            f.write("39,95000,20,8.9\n")
            f.write("51,110000,16,7.5\n")
            f.write("33,67000,14,7.1\n")
            f.write("29,59000,16,7.8\n")
            f.write("45,71000,12,6.2\n")
            f.write("38,76000,18,8.4\n")

# Call the function to create example data
create_example_data()

if __name__ == "__main__":
    print(f"Starting Data Science MCP server...")
    print(f"Data directory: {DATA_DIR}")
    mcp.run(transport="stdio")