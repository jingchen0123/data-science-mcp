"""
Data loading and management tools for the Data Science MCP.
This module contains tools for loading, accessing, and managing CSV datasets.
"""

from mcp.server.fastmcp import FastMCP, Context, Image
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tempfile
from typing import Optional, Dict, List, Any, Union

# The DATA_DIR will be initialized by the main server module
# This is a placeholder that will be overwritten
DATA_DIR = None
# The MCP instance will be initialized by the main server module
mcp = None

# Simple in-memory data store as a replacement for Context user data
_user_data = {}

def set_user_data(key, value):
    """Store data in the module-level dictionary"""
    global _user_data
    _user_data[key] = value

def get_user_data(key, default=None):
    """Retrieve data from the module-level dictionary"""
    global _user_data
    return _user_data.get(key, default)


def initialize(mcp_instance: FastMCP, data_dir: Path):
    """
    Initialize this module with the MCP instance and data directory.
    
    Args:
        mcp_instance: The FastMCP instance
        data_dir: Path to the data directory
    """
    global DATA_DIR, mcp
    DATA_DIR = data_dir
    mcp = mcp_instance
    
    # Register all tools with the MCP instance
    mcp_instance.add_tool(load_csv)
    mcp_instance.add_tool(list_datasets)
    mcp_instance.add_tool(get_dataset_info)
    mcp_instance.add_tool(save_csv)
    mcp_instance.add_tool(upload_csv)
    
    # Register resource using the decorator syntax
    @mcp_instance.resource("csv://{filename}")
    def csv_resource_handler(filename: str):
        return get_csv_data(filename)

def load_csv(filename: str, ctx: Context) -> str:
    """
    Load a CSV file and make it available as a resource.
    
    Args:
        filename: Name of the CSV file (with or without .csv extension)
    
    Returns:
        A message confirming the file was loaded
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    # Check if file exists
    if not filepath.exists():
        return f"Error: File {filename} not found in the data directory."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Store file info in our state management
        set_user_data(f"csv_{filename}", {
            "path": str(filepath),
            "columns": list(df.columns),
            "rows": len(df),
            "last_accessed": pd.Timestamp.now().isoformat()
        })
        
        # Set as active dataset
        set_user_data("active_dataset", {
            "name": filename,
            "path": str(filepath),
            "columns": list(df.columns),
            "rows": len(df)
        })
        
        # Return success message with file info
        return f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns: {', '.join(df.columns)}"
    
    except Exception as e:
        return f"Error loading CSV: {str(e)}"


def get_csv_data(filename: str) -> str:
    """
    Access a CSV file as a resource.
    
    Args:
        filename: Name of the CSV file
        
    Returns:
        CSV data as a formatted string
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return f"Error: File {filename} not found."
    
    try:
        df = pd.read_csv(filepath)
        # Format the first 10 rows as a readable string
        preview = df.head(10).to_string()
        
        # Add summary statistics
        summary = f"""
# CSV Data: {filename}
Shape: {df.shape[0]} rows × {df.shape[1]} columns

## Preview (first 10 rows):
{preview}

## Summary Statistics:
{df.describe().to_string()}
"""
        return summary
    except Exception as e:
        return f"Error reading CSV data: {str(e)}"


def list_datasets(ctx: Context) -> str:
    """
    List all available datasets in the data directory.
    
    Returns:
        Formatted list of available datasets
    """
    try:
        # Get all CSV files in the data directory
        csv_files = list(DATA_DIR.glob("*.csv"))
        
        if not csv_files:
            return "No datasets found in the data directory."
        
        # Format the list with basic information
        result = "# Available Datasets\n\n"
        result += "| Filename | Rows | Columns | Last Accessed |\n"
        result += "|----------|------|---------|---------------|\n"
        
        for file_path in csv_files:
            filename = file_path.name
            
            # Try to get info from our state management
            file_info = get_user_data(f"csv_{filename}", None)
            
            if file_info:
                # Use cached info
                rows = file_info.get("rows", "?")
                columns = len(file_info.get("columns", []))
                last_accessed = file_info.get("last_accessed", "Never")
            else:
                # Read file to get info
                try:
                    df = pd.read_csv(file_path)
                    rows = len(df)
                    columns = len(df.columns)
                    last_accessed = "Not previously accessed"
                except:
                    rows = "?"
                    columns = "?"
                    last_accessed = "Error reading file"
            
            result += f"| {filename} | {rows} | {columns} | {last_accessed} |\n"
        
        # Add note about active dataset if there is one
        active_dataset = get_user_data("active_dataset", None)
        if active_dataset:
            result += f"\nActive dataset: **{active_dataset.get('name', '')}**\n"
        
        return result
    
    except Exception as e:
        return f"Error listing datasets: {str(e)}"


def get_dataset_info(filename: str, ctx: Context) -> str:
    """
    Get detailed information about a specific dataset.
    
    Args:
        filename: Name of the CSV file
        
    Returns:
        Detailed information about the dataset
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return f"Error: File {filename} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Calculate size
        file_size = filepath.stat().st_size
        size_kb = file_size / 1024
        size_mb = size_kb / 1024
        
        # Format file info
        result = f"# Dataset Information: {filename}\n\n"
        result += f"- **Filename**: {filename}\n"
        result += f"- **Path**: {filepath}\n"
        result += f"- **Size**: {size_kb:.2f} KB ({size_mb:.2f} MB)\n"
        result += f"- **Rows**: {len(df)}\n"
        result += f"- **Columns**: {len(df.columns)}\n\n"
        
        # Column information
        result += "## Column Information\n\n"
        result += "| Column | Data Type | Non-Null Count | Memory Usage |\n"
        result += "|--------|-----------|----------------|-------------|\n"
        
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        for col in df.columns:
            col_dtype = str(df[col].dtype)
            non_null = df[col].count()
            non_null_pct = (non_null / len(df)) * 100
            mem_usage = memory_usage[df.columns.get_loc(col) + 1]  # +1 for index
            mem_pct = (mem_usage / total_memory) * 100
            
            mem_str = f"{mem_usage / 1024:.2f} KB ({mem_pct:.1f}%)"
            result += f"| {col} | {col_dtype} | {non_null} ({non_null_pct:.1f}%) | {mem_str} |\n"
        
        # Update our state management with file info
        set_user_data(f"csv_{filename}", {
            "path": str(filepath),
            "columns": list(df.columns),
            "rows": len(df),
            "last_accessed": pd.Timestamp.now().isoformat()
        })
        
        return result
    
    except Exception as e:
        return f"Error getting dataset info: {str(e)}"


def save_csv(data: Any, filename: str, overwrite: bool = False) -> str:
    """
    Save data to a CSV file in the data directory.
    
    Args:
        data: DataFrame or data that can be converted to a DataFrame
        filename: Name of the CSV file to save
        overwrite: Whether to overwrite an existing file
        
    Returns:
        Confirmation message
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    # Check if file exists and we're not overwriting
    if filepath.exists() and not overwrite:
        return f"Error: File {filename} already exists. Set overwrite=True to replace it."
    
    try:
        # Convert to DataFrame if not already
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        return f"Successfully saved {filename} with {len(df)} rows and {len(df.columns)} columns."
    
    except Exception as e:
        return f"Error saving CSV: {str(e)}"


def upload_csv(content: str, filename: str, overwrite: bool = False) -> str:
    """
    Upload CSV content as a new file in the data directory.
    
    Args:
        content: CSV content as a string
        filename: Name of the CSV file to create
        overwrite: Whether to overwrite an existing file
        
    Returns:
        Confirmation message
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    # Check if file exists and we're not overwriting
    if filepath.exists() and not overwrite:
        return f"Error: File {filename} already exists. Set overwrite=True to replace it."
    
    try:
        # Write content to file
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Verify by loading
        df = pd.read_csv(filepath)
        
        return f"Successfully uploaded {filename} with {len(df)} rows and {len(df.columns)} columns."
    
    except Exception as e:
        return f"Error uploading CSV: {str(e)}"