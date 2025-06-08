"""
Data processing tools for the Data Science MCP.
This module contains tools for transforming, filtering, and manipulating datasets.
"""

from mcp.server.fastmcp import FastMCP, Context, Image
import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Optional, Dict, List, Any, Union

# The DATA_DIR will be initialized by the main server module
# This is a placeholder that will be overwritten
DATA_DIR = None
# The MCP instance will be initialized by the main server module
mcp = None

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
    mcp_instance.add_tool(filter_data)
    mcp_instance.add_tool(transform_column)
    mcp_instance.add_tool(group_and_aggregate)
    mcp_instance.add_tool(handle_missing_values)


def resolve_file_path(file_path: str, ensure_extension: str = None) -> Path:
    """
    Resolve a file path, supporting both relative (to DATA_DIR) and absolute paths.
    
    Args:
        file_path: File path or name
        ensure_extension: Optional extension to add if missing (e.g. '.csv')
        
    Returns:
        Resolved Path object
    """
    path = Path(file_path)
    
    # If only a filename was provided (no parent directory or relative path), use DATA_DIR
    if not path.is_absolute() and (not path.parent or str(path.parent) == '.'):
        resolved_path = DATA_DIR / path.name
    else:
        # Use the provided path as is (absolute or relative to current directory)
        resolved_path = path
    
    # Ensure filename has the specified extension
    if ensure_extension and not resolved_path.name.lower().endswith(ensure_extension.lower()):
        resolved_path = Path(f"{resolved_path}{ensure_extension}")
    
    return resolved_path


def filter_data(file_path: str, condition: str, output_file_path: str = None, overwrite: bool = False) -> str:
    """
    Filter data based on a condition and save as a new dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        condition: Python expression for filtering (e.g., "age > 30 and income < 70000")
        output_file_path: Path to save the filtered data (absolute or relative to DATA_DIR, if None, just returns summary)
        overwrite: Whether to overwrite an existing output file
        
    Returns:
        Summary of filtered data
    """
    input_filepath = resolve_file_path(file_path, '.csv')
    
    if not input_filepath.exists():
        return f"Error: File {input_filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(input_filepath)
        
        # Apply filter condition
        try:
            filtered_df = df.query(condition)
        except Exception as e:
            return f"Error applying filter condition: {str(e)}\n\nPlease check your condition syntax."
        
        result = f"# Filtering Results\n\n"
        result += f"**Source File**: {input_filepath}\n"
        result += f"**Filter Condition**: `{condition}`\n\n"
        result += f"- Original rows: {len(df)}\n"
        result += f"- Filtered rows: {len(filtered_df)}\n"
        result += f"- Rows removed: {len(df) - len(filtered_df)}\n"
        result += f"- Percentage kept: {(len(filtered_df) / len(df) * 100):.2f}%\n\n"
        
        # Save filtered data if output_file_path is provided
        if output_file_path:
            output_filepath = resolve_file_path(output_file_path, '.csv')
            
            # Check if file exists and we're not overwriting
            if output_filepath.exists() and not overwrite:
                return f"{result}\nError: Output file {output_filepath} already exists. Set overwrite=True to replace it."
            
            # Create parent directories if they don't exist
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            filtered_df.to_csv(output_filepath, index=False)
            result += f"Filtered data saved to: {output_filepath}\n"
        else:
            # If no output file, include a preview of filtered data
            if len(filtered_df) > 0:
                result += "## Preview of Filtered Data\n"
                result += f"```\n{filtered_df.head(5).to_string()}\n```\n"
                
                if len(filtered_df) > 5:
                    result += f"... and {len(filtered_df) - 5} more rows\n"
            else:
                result += "No rows match the filter condition.\n"
        
        return result
    
    except Exception as e:
        return f"Error filtering data: {str(e)}"


def transform_column(file_path: str, column: str, transformation: str, 
                     new_column: str = None, output_file_path: str = None, 
                     overwrite: bool = False) -> str:
    """
    Apply a transformation to a column in the dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column: Column to transform
        transformation: Type of transformation ('log', 'sqrt', 'square', 'standardize', 'normalize', 'bin', 'abs')
        new_column: Name for the transformed column (if None, overwrites original)
        output_file_path: Path to save the transformed data (absolute or relative to DATA_DIR, if None, returns summary without saving)
        overwrite: Whether to overwrite existing columns/files
        
    Returns:
        Summary of the transformation results
    """
    input_filepath = resolve_file_path(file_path, '.csv')
    
    if not input_filepath.exists():
        return f"Error: File {input_filepath} not found."
    
    # Validate transformation type
    valid_transformations = ['log', 'sqrt', 'square', 'standardize', 'normalize', 'bin', 'abs']
    if transformation not in valid_transformations:
        return f"Error: Invalid transformation type. Choose from: {', '.join(valid_transformations)}"
    
    try:
        # Load the data
        df = pd.read_csv(input_filepath)
        
        # Check if column exists
        if column not in df.columns:
            return f"Error: Column '{column}' not found in {input_filepath}."
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Error: Column '{column}' is not numeric."
        
        # Check if new column already exists
        if new_column and new_column in df.columns and not overwrite:
            return f"Error: Column '{new_column}' already exists. Set overwrite=True to replace it."
        
        # Make a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Apply transformation
        col_data = df[column]
        
        if transformation == 'log':
            # Check for negative or zero values
            if (col_data <= 0).any():
                return f"Error: Log transformation cannot be applied to column '{column}' because it contains zero or negative values."
            transformed = np.log(col_data)
            transform_desc = "Natural logarithm"
            
        elif transformation == 'sqrt':
            # Check for negative values
            if (col_data < 0).any():
                return f"Error: Square root transformation cannot be applied to column '{column}' because it contains negative values."
            transformed = np.sqrt(col_data)
            transform_desc = "Square root"
            
        elif transformation == 'square':
            transformed = np.square(col_data)
            transform_desc = "Square"
            
        elif transformation == 'standardize':
            transformed = (col_data - col_data.mean()) / col_data.std()
            transform_desc = "Standardization (z-score)"
            
        elif transformation == 'normalize':
            min_val = col_data.min()
            max_val = col_data.max()
            if min_val == max_val:
                return f"Error: Cannot normalize column '{column}' because all values are the same."
            transformed = (col_data - min_val) / (max_val - min_val)
            transform_desc = "Min-max normalization (0-1 scale)"
            
        elif transformation == 'bin':
            # Create 10 bins by default
            bins = 10
            transformed = pd.cut(col_data, bins=bins, labels=list(range(bins)))
            transform_desc = f"Binning into {bins} equal-width bins"
            
        elif transformation == 'abs':
            transformed = np.abs(col_data)
            transform_desc = "Absolute value"
        
        # Add transformed column to dataframe
        if new_column:
            result_df[new_column] = transformed
            target_column = new_column
        else:
            result_df[column] = transformed
            target_column = column
        
        # Prepare result summary
        result = f"# Column Transformation Results\n\n"
        result += f"**Source File**: {input_filepath}\n"
        result += f"**Column**: {column}\n"
        result += f"**Transformation**: {transform_desc}\n"
        if new_column:
            result += f"**New Column**: {new_column}\n"
        result += "\n"
        
        # Add basic statistics
        result += "## Before Transformation\n"
        result += f"```\n{df[column].describe().to_string()}\n```\n\n"
        
        result += "## After Transformation\n"
        result += f"```\n{result_df[target_column].describe().to_string()}\n```\n\n"
        
        # Save transformed data if output_file_path is provided
        if output_file_path:
            output_filepath = resolve_file_path(output_file_path, '.csv')
            
            # Check if file exists and we're not overwriting
            if output_filepath.exists() and not overwrite:
                return f"{result}\nError: Output file {output_filepath} already exists. Set overwrite=True to replace it."
            
            # Create parent directories if they don't exist
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            result_df.to_csv(output_filepath, index=False)
            result += f"Transformed data saved to: {output_filepath}\n"
        
        return result
    
    except Exception as e:
        return f"Error transforming column: {str(e)}"


def group_and_aggregate(file_path: str, group_by: Union[str, List[str]], 
                        aggregate_cols: List[str], aggregate_funcs: List[str],
                        output_file_path: str = None, overwrite: bool = False) -> str:
    """
    Group data by column(s) and calculate aggregate statistics.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        group_by: Column(s) to group by (string or list of strings)
        aggregate_cols: Columns to aggregate
        aggregate_funcs: Aggregate functions to apply (e.g., ['mean', 'sum', 'count'])
        output_file_path: Path to save the aggregated data (absolute or relative to DATA_DIR, if None, returns summary without saving)
        overwrite: Whether to overwrite an existing output file
        
    Returns:
        Summary of the aggregation results
    """
    input_filepath = resolve_file_path(file_path, '.csv')
    
    if not input_filepath.exists():
        return f"Error: File {input_filepath} not found."
    
    # Validate aggregate functions
    valid_aggs = ['count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var']
    invalid_aggs = [func for func in aggregate_funcs if func not in valid_aggs]
    if invalid_aggs:
        return f"Error: Invalid aggregation function(s): {', '.join(invalid_aggs)}. Choose from: {', '.join(valid_aggs)}"
    
    try:
        # Load the data
        df = pd.read_csv(input_filepath)
        
        # Convert group_by to list if it's a string
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Check if all columns exist
        missing_group_cols = [col for col in group_by if col not in df.columns]
        if missing_group_cols:
            return f"Error: Grouping columns not found: {', '.join(missing_group_cols)}"
        
        missing_agg_cols = [col for col in aggregate_cols if col not in df.columns]
        if missing_agg_cols:
            return f"Error: Aggregation columns not found: {', '.join(missing_agg_cols)}"
        
        # Check if aggregation columns are numeric
        non_numeric = [col for col in aggregate_cols if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            return f"Error: Non-numeric aggregation columns: {', '.join(non_numeric)}"
        
        # Create aggregation dictionary
        agg_dict = {col: aggregate_funcs for col in aggregate_cols}
        
        # Group and aggregate
        grouped_df = df.groupby(group_by).agg(agg_dict)
        
        # Prepare result summary
        result = f"# Aggregation Results\n\n"
        result += f"**Source File**: {input_filepath}\n"
        result += f"**Grouping Columns**: {', '.join(group_by)}\n"
        result += f"**Aggregation Columns**: {', '.join(aggregate_cols)}\n"
        result += f"**Aggregation Functions**: {', '.join(aggregate_funcs)}\n\n"
        
        result += f"Generated {len(grouped_df)} groups.\n\n"
        
        # Display sample of results
        if len(grouped_df) > 0:
            result += "## Sample of Aggregation Results\n"
            preview_rows = min(5, len(grouped_df))
            result += f"```\n{grouped_df.head(preview_rows).to_string()}\n```\n"
            
            if len(grouped_df) > preview_rows:
                result += f"... and {len(grouped_df) - preview_rows} more groups\n\n"
        
        # Save aggregated data if output_file_path is provided
        if output_file_path:
            output_filepath = resolve_file_path(output_file_path, '.csv')
            
            # Check if file exists and we're not overwriting
            if output_filepath.exists() and not overwrite:
                return f"{result}\nError: Output file {output_filepath} already exists. Set overwrite=True to replace it."
            
            # Create parent directories if they don't exist
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Reset index to convert multi-level columns to regular columns
            grouped_df = grouped_df.reset_index()
            
            # Save to CSV
            grouped_df.to_csv(output_filepath, index=False)
            result += f"Aggregated data saved to: {output_filepath}\n"
        
        return result
    
    except Exception as e:
        return f"Error performing aggregation: {str(e)}"


def handle_missing_values(file_path: str, columns: List[str] = None, method: str = "mean", 
                          output_file_path: str = None, overwrite: bool = False) -> str:
    """
    Handle missing values in a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        columns: List of column names to process (if None, processes all columns with missing values)
        method: Imputation method - 'mean', 'median', 'mode', 'constant', 'drop_rows', 'ffill', 'bfill'
        output_file_path: Path to save the processed data (absolute or relative to DATA_DIR, if None, returns summary without saving)
        overwrite: Whether to overwrite an existing output file
        
    Returns:
        Summary of the missing value handling
    """
    input_filepath = resolve_file_path(file_path, '.csv')
    
    if not input_filepath.exists():
        return f"Error: File {input_filepath} not found."
    
    # Validate method
    valid_methods = ['mean', 'median', 'mode', 'constant', 'drop_rows', 'ffill', 'bfill']
    if method not in valid_methods:
        return f"Error: Invalid imputation method. Choose from: {', '.join(valid_methods)}"
    
    try:
        # Load the data
        df = pd.read_csv(input_filepath)
        
        # If no columns specified, use all columns with missing values
        if not columns:
            missing_counts = df.isnull().sum()
            columns = missing_counts[missing_counts > 0].index.tolist()
            
            if not columns:
                return f"No missing values found in {input_filepath}."
        else:
            # Check if specified columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return f"Error: Columns not found: {', '.join(missing_cols)}"
            
            # Filter to columns that actually have missing values
            has_missing = [col for col in columns if df[col].isnull().any()]
            if not has_missing:
                return f"No missing values found in the specified columns."
            columns = has_missing
        
        # Make a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Prepare result summary
        result = f"# Missing Value Handling Results\n\n"
        result += f"**Source File**: {input_filepath}\n"
        result += f"**Method**: {method}\n\n"
        
        result += "## Columns Processed\n\n"
        result += "| Column | Missing Values | Missing Percentage |\n"
        result += "|--------|---------------|--------------------|\n"
        
        # Track filled values for reporting
        filled_values = {}
        
        # Process each column
        if method == 'drop_rows':
            # Keep track of original row count
            original_rows = len(result_df)
            
            # Drop rows with missing values in specified columns
            result_df = result_df.dropna(subset=columns)
            
            # Report on dropped rows
            dropped_rows = original_rows - len(result_df)
            result += f"\n{dropped_rows} rows dropped ({dropped_rows/original_rows*100:.2f}% of dataset)\n"
            
            for col in columns:
                missing_count = df[col].isnull().sum()
                missing_pct = missing_count / len(df) * 100
                result += f"| {col} | {missing_count} | {missing_pct:.2f}% |\n"
        else:
            # Process each column individually
            for col in columns:
                missing_count = df[col].isnull().sum()
                missing_pct = missing_count / len(df) * 100
                result += f"| {col} | {missing_count} | {missing_pct:.2f}% |\n"
                
                # Skip columns with no missing values
                if missing_count == 0:
                    continue
                
                # Handle numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    if method == 'mean':
                        fill_value = df[col].mean()
                        result_df[col].fillna(fill_value, inplace=True)
                        filled_values[col] = f"mean: {fill_value:.4g}"
                        
                    elif method == 'median':
                        fill_value = df[col].median()
                        result_df[col].fillna(fill_value, inplace=True)
                        filled_values[col] = f"median: {fill_value:.4g}"
                        
                    elif method == 'mode':
                        fill_value = df[col].mode()[0]
                        result_df[col].fillna(fill_value, inplace=True)
                        filled_values[col] = f"mode: {fill_value:.4g}"
                        
                    elif method == 'constant':
                        # For numeric columns, use 0 as the constant
                        result_df[col].fillna(0, inplace=True)
                        filled_values[col] = "constant: 0"
                        
                    elif method == 'ffill':
                        result_df[col].fillna(method='ffill', inplace=True)
                        # Check if any missing values remain (e.g., at the beginning)
                        if result_df[col].isnull().any():
                            result_df[col].fillna(method='bfill', inplace=True)
                        filled_values[col] = "forward fill with backfill for any remaining"
                        
                    elif method == 'bfill':
                        result_df[col].fillna(method='bfill', inplace=True)
                        # Check if any missing values remain (e.g., at the end)
                        if result_df[col].isnull().any():
                            result_df[col].fillna(method='ffill', inplace=True)
                        filled_values[col] = "backward fill with forward fill for any remaining"
                
                # Handle non-numeric columns
                else:
                    if method in ['mean', 'median']:
                        # For non-numeric, use mode instead of mean/median
                        fill_value = df[col].mode()[0]
                        result_df[col].fillna(fill_value, inplace=True)
                        filled_values[col] = f"mode: '{fill_value}' (non-numeric column)"
                        
                    elif method == 'mode':
                        fill_value = df[col].mode()[0]
                        result_df[col].fillna(fill_value, inplace=True)
                        filled_values[col] = f"mode: '{fill_value}'"
                        
                    elif method == 'constant':
                        # For non-numeric columns, use 'MISSING' as the constant
                        result_df[col].fillna('MISSING', inplace=True)
                        filled_values[col] = "constant: 'MISSING'"
                        
                    elif method == 'ffill':
                        result_df[col].fillna(method='ffill', inplace=True)
                        # Check if any missing values remain (e.g., at the beginning)
                        if result_df[col].isnull().any():
                            result_df[col].fillna(method='bfill', inplace=True)
                        filled_values[col] = "forward fill with backfill for any remaining"
                        
                    elif method == 'bfill':
                        result_df[col].fillna(method='bfill', inplace=True)
                        # Check if any missing values remain (e.g., at the end)
                        if result_df[col].isnull().any():
                            result_df[col].fillna(method='ffill', inplace=True)
                        filled_values[col] = "backward fill with forward fill for any remaining"
        
        # Add section about imputation values
        if method != 'drop_rows' and filled_values:
            result += "\n## Imputation Values\n\n"
            for col, value in filled_values.items():
                result += f"- **{col}**: {value}\n"
            result += "\n"
        
        # Check for any remaining missing values
        remaining_missing = result_df.isnull().sum()
        if remaining_missing.sum() > 0:
            result += "\n## Remaining Missing Values\n\n"
            for col, count in remaining_missing.items():
                if count > 0:
                    result += f"- **{col}**: {count} missing values\n"
            result += "\n"
        
        # Save processed data if output_file_path is provided
        if output_file_path:
            output_filepath = resolve_file_path(output_file_path, '.csv')
            
            # Check if file exists and we're not overwriting
            if output_filepath.exists() and not overwrite:
                return f"{result}\nError: Output file {output_filepath} already exists. Set overwrite=True to replace it."
            
            # Create parent directories if they don't exist
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            result_df.to_csv(output_filepath, index=False)
            result += f"Processed data saved to: {output_filepath}\n"
        
        return result
    
    except Exception as e:
        return f"Error handling missing values: {str(e)}"