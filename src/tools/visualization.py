"""
Data visualization tools for the Data Science MCP.
This module contains tools for creating various visualizations from datasets.
"""

from mcp.server.fastmcp import FastMCP, Context, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
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
    mcp_instance.add_tool(plot_scatter)
    mcp_instance.add_tool(plot_histogram)
    mcp_instance.add_tool(plot_bar)
    mcp_instance.add_tool(plot_line)
    mcp_instance.add_tool(plot_correlation_matrix)
    mcp_instance.add_tool(plot_box)
    mcp_instance.add_tool(get_correlation)


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


def plot_scatter(file_path: str, x_column: str, y_column: str, color_column: str = None, title: str = None) -> Image:
    """
    Create a scatter plot of two columns from a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        x_column: Column to plot on x-axis
        y_column: Column to plot on y-axis
        color_column: Column to use for point colors (optional)
        title: Custom title for the plot (optional)
        
    Returns:
        A scatter plot image
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if x_column not in df.columns:
            return f"Error: Column '{x_column}' not found in {filepath}."
        if y_column not in df.columns:
            return f"Error: Column '{y_column}' not found in {filepath}."
        if color_column and color_column not in df.columns:
            return f"Error: Color column '{color_column}' not found in {filepath}."
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        if color_column:
            scatter = plt.scatter(df[x_column], df[y_column], c=df[color_column], 
                                 cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label=color_column)
        else:
            plt.scatter(df[x_column], df[y_column], alpha=0.7)
        
        # Set labels and title
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title if title else f"Scatter Plot: {x_column} vs {y_column}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Create and return image
        img_data = img_buf.getvalue()
        return Image(data=img_data, format="png")
    
    except Exception as e:
        plt.close()
        return f"Error creating scatter plot: {str(e)}"


def plot_histogram(file_path: str, column: str, bins: int = 20, kde: bool = False, title: str = None) -> Image:
    """
    Create a histogram of a column from a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column: Column to plot histogram for
        bins: Number of bins (default: 20)
        kde: Whether to overlay a kernel density estimate (default: False)
        title: Custom title for the plot (optional)
        
    Returns:
        A histogram image
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if column exists
        if column not in df.columns:
            return f"Error: Column '{column}' not found in {filepath}."
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Error: Column '{column}' is not numeric."
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        if kde:
            sns.histplot(df[column].dropna(), bins=bins, kde=True)
        else:
            plt.hist(df[column].dropna(), bins=bins, alpha=0.7)
        
        # Set labels and title
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(title if title else f"Distribution of {column}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Create and return image
        img_data = img_buf.getvalue()
        return Image(data=img_data, format="png")
    
    except Exception as e:
        plt.close()
        return f"Error creating histogram: {str(e)}"


def plot_bar(file_path: str, x_column: str, y_column: str = None, 
             aggfunc: str = "count", horizontal: bool = False, title: str = None) -> Image:
    """
    Create a bar chart from a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        x_column: Column to use for bar categories
        y_column: Column to use for bar heights (optional, if None will count occurrences of x_column values)
        aggfunc: Aggregation function to use ('count', 'sum', 'mean', 'median', 'min', 'max')
        horizontal: Whether to create a horizontal bar chart (default: False)
        title: Custom title for the plot (optional)
        
    Returns:
        A bar chart image
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if x_column not in df.columns:
            return f"Error: Column '{x_column}' not found in {filepath}."
        if y_column and y_column not in df.columns:
            return f"Error: Column '{y_column}' not found in {filepath}."
        
        # Validate aggfunc
        valid_aggs = ['count', 'sum', 'mean', 'median', 'min', 'max']
        if aggfunc not in valid_aggs:
            return f"Error: Invalid aggregation function. Choose from: {', '.join(valid_aggs)}"
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        if y_column:
            # Aggregate data
            if aggfunc == 'count':
                data = df.groupby(x_column).size()
            else:
                agg_dict = {
                    'sum': np.sum,
                    'mean': np.mean,
                    'median': np.median,
                    'min': np.min,
                    'max': np.max
                }
                data = df.groupby(x_column)[y_column].agg(agg_dict[aggfunc])
        else:
            # Just count occurrences
            data = df[x_column].value_counts()
        
        # Sort data for better visualization
        data = data.sort_values(ascending=horizontal)
        
        # Create horizontal or vertical bar chart
        if horizontal:
            data.plot(kind='barh')
        else:
            data.plot(kind='bar')
        
        # Set labels and title
        if y_column:
            plt.ylabel(f"{aggfunc.capitalize()} of {y_column}")
            plt_title = title if title else f"{aggfunc.capitalize()} of {y_column} by {x_column}"
        else:
            plt.ylabel("Count")
            plt_title = title if title else f"Count of {x_column}"
        
        plt.title(plt_title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Create and return image
        img_data = img_buf.getvalue()
        return Image(data=img_data, format="png")
    
    except Exception as e:
        plt.close()
        return f"Error creating bar chart: {str(e)}"


def plot_line(file_path: str, x_column: str, y_columns: List[str], title: str = None) -> Image:
    """
    Create a line plot from a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        x_column: Column to use for x-axis
        y_columns: List of columns to plot as lines
        title: Custom title for the plot (optional)
        
    Returns:
        A line plot image
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if x_column not in df.columns:
            return f"Error: Column '{x_column}' not found in {filepath}."
        
        missing_cols = [col for col in y_columns if col not in df.columns]
        if missing_cols:
            return f"Error: Columns not found: {', '.join(missing_cols)}"
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        for col in y_columns:
            plt.plot(df[x_column], df[col], marker='o', linestyle='-', alpha=0.7, label=col)
        
        # Set labels and title
        plt.xlabel(x_column)
        plt.ylabel("Value")
        plt.title(title if title else f"Line Plot: {', '.join(y_columns)} vs {x_column}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Create and return image
        img_data = img_buf.getvalue()
        return Image(data=img_data, format="png")
    
    except Exception as e:
        plt.close()
        return f"Error creating line plot: {str(e)}"


def plot_correlation_matrix(file_path: str, columns: List[str] = None, title: str = None) -> Image:
    """
    Create a correlation matrix heatmap from a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        columns: List of columns to include (if None, uses all numeric columns)
        title: Custom title for the plot (optional)
        
    Returns:
        A correlation matrix heatmap image
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Filter to numeric columns if not specified
        if columns is None:
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.empty:
                return "Error: No numeric columns found in the dataset."
        else:
            # Check if specified columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return f"Error: Columns not found: {', '.join(missing_cols)}"
            
            # Check if specified columns are numeric
            non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return f"Error: Non-numeric columns: {', '.join(non_numeric)}"
            
            numeric_df = df[columns]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                   linewidths=0.5, square=True)
        
        # Set title
        plt.title(title if title else "Correlation Matrix")
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Create and return image
        img_data = img_buf.getvalue()
        return Image(data=img_data, format="png")
    
    except Exception as e:
        plt.close()
        return f"Error creating correlation matrix: {str(e)}"


def plot_box(file_path: str, columns: List[str], by_column: str = None, title: str = None) -> Image:
    """
    Create box plots from a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        columns: List of numeric columns to create box plots for
        by_column: Optional column to group by (creates separate box plots for each group)
        title: Custom title for the plot (optional)
        
    Returns:
        A box plot image
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return f"Error: Columns not found: {', '.join(missing_cols)}"
        
        # Check if by_column exists
        if by_column and by_column not in df.columns:
            return f"Error: By-column '{by_column}' not found in {filepath}."
        
        # Check if columns are numeric
        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            return f"Error: Non-numeric columns: {', '.join(non_numeric)}"
        
        # Create the plot
        if by_column:
            # Create a single plot for each numeric column, grouped by by_column
            if len(columns) > 1:
                fig, axes = plt.subplots(len(columns), 1, figsize=(10, 5*len(columns)))
                for i, col in enumerate(columns):
                    df.boxplot(column=col, by=by_column, ax=axes[i])
                    axes[i].set_title(f"Box Plot: {col} by {by_column}")
                    axes[i].set_ylabel(col)
            else:
                plt.figure(figsize=(10, 6))
                df.boxplot(column=columns[0], by=by_column)
                plt.title(f"Box Plot: {columns[0]} by {by_column}")
                plt.ylabel(columns[0])
        else:
            # Create a single plot with all numeric columns
            plt.figure(figsize=(10, 6))
            df.boxplot(column=columns)
            plt.title(title if title else "Box Plot")
            plt.ylabel("Value")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Create and return image
        img_data = img_buf.getvalue()
        return Image(data=img_data, format="png")
    
    except Exception as e:
        plt.close()
        return f"Error creating box plot: {str(e)}"


def get_correlation(file_path: str, column1: str, column2: str) -> str:
    """
    Calculate the correlation between two numeric columns in a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column1: First column name
        column2: Second column name
        
    Returns:
        Correlation coefficient and interpretation
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if column1 not in df.columns:
            return f"Error: Column '{column1}' not found in {filepath}."
        if column2 not in df.columns:
            return f"Error: Column '{column2}' not found in {filepath}."
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[column1]):
            return f"Error: Column '{column1}' is not numeric."
        if not pd.api.types.is_numeric_dtype(df[column2]):
            return f"Error: Column '{column2}' is not numeric."
        
        # Calculate correlation
        correlation = df[column1].corr(df[column2])
        
        # Interpret correlation
        if abs(correlation) < 0.1:
            interpretation = "negligible or no"
        elif abs(correlation) < 0.3:
            interpretation = "weak"
        elif abs(correlation) < 0.5:
            interpretation = "moderate"
        elif abs(correlation) < 0.7:
            interpretation = "moderately strong"
        elif abs(correlation) < 0.9:
            interpretation = "strong"
        else:
            interpretation = "very strong"
        
        direction = "positive" if correlation > 0 else "negative"
        
        result = f"# Correlation Analysis: {column1} vs {column2}\n\n"
        result += f"**File Path**: {filepath}\n\n"
        result += f"Correlation coefficient: **{correlation:.4f}**\n\n"
        
        if abs(correlation) < 0.1:
            result += f"There is {interpretation} correlation between {column1} and {column2}."
        else:
            result += f"There is a {interpretation} {direction} correlation between {column1} and {column2}.\n\n"
        
        if correlation > 0:
            result += f"This means that as {column1} increases, {column2} tends to increase as well."
        elif correlation < 0:
            result += f"This means that as {column1} increases, {column2} tends to decrease."
        
        # Add explanation of statistical significance
        n = len(df[[column1, column2]].dropna())
        if n > 0:
            result += f"\n\nThis correlation is based on {n} observations."
            
            if n >= 30:
                # Simplified estimation of significance using Fisher transformation
                # For large sample sizes, we can use the Fisher transformation to test significance
                import math
                r = correlation
                z = 0.5 * math.log((1+r)/(1-r))
                std_error = 1/math.sqrt(n-3)
                p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z/std_error)/math.sqrt(2))))
                
                if p_value < 0.001:
                    significance = "highly statistically significant (p < 0.001)"
                elif p_value < 0.01:
                    significance = "statistically significant (p < 0.01)"
                elif p_value < 0.05:
                    significance = "statistically significant (p < 0.05)"
                else:
                    significance = f"not statistically significant (p = {p_value:.3f})"
                
                result += f"\nThe correlation is {significance}."
        
        return result
    
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"