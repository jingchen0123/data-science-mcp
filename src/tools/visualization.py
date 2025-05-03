"""
Visualization tools for the Data Science MCP.
This module contains tools for creating various data visualizations.
"""

from mcp.server.fastmcp import FastMCP, Context, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
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



def plot_scatter(filename: str, x_column: str, y_column: str, color_column: str = None, title: str = None) -> Image:
    """
    Create a scatter plot of two columns from a dataset.
    
    Args:
        filename: Name of the CSV file
        x_column: Column to plot on x-axis
        y_column: Column to plot on y-axis
        color_column: Column to use for point colors (optional)
        title: Custom title for the plot (optional)
        
    Returns:
        A scatter plot image
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise ValueError(f"File {filename} not found.")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset.")
        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in dataset.")
        if color_column and color_column not in df.columns:
            raise ValueError(f"Column '{color_column}' not found in dataset.")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        if color_column:
            # Create colored scatter plot
            if pd.api.types.is_numeric_dtype(df[color_column]):
                scatter = plt.scatter(df[x_column], df[y_column], c=df[color_column], cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label=color_column)
            else:
                # For categorical color column
                categories = df[color_column].unique()
                for category in categories:
                    subset = df[df[color_column] == category]
                    plt.scatter(subset[x_column], subset[y_column], label=category, alpha=0.7)
                plt.legend()
        else:
            plt.scatter(df[x_column], df[y_column], alpha=0.7)
        
        # Add title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f"{y_column} vs {x_column}")
        
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = df[x_column].corr(df[y_column])
        plt.annotate(f"Correlation: {correlation:.4f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Return as Image
        return Image(data=buf.getvalue(), format="png")
    
    except Exception as e:
        plt.close()  # Ensure figure is closed even on error
        raise ValueError(f"Error creating scatter plot: {str(e)}")



def plot_histogram(filename: str, column: str, bins: int = 20, kde: bool = False, title: str = None) -> Image:
    """
    Create a histogram of a column from a dataset.
    
    Args:
        filename: Name of the CSV file
        column: Column to plot histogram for
        bins: Number of bins (default: 20)
        kde: Whether to overlay a kernel density estimate (default: False)
        title: Custom title for the plot (optional)
        
    Returns:
        A histogram image
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise ValueError(f"File {filename} not found.")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check if column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = plt.hist(df[column].dropna(), bins=bins, alpha=0.7, density=kde)
        
        # Add KDE if requested
        if kde:
            from scipy import stats
            kde_x = np.linspace(df[column].min(), df[column].max(), 1000)
            kde_y = stats.gaussian_kde(df[column].dropna())(kde_x)
            plt.plot(kde_x, kde_y, 'r-', linewidth=2)
        
        # Add title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f"Histogram of {column}")
        
        plt.xlabel(column)
        plt.ylabel("Frequency" if not kde else "Density")
        plt.grid(True, alpha=0.3)
        
        # Add mean and median lines
        mean_val = df[column].mean()
        median_val = df[column].median()
        plt.axvline(mean_val, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='g', linestyle='-.', linewidth=1.5, label=f'Median: {median_val:.2f}')
        
        # Add descriptive statistics
        stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {df[column].std():.2f}"
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     verticalalignment='top')
        
        plt.legend()
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Return as Image
        return Image(data=buf.getvalue(), format="png")
    
    except Exception as e:
        plt.close()  # Ensure figure is closed even on error
        raise ValueError(f"Error creating histogram: {str(e)}")



def plot_bar(filename: str, x_column: str, y_column: str = None, aggfunc: str = 'count', 
             horizontal: bool = False, title: str = None) -> Image:
    """
    Create a bar chart from a dataset.
    
    Args:
        filename: Name of the CSV file
        x_column: Column to use for bar categories
        y_column: Column to use for bar heights (optional, if None will count occurrences of x_column values)
        aggfunc: Aggregation function to use ('count', 'sum', 'mean', 'median', 'min', 'max')
        horizontal: Whether to create a horizontal bar chart (default: False)
        title: Custom title for the plot (optional)
        
    Returns:
        A bar chart image
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise ValueError(f"File {filename} not found.")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset.")
        if y_column and y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in dataset.")
        
        # Valid aggregation functions
        valid_aggfuncs = ['count', 'sum', 'mean', 'median', 'min', 'max']
        if aggfunc not in valid_aggfuncs:
            raise ValueError(f"Invalid aggregation function. Choose from: {', '.join(valid_aggfuncs)}")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        if y_column:
            if aggfunc == 'count':
                data = df.groupby(x_column).size()
            elif aggfunc == 'sum':
                data = df.groupby(x_column)[y_column].sum()
            elif aggfunc == 'mean':
                data = df.groupby(x_column)[y_column].mean()
            elif aggfunc == 'median':
                data = df.groupby(x_column)[y_column].median()
            elif aggfunc == 'min':
                data = df.groupby(x_column)[y_column].min()
            elif aggfunc == 'max':
                data = df.groupby(x_column)[y_column].max()
        else:
            # If no y_column, just count occurrences
            data = df[x_column].value_counts()
        
        # Sort data
        data = data.sort_values()
        
        # Plot horizontal or vertical bar chart
        if horizontal:
            data.plot(kind='barh', ax=plt.gca(), color='skyblue')
        else:
            data.plot(kind='bar', ax=plt.gca(), color='skyblue')
        
        # Add title and labels
        if title:
            plt.title(title)
        else:
            if y_column:
                plt.title(f"{aggfunc.capitalize()} of {y_column} by {x_column}")
            else:
                plt.title(f"Count of {x_column}")
        
        # Axis labels
        if horizontal:
            if y_column:
                plt.xlabel(f"{aggfunc.capitalize()} of {y_column}")
            else:
                plt.xlabel("Count")
            plt.ylabel(x_column)
        else:
            plt.xlabel(x_column)
            if y_column:
                plt.ylabel(f"{aggfunc.capitalize()} of {y_column}")
            else:
                plt.ylabel("Count")
        
        # Add data labels on bars
        ax = plt.gca()
        bars = ax.patches if horizontal else ax.containers[0]
        
        for bar in bars:
            if horizontal:
                width = bar.get_width()
                x = bar.get_width() + (max(data) * 0.01)
                y = bar.get_y() + bar.get_height() / 2
                va = 'center'
            else:
                height = bar.get_height()
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height() + (max(data) * 0.01)
                va = 'bottom'
            
            value = width if horizontal else height
            plt.annotate(f"{value:.1f}", (x, y), ha='center', va=va)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Return as Image
        return Image(data=buf.getvalue(), format="png")
    
    except Exception as e:
        plt.close()  # Ensure figure is closed even on error
        raise ValueError(f"Error creating bar chart: {str(e)}")


def plot_line(filename: str, x_column: str, y_columns: List[str], title: str = None) -> Image:
    """
    Create a line plot from a dataset.
    
    Args:
        filename: Name of the CSV file
        x_column: Column to use for x-axis
        y_columns: List of columns to plot as lines
        title: Custom title for the plot (optional)
        
    Returns:
        A line plot image
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise ValueError(f"File {filename} not found.")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset.")
        
        missing_columns = [col for col in y_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in dataset: {', '.join(missing_columns)}")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Try to convert x to datetime if it looks like a date
        try:
            if df[x_column].dtype == 'object':
                df[x_column] = pd.to_datetime(df[x_column])
        except:
            pass
        
        # Sort by x column
        df = df.sort_values(by=x_column)
        
        # Plot each y column
        for column in y_columns:
            plt.plot(df[x_column], df[column], marker='o', linestyle='-', label=column)
        
        # Add title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f"Line Plot of {', '.join(y_columns)} vs {x_column}")
        
        plt.xlabel(x_column)
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis if it's datetime
        if pd.api.types.is_datetime64_dtype(df[x_column]):
            plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Return as Image
        return Image(data=buf.getvalue(), format="png")
    
    except Exception as e:
        plt.close()  # Ensure figure is closed even on error
        raise ValueError(f"Error creating line plot: {str(e)}")



def plot_correlation_matrix(filename: str, columns: List[str] = None, title: str = None) -> Image:
    """
    Create a correlation matrix heatmap from a dataset.
    
    Args:
        filename: Name of the CSV file
        columns: List of columns to include (if None, uses all numeric columns)
        title: Custom title for the plot (optional)
        
    Returns:
        A correlation matrix heatmap image
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise ValueError(f"File {filename} not found.")
    
    try:
        df = pd.read_csv(filepath)
        
        # If columns not specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if specified columns exist and are numeric
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columns not found in dataset: {', '.join(missing_columns)}")
            
            # Filter non-numeric columns
            non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(f"Non-numeric columns cannot be used in correlation matrix: {', '.join(non_numeric)}")
        
        if len(columns) < 2:
            raise ValueError("At least two numeric columns are required for a correlation matrix")
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Create plot
        plt.figure(figsize=(max(8, len(columns) * 0.7), max(6, len(columns) * 0.7)))
        
        # Create heatmap
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, label='Correlation Coefficient')
        
        # Add title
        if title:
            plt.title(title)
        else:
            plt.title("Correlation Matrix")
        
        # Add labels for each cell
        for i in range(len(columns)):
            for j in range(len(columns)):
                text = f"{corr_matrix.iloc[i, j]:.2f}"
                plt.annotate(text, xy=(j, i), ha='center', va='center',
                             color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        # Set ticks and labels
        plt.xticks(range(len(columns)), columns, rotation=45, ha='right')
        plt.yticks(range(len(columns)), columns)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Return as Image
        return Image(data=buf.getvalue(), format="png")
    
    except Exception as e:
        plt.close()  # Ensure figure is closed even on error
        raise ValueError(f"Error creating correlation matrix: {str(e)}")



def plot_box(filename: str, columns: List[str], by_column: str = None, title: str = None) -> Image:
    """
    Create box plots from a dataset.
    
    Args:
        filename: Name of the CSV file
        columns: List of numeric columns to create box plots for
        by_column: Optional column to group by (creates separate box plots for each group)
        title: Custom title for the plot (optional)
        
    Returns:
        A box plot image
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise ValueError(f"File {filename} not found.")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check if columns exist and are numeric
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in dataset: {', '.join(missing_columns)}")
        
        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns cannot be used for box plots: {', '.join(non_numeric)}")
        
        # Check if by_column exists
        if by_column and by_column not in df.columns:
            raise ValueError(f"Column '{by_column}' not found in dataset.")
        
        # Determine plot size and layout
        if by_column:
            # One row per column, with box plots for each category
            n_groups = df[by_column].nunique()
            if n_groups > 10:
                raise ValueError(f"Too many groups in '{by_column}' (max 10 supported)")
            
            fig, axes = plt.subplots(len(columns), 1, figsize=(max(10, n_groups * 1.5), len(columns) * 4))
            if len(columns) == 1:
                axes = [axes]  # Make it iterable if only one subplot
            
            for i, column in enumerate(columns):
                # Create notched box plot (notch around median shows confidence interval)
                df.boxplot(column=column, by=by_column, ax=axes[i], notch=True, 
                           patch_artist=True, return_type='dict')
                
                axes[i].set_title(f"Box Plot of {column} by {by_column}")
                axes[i].set_ylabel(column)
                
                # Add individual points as a scatter plot for more detail
                categories = df[by_column].unique()
                for j, category in enumerate(categories):
                    subset = df[df[by_column] == category][column].dropna()
                    # Add jitter to x position
                    x = [j + 1 + (np.random.random() - 0.5) * 0.25 for _ in range(len(subset))]
                    axes[i].scatter(x, subset, alpha=0.5, marker='o', s=20, color='black')
            
            # Remove the "by_column" text at the top of the figure
            plt.suptitle("")
            
        else:
            # No grouping, create one row of box plots
            plt.figure(figsize=(max(8, len(columns) * 2.5), 8))
            
            boxplot = df[columns].boxplot(notch=True, patch_artist=True, return_type='dict')
            
            # Add individual points for each column
            for i, column in enumerate(columns):
                data = df[column].dropna()
                # Add jitter to x position
                x = [i + 1 + (np.random.random() - 0.5) * 0.25 for _ in range(len(data))]
                plt.scatter(x, data, alpha=0.5, marker='o', s=20, color='black')
        
        # Add overall title if provided
        if title:
            if by_column:
                fig.suptitle(title, y=1.02)
            else:
                plt.title(title)
        
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Return as Image
        return Image(data=buf.getvalue(), format="png")
    
    except Exception as e:
        plt.close()  # Ensure figure is closed even on error
        raise ValueError(f"Error creating box plot: {str(e)}")