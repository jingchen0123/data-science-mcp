"""
Data exploration tools for the Data Science MCP.
This module contains tools for exploring and analyzing datasets through descriptive
statistics, quality assessment, and other exploratory techniques.
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
    mcp_instance.add_tool(explore_data)
    mcp_instance.add_tool(describe_dataset)
    mcp_instance.add_tool(get_columns_info)
    mcp_instance.add_tool(detect_outliers)
    mcp_instance.add_tool(check_data_quality)


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


def explore_data(file_path: str, sample_rows: int = 5) -> str:
    """
    Perform preliminary data exploration and quality assessment on a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        sample_rows: Number of sample rows to display (default: 5)
        
    Returns:
        Comprehensive data exploration report
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Prepare the exploration report
        report = f"# Preliminary Data Exploration\n\n"
        report += f"**File Path**: {filepath}\n\n"
        
        # === Basic Dataset Characteristics ===
        report += "## 1. Basic Dataset Characteristics\n\n"
        
        # Shape information
        report += f"### 1.1 Dataset Dimensions\n"
        report += f"- Total rows: {df.shape[0]:,}\n"
        report += f"- Total columns: {df.shape[1]:,}\n\n"
        
        # Data types
        report += f"### 1.2 Column Data Types\n"
        dtype_counts = df.dtypes.value_counts().to_dict()
        report += f"- Data type distribution: {', '.join([f'{count} {dtype}' for dtype, count in dtype_counts.items()])}\n\n"
        
        # Detailed column information
        report += "| Column | Data Type | Non-Null Count | Memory Usage |\n"
        report += "|--------|-----------|----------------|-------------|\n"
        
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        for col in df.columns:
            col_dtype = str(df[col].dtype)
            non_null = df[col].count()
            non_null_pct = (non_null / len(df)) * 100
            mem_usage = memory_usage[df.columns.get_loc(col)]
            mem_pct = (mem_usage / total_memory) * 100
            
            mem_str = f"{mem_usage / 1024:.2f} KB ({mem_pct:.1f}%)"
            report += f"| {col} | {col_dtype} | {non_null:,} ({non_null_pct:.1f}%) | {mem_str} |\n"
        
        report += f"\n- Total memory usage: {total_memory / (1024*1024):.2f} MB\n\n"
        
        # Data sample
        report += f"### 1.3 Data Sample (First {sample_rows} rows)\n"
        report += f"```\n{df.head(sample_rows).to_string()}\n```\n\n"
        
        # Basic statistics
        report += f"### 1.4 Summary Statistics\n"
        
        # For numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            report += f"#### Numeric Columns\n"
            report += f"```\n{df[numeric_cols].describe().to_string()}\n```\n\n"
        
        # For categorical/object columns
        cat_cols = df.select_dtypes(exclude=['number']).columns
        if len(cat_cols) > 0:
            report += f"#### Categorical Columns\n"
            for col in cat_cols:
                value_counts = df[col].value_counts().head(5)
                unique_count = df[col].nunique()
                report += f"- **{col}**: {unique_count:,} unique values\n"
                report += f"  Top values: "
                report += ", ".join([f"'{val}' ({count:,})" for val, count in value_counts.items()])
                report += "\n"
            report += "\n"
        
        # === Data Quality Check ===
        report += "## 2. Data Quality Assessment\n\n"
        
        # Missing values
        report += "### 2.1 Missing Values\n"
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        if missing.sum() > 0:
            report += "| Column | Missing Count | Missing Percentage |\n"
            report += "|--------|--------------|-------------------|\n"
            
            for col, count in missing.items():
                if count > 0:
                    report += f"| {col} | {count:,} | {missing_pct[col]:.2f}% |\n"
            
            # Missing values pattern
            report += "\n#### Missing Value Patterns\n"
            
            # Check if missingness is correlated across columns
            high_missing_cols = missing[missing > 0].index.tolist()
            if len(high_missing_cols) > 1:
                missing_df = df[high_missing_cols].isnull()
                pattern_counts = missing_df.value_counts().head(5)
                
                report += "Top missing patterns:\n"
                for pattern, count in pattern_counts.items():
                    pattern_str = ", ".join([f"{col}: {'Missing' if val else 'Present'}" for col, val in zip(high_missing_cols, pattern)])
                    report += f"- {pattern_str}: {count:,} rows ({count/len(df)*100:.2f}%)\n"
        else:
            report += "No missing values found in the dataset.\n"
        
        report += "\n"
        
        # Duplicate checks
        report += "### 2.2 Duplicate Rows\n"
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            report += f"- Found {duplicates:,} duplicate rows ({dup_pct:.2f}% of the dataset)\n\n"
        else:
            report += "- No duplicate rows found\n\n"
        
        # Outlier detection for numeric columns
        if len(numeric_cols) > 0:
            report += "### 2.3 Potential Outliers in Numeric Columns\n"
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_pct = (outlier_count / df[col].count()) * 100
                    report += f"- **{col}**: {outlier_count:,} potential outliers ({outlier_pct:.2f}%)\n"
                    report += f"  - Range for normal values: [{lower_bound:.2f}, {upper_bound:.2f}]\n"
                    report += f"  - Actual range: [{df[col].min():.2f}, {df[col].max():.2f}]\n"
            
            report += "\n"
        
        # Range and distribution checks
        if len(numeric_cols) > 0:
            report += "### 2.4 Value Ranges and Distributions\n"
            
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                median_val = df[col].median()
                skew_val = df[col].skew()
                
                report += f"- **{col}**:\n"
                report += f"  - Range: [{min_val:.2f}, {max_val:.2f}]\n"
                report += f"  - Mean: {mean_val:.2f}, Median: {median_val:.2f}\n"
                
                # Interpret skewness
                if abs(skew_val) < 0.5:
                    skew_desc = "approximately symmetric"
                elif abs(skew_val) < 1:
                    skew_desc = f"moderately {'positively' if skew_val > 0 else 'negatively'} skewed"
                else:
                    skew_desc = f"highly {'positively' if skew_val > 0 else 'negatively'} skewed"
                
                report += f"  - Skewness: {skew_val:.2f} ({skew_desc})\n"
            
            report += "\n"
        
        # Categorical data consistency
        if len(cat_cols) > 0:
            report += "### 2.5 Categorical Data Consistency\n"
            
            for col in cat_cols:
                unique_count = df[col].nunique()
                # Check for potential case inconsistencies
                if df[col].dtype == 'object':
                    case_insensitive_count = df[col].str.lower().nunique() if not df[col].isnull().all() else 0
                    if case_insensitive_count < unique_count:
                        report += f"- **{col}**: Potential case inconsistency detected. "
                        report += f"{unique_count} unique values vs {case_insensitive_count} case-insensitive unique values\n"
                
                # Check for trailing whitespace
                if df[col].dtype == 'object':
                    if not df[col].isnull().all() and (df[col].str.strip() != df[col]).any():
                        report += f"- **{col}**: Contains values with leading or trailing whitespace\n"
                
                # Check for very high cardinality relative to dataset size
                if unique_count > 0.5 * len(df) and unique_count > 10:
                    report += f"- **{col}**: High cardinality column with {unique_count:,} unique values "
                    report += f"({unique_count/len(df)*100:.1f}% of total rows)\n"
            
            report += "\n"
        
        # === Recommendations ===
        report += "## 3. Initial Insights and Recommendations\n\n"
        
        # Generate basic recommendations based on findings
        recommendations = []
        
        # Missing data recommendations
        if missing.sum() > 0:
            high_missing_cols = missing[missing/len(df) > 0.2].index.tolist()
            if high_missing_cols:
                recommendations.append(f"Consider strategies for handling columns with high missing rates: {', '.join(high_missing_cols)}")
        
        # Outlier recommendations
        if len(numeric_cols) > 0:
            outlier_cols = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)][col])
                if outlier_count > 0.05 * len(df):
                    outlier_cols.append(col)
            
            if outlier_cols:
                recommendations.append(f"Investigate potential outliers in: {', '.join(outlier_cols)}")
        
        # Skewness recommendations
        if len(numeric_cols) > 0:
            skewed_cols = [col for col in numeric_cols if abs(df[col].skew()) > 1]
            if skewed_cols:
                recommendations.append(f"Consider transformations for highly skewed columns: {', '.join(skewed_cols)}")
        
        # Memory usage recommendations
        if total_memory > 100 * 1024 * 1024:  # If more than 100MB
            high_mem_cols = []
            for col in df.columns:
                mem_usage = memory_usage[df.columns.get_loc(col)]
                if mem_usage > 0.1 * total_memory:  # If column uses more than 10% of memory
                    high_mem_cols.append((col, mem_usage))
            
            if high_mem_cols:
                recommendations.append("Consider memory optimization for large columns: " + 
                                      ", ".join([f"{col} ({mem/1024/1024:.1f} MB)" for col, mem in high_mem_cols]))
        
        # Data type recommendations
        int_cols_as_float = [col for col in numeric_cols if 
                             df[col].dtype == 'float64' and 
                             df[col].notnull().all() and 
                             df[col].apply(lambda x: x.is_integer() if not pd.isna(x) else True).all()]
        
        if int_cols_as_float:
            recommendations.append(f"Consider converting float columns with integer values to integer type: {', '.join(int_cols_as_float)}")
        
        # Categorical encoding recommendations
        if len(cat_cols) > 0:
            high_cardinality_cols = [col for col in cat_cols if df[col].nunique() > min(50, 0.2 * len(df))]
            if high_cardinality_cols:
                recommendations.append(f"High cardinality categorical columns may need special encoding approaches: {', '.join(high_cardinality_cols)}")
        
        # Add recommendations to report
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        else:
            report += "- The dataset appears to be well-structured. Consider proceeding with exploratory data analysis to understand relationships between variables.\n"
        
        return report
    
    except Exception as e:
        return f"Error during data exploration: {str(e)}"


def describe_dataset(file_path: str) -> str:
    """
    Generate descriptive statistics for a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        
    Returns:
        Descriptive statistics of the dataset
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        df = pd.read_csv(filepath)
        
        # Create a comprehensive summary
        result = f"# Dataset Summary\n\n"
        result += f"**File Path**: {filepath}\n\n"
        
        # Basic info
        result += f"## Basic Information\n"
        result += f"- Rows: {df.shape[0]}\n"
        result += f"- Columns: {df.shape[1]}\n"
        result += f"- Column Names: {', '.join(df.columns)}\n\n"
        
        # Data types
        result += f"## Data Types\n"
        for col, dtype in df.dtypes.items():
            result += f"- {col}: {dtype}\n"
        result += "\n"
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            result += f"## Missing Values\n"
            for col, count in missing.items():
                if count > 0:
                    result += f"- {col}: {count} ({(count/len(df))*100:.2f}%)\n"
            result += "\n"
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result += f"## Numeric Column Statistics\n"
            result += df[numeric_cols].describe().to_string()
            result += "\n\n"
        
        # Categorical statistics
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            result += f"## Categorical Column Information\n"
            for col in cat_cols:
                result += f"### {col}\n"
                result += f"- Unique Values: {df[col].nunique()}\n"
                result += f"- Top 5 Values: {', '.join(str(x) for x in df[col].value_counts().head(5).index)}\n\n"
        
        return result
    
    except Exception as e:
        return f"Error analyzing dataset: {str(e)}"


def get_columns_info(file_path: str, columns: List[str] = None) -> str:
    """
    Get detailed information about specific columns in a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        columns: List of column names to analyze (if None, analyzes all columns)
        
    Returns:
        Detailed information about the specified columns
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        df = pd.read_csv(filepath)
        
        # If no columns specified, use all columns
        if not columns:
            columns = df.columns.tolist()
        else:
            # Check if all specified columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return f"Error: Columns not found in dataset: {', '.join(missing_cols)}"
        
        result = f"# Column Information\n\n"
        result += f"**File Path**: {filepath}\n\n"
        
        for col in columns:
            result += f"## {col}\n\n"
            
            # Basic info
            result += f"- **Data Type**: {df[col].dtype}\n"
            result += f"- **Non-Null Values**: {df[col].count()} ({df[col].count()/len(df)*100:.2f}%)\n"
            result += f"- **Null Values**: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)\n"
            
            # Type-specific info
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric column
                result += f"- **Min**: {df[col].min()}\n"
                result += f"- **Max**: {df[col].max()}\n"
                result += f"- **Mean**: {df[col].mean()}\n"
                result += f"- **Median**: {df[col].median()}\n"
                result += f"- **Standard Deviation**: {df[col].std()}\n"
                result += f"- **Skewness**: {df[col].skew()}\n"
                result += f"- **Kurtosis**: {df[col].kurtosis()}\n\n"
                
                # Calculate outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_count = len(outliers)
                
                result += f"- **Potential Outliers**: {outlier_count} ({outlier_count/df[col].count()*100:.2f}%)\n"
                result += f"- **Normal Range**: [{lower_bound}, {upper_bound}]\n\n"
                
                # Histogram representation
                hist, bin_edges = np.histogram(df[col].dropna(), bins=10)
                hist_str = "- **Distribution**:\n  ```\n"
                
                max_count = max(hist) if max(hist) > 0 else 1
                for i, count in enumerate(hist):
                    bar_len = int(50 * count / max_count)
                    hist_str += f"  [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {'#' * bar_len} ({count})\n"
                
                hist_str += "  ```\n\n"
                result += hist_str
                
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                # Categorical column
                unique_count = df[col].nunique()
                result += f"- **Unique Values**: {unique_count}\n"
                
                if unique_count <= 20:
                    # Show value counts if not too many unique values
                    value_counts = df[col].value_counts().head(10)
                    result += "- **Value Counts**:\n"
                    
                    max_count = value_counts.max() if not value_counts.empty else 1
                    for val, count in value_counts.items():
                        bar_len = int(30 * count / max_count)
                        result += f"  {val}: {'#' * bar_len} ({count}, {count/len(df)*100:.2f}%)\n"
                else:
                    # Just show top 10 values for high cardinality columns
                    result += f"- **Top 10 Values**:\n"
                    for val, count in df[col].value_counts().head(10).items():
                        result += f"  {val}: {count} ({count/len(df)*100:.2f}%)\n"
                
                # Check for potential case inconsistencies
                if df[col].dtype == 'object' and not df[col].isnull().all():
                    case_insensitive_count = df[col].str.lower().nunique()
                    if case_insensitive_count < unique_count:
                        result += f"\n- **Note**: Potential case inconsistency detected. "
                        result += f"{unique_count} unique values vs {case_insensitive_count} case-insensitive unique values\n"
                
                result += "\n"
            
            elif pd.api.types.is_datetime64_dtype(df[col]):
                # Datetime column
                result += f"- **Min Date**: {df[col].min()}\n"
                result += f"- **Max Date**: {df[col].max()}\n"
                result += f"- **Range**: {df[col].max() - df[col].min()}\n\n"
                
                # Show distribution by year or month
                try:
                    year_counts = df[col].dt.year.value_counts().sort_index()
                    result += "- **Distribution by Year**:\n"
                    for year, count in year_counts.items():
                        result += f"  {year}: {count}\n"
                except:
                    pass
                
                result += "\n"
            
            # Correlation with other numeric columns (if current column is numeric)
            if pd.api.types.is_numeric_dtype(df[col]):
                other_numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != col]
                
                if other_numeric_cols:
                    result += "- **Correlation with Other Numeric Columns**:\n"
                    for other_col in other_numeric_cols:
                        corr = df[col].corr(df[other_col])
                        result += f"  {other_col}: {corr:.4f}\n"
                
                result += "\n"
        
        return result
    
    except Exception as e:
        return f"Error analyzing columns: {str(e)}"


def detect_outliers(file_path: str, columns: List[str] = None, method: str = "iqr") -> str:
    """
    Detect outliers in a dataset using various methods.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        columns: List of column names to analyze (if None, analyzes all numeric columns)
        method: Method to use for outlier detection ('iqr', 'zscore', or 'both')
        
    Returns:
        Detailed report of outliers found
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    # Validate method
    valid_methods = ["iqr", "zscore", "both"]
    if method not in valid_methods:
        return f"Error: Invalid method '{method}'. Valid methods are: {', '.join(valid_methods)}"
    
    try:
        df = pd.read_csv(filepath)
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if all specified columns exist and are numeric
            for col in columns:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found in dataset."
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return f"Error: Column '{col}' is not numeric."
        
        if not columns:
            return "No numeric columns found in the dataset."
        
        result = f"# Outlier Detection Report\n\n"
        result += f"**File Path**: {filepath}\n\n"
        
        # Track total outliers
        total_outliers = {}
        
        # IQR method
        if method in ["iqr", "both"]:
            result += "## Outliers Using IQR Method (Tukey's Method)\n\n"
            
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                result += f"### {col}\n"
                result += f"- **IQR**: {IQR}\n"
                result += f"- **Normal Range**: [{lower_bound}, {upper_bound}]\n"
                result += f"- **Outliers**: {outlier_count} ({outlier_count/len(df)*100:.2f}%)\n"
                
                if outlier_count > 0:
                    result += f"- **Lower Outliers**: {len(df[df[col] < lower_bound])} values < {lower_bound}\n"
                    result += f"- **Upper Outliers**: {len(df[df[col] > upper_bound])} values > {upper_bound}\n"
                    
                    # Show a few example outliers
                    if len(df[df[col] < lower_bound]) > 0:
                        result += f"- **Sample Lower Outliers**: {df[df[col] < lower_bound][col].head(3).tolist()}\n"
                    if len(df[df[col] > upper_bound]) > 0:
                        result += f"- **Sample Upper Outliers**: {df[df[col] > upper_bound][col].head(3).tolist()}\n"
                
                result += "\n"
                
                # Track total outliers
                total_outliers[col] = outliers.index.tolist() if col not in total_outliers else list(set(total_outliers[col] + outliers.index.tolist()))
        
        # Z-score method
        if method in ["zscore", "both"]:
            result += "## Outliers Using Z-Score Method\n\n"
            
            for col in columns:
                mean = df[col].mean()
                std = df[col].std()
                
                # Using abs(z-score) > 3 as outlier threshold
                z_scores = np.abs((df[col] - mean) / std)
                outliers = df[z_scores > 3]
                outlier_count = len(outliers)
                
                result += f"### {col}\n"
                result += f"- **Mean**: {mean}\n"
                result += f"- **Standard Deviation**: {std}\n"
                result += f"- **Outliers** (|z-score| > 3): {outlier_count} ({outlier_count/len(df)*100:.2f}%)\n"
                
                if outlier_count > 0:
                    min_outlier = df[z_scores > 3][col].min()
                    max_outlier = df[z_scores > 3][col].max()
                    
                    result += f"- **Outlier Range**: [{min_outlier}, {max_outlier}]\n"
                    
                    # Show a few example outliers with their z-scores
                    sample_outliers = df[z_scores > 3].head(3)
                    if not sample_outliers.empty:
                        result += "- **Sample Outliers**:\n"
                        for idx, value in sample_outliers[col].items():
                            result += f"  {value} (z-score: {z_scores[idx]:.2f})\n"
                
                result += "\n"
                
                # Track total outliers
                total_outliers[col] = list(set(total_outliers.get(col, []) + outliers.index.tolist()))
        
        # Summary of findings
        result += "## Summary of Findings\n\n"
        
        for col in columns:
            outlier_count = len(total_outliers.get(col, []))
            outlier_percentage = outlier_count / len(df) * 100
            result += f"- **{col}**: {outlier_count} outliers ({outlier_percentage:.2f}%)\n"
        
        result += "\n"
        
        # Recommendations
        result += "## Recommendations\n\n"
        high_outlier_cols = []
        
        for col in columns:
            outlier_count = len(total_outliers.get(col, []))
            outlier_percentage = outlier_count / len(df) * 100
            
            if outlier_percentage > 5:
                high_outlier_cols.append((col, outlier_percentage))
        
        if high_outlier_cols:
            result += "Columns with high outlier percentages:\n"
            for col, pct in sorted(high_outlier_cols, key=lambda x: x[1], reverse=True):
                result += f"- **{col}**: {pct:.2f}% outliers\n"
            
            result += "\nConsider these approaches for handling outliers:\n"
            result += "1. **Investigation**: Examine if these are genuine outliers or data errors\n"
            result += "2. **Capping/Winsorizing**: Cap extreme values at a specified percentile\n"
            result += "3. **Transformation**: Apply log or other transformations to reduce the impact of outliers\n"
            result += "4. **Robust Statistics**: Use median instead of mean, IQR instead of standard deviation\n"
            result += "5. **Separate Models**: Create separate models for outlier and non-outlier data\n"
        else:
            result += "No columns with significant outlier percentages (>5%) were found.\n"
        
        return result
    
    except Exception as e:
        return f"Error detecting outliers: {str(e)}"


def check_data_quality(file_path: str) -> str:
    """
    Perform a comprehensive data quality check on a dataset.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        
    Returns:
        Detailed data quality report with scores
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        df = pd.read_csv(filepath)
        
        result = f"# Data Quality Assessment\n\n"
        result += f"**File Path**: {filepath}\n\n"
        
        # Define quality dimensions and initialize scores
        quality_scores = {
            "Completeness": 0,
            "Accuracy": 0,
            "Consistency": 0,
            "Validity": 0,
            "Uniqueness": 0
        }
        
        # Track issue details
        issues = {
            "Completeness": [],
            "Accuracy": [],
            "Consistency": [],
            "Validity": [],
            "Uniqueness": []
        }
        
        # 1. Completeness: Check for missing values
        result += "## 1. Completeness\n\n"
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        completeness_score = 100 - (missing.sum() / (len(df) * len(df.columns)) * 100)
        quality_scores["Completeness"] = completeness_score
        
        result += f"- **Overall Completeness Score**: {completeness_score:.2f}%\n\n"
        
        if missing.sum() > 0:
            result += "| Column | Missing Count | Missing Percentage |\n"
            result += "|--------|--------------|-------------------|\n"
            
            for col, count in missing.items():
                if count > 0:
                    result += f"| {col} | {count:,} | {missing_pct[col]:.2f}% |\n"
                    if missing_pct[col] > 5:
                        issues["Completeness"].append(f"{col} ({missing_pct[col]:.2f}% missing)")
        else:
            result += "No missing values found in the dataset.\n"
        
        result += "\n"
        
        # 2. Accuracy: Check for outliers in numeric columns
        result += "## 2. Accuracy\n\n"
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outlier_columns = 0
            total_outlier_percentage = 0
            
            for col in numeric_cols:
                # Use IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / df[col].count()) * 100
                
                if outlier_count > 0:
                    result += f"- **{col}**: {outlier_count:,} potential outliers ({outlier_pct:.2f}%)\n"
                    
                    if outlier_pct > 5:
                        outlier_columns += 1
                        total_outlier_percentage += outlier_pct
                        issues["Accuracy"].append(f"{col} ({outlier_pct:.2f}% outliers)")
            
            # Calculate accuracy score based on outliers
            if len(numeric_cols) > 0:
                avg_outlier_pct = total_outlier_percentage / max(1, outlier_columns)
                accuracy_score = 100 - min(avg_outlier_pct, 100)
                quality_scores["Accuracy"] = accuracy_score
                
                result += f"\n- **Overall Accuracy Score**: {accuracy_score:.2f}%\n"
                result += f"  (Based on average percentage of outliers in problematic columns)\n\n"
            else:
                quality_scores["Accuracy"] = 100
                result += "\n- **Overall Accuracy Score**: 100.00%\n\n"
        else:
            quality_scores["Accuracy"] = "N/A"
            result += "No numeric columns to assess accuracy through outlier detection.\n\n"
        
        # 3. Consistency: Check for data consistency issues
        result += "## 3. Consistency\n\n"
        
        consistency_issues = 0
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Check for case inconsistencies in strings
        if len(cat_cols) > 0:
            for col in cat_cols:
                if df[col].dtype == 'object':
                    # Check for case inconsistencies
                    if not df[col].isnull().all():
                        unique_count = df[col].nunique()
                        case_insensitive_count = df[col].str.lower().nunique()
                        
                        if case_insensitive_count < unique_count:
                            diff = unique_count - case_insensitive_count
                            diff_pct = (diff / unique_count) * 100
                            consistency_issues += 1
                            
                            result += f"- **{col}**: Case inconsistency detected - "
                            result += f"{unique_count} unique values vs {case_insensitive_count} case-insensitive unique values"
                            result += f" ({diff_pct:.2f}% potential inconsistency)\n"
                            
                            issues["Consistency"].append(f"{col} (case inconsistency)")
                    
                    # Check for leading/trailing whitespace
                    if not df[col].isnull().all():
                        ws_issues = (df[col].astype(str).str.strip() != df[col].astype(str)).sum()
                        if ws_issues > 0:
                            ws_pct = (ws_issues / len(df)) * 100
                            consistency_issues += 1
                            
                            result += f"- **{col}**: {ws_issues} values ({ws_pct:.2f}%) have leading/trailing whitespace\n"
                            issues["Consistency"].append(f"{col} (whitespace issues)")
        
        # Calculate consistency score
        total_possible_issues = max(1, len(cat_cols) * 2)  # 2 checks per column
        consistency_score = 100 - (consistency_issues / total_possible_issues * 100)
        quality_scores["Consistency"] = consistency_score
        
        result += f"\n- **Overall Consistency Score**: {consistency_score:.2f}%\n\n"
        
        # 4. Validity: Check if data falls within expected ranges/formats
        result += "## 4. Validity\n\n"
        
        validity_issues = 0
        total_checks = 0
        
        # For numeric columns, check if values are within reasonable bounds
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                total_checks += 1
                
                # Check for negative values in typically positive columns
                if col.lower() in ['age', 'quantity', 'price', 'amount', 'count', 'height', 'weight']:
                    neg_count = (df[col] < 0).sum()
                    if neg_count > 0:
                        neg_pct = (neg_count / len(df)) * 100
                        validity_issues += 1
                        
                        result += f"- **{col}**: {neg_count} negative values ({neg_pct:.2f}%) "
                        result += f"which may be invalid for this type of data\n"
                        
                        issues["Validity"].append(f"{col} (negative values)")
                
                # Check for unreasonably large values (heuristic: > 5 times the 95th percentile)
                p95 = df[col].quantile(0.95)
                extreme_count = (df[col] > (5 * p95)).sum() if p95 > 0 else 0
                
                if extreme_count > 0:
                    extreme_pct = (extreme_count / len(df)) * 100
                    validity_issues += 0.5  # Partial penalty (might be valid)
                    
                    result += f"- **{col}**: {extreme_count} extremely large values ({extreme_pct:.2f}%) "
                    result += f"that are > 5x the 95th percentile ({p95:.2f})\n"
                    
                    issues["Validity"].append(f"{col} (extreme values)")
        
        # For categorical/string columns, check for unexpected formats
        for col in cat_cols:
            if df[col].dtype == 'object':
                total_checks += 1
                
                # Check for mixed data types in the same column
                try:
                    # Check if some values look like numbers and others don't
                    numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if 0 < numeric_count < len(df[col]) - df[col].isna().sum():
                        mixed_pct = (numeric_count / len(df)) * 100
                        validity_issues += 1
                        
                        result += f"- **{col}**: Mixed data types detected. "
                        result += f"{numeric_count} values ({mixed_pct:.2f}%) appear to be numeric\n"
                        
                        issues["Validity"].append(f"{col} (mixed types)")
                except:
                    pass
        
        # Calculate validity score
        validity_score = 100 - (validity_issues / max(1, total_checks) * 100)
        quality_scores["Validity"] = validity_score
        
        result += f"\n- **Overall Validity Score**: {validity_score:.2f}%\n\n"
        
        # 5. Uniqueness: Check for duplicates
        result += "## 5. Uniqueness\n\n"
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / len(df)) * 100
        
        result += f"- **Duplicate Rows**: {duplicate_rows} ({duplicate_pct:.2f}% of the dataset)\n"
        
        if duplicate_rows > 0:
            issues["Uniqueness"].append(f"Duplicate rows ({duplicate_pct:.2f}%)")
        
        # Check for potential ID columns with duplicates
        id_columns = [col for col in df.columns if 'id' in col.lower() or 'key' in col.lower() or 'code' in col.lower()]
        
        if id_columns:
            result += "\n- **Potential ID Columns**:\n"
            
            for col in id_columns:
                unique_vals = df[col].nunique()
                duplicate_count = len(df) - unique_vals - df[col].isna().sum()
                duplicate_id_pct = (duplicate_count / len(df)) * 100
                
                result += f"  - **{col}**: {unique_vals} unique values "
                
                if duplicate_count > 0:
                    result += f"({duplicate_id_pct:.2f}% duplicates)\n"
                    issues["Uniqueness"].append(f"{col} ({duplicate_id_pct:.2f}% duplicates)")
                else:
                    result += "(no duplicates)\n"
        
        # Calculate uniqueness score
        uniqueness_score = 100 - duplicate_pct - sum([duplicate_id_pct for col in id_columns if 
                                                     (duplicate_id_pct := (len(df) - df[col].nunique() - df[col].isna().sum()) / len(df) * 100) > 0])
        uniqueness_score = max(0, uniqueness_score)  # Ensure score isn't negative
        quality_scores["Uniqueness"] = uniqueness_score
        
        result += f"\n- **Overall Uniqueness Score**: {uniqueness_score:.2f}%\n\n"
        
        # Overall Data Quality Score
        result += "## Overall Data Quality\n\n"
        
        # Calculate average score (ignoring N/A values)
        valid_scores = [score for score in quality_scores.values() if isinstance(score, (int, float))]
        overall_score = sum(valid_scores) / len(valid_scores)
        
        result += "| Dimension | Score | Issues |\n"
        result += "|-----------|-------|--------|\n"
        
        for dimension, score in quality_scores.items():
            score_str = f"{score:.2f}%" if isinstance(score, (int, float)) else score
            issue_str = "; ".join(issues[dimension]) if issues[dimension] else "None"
            result += f"| {dimension} | {score_str} | {issue_str} |\n"
        
        result += f"\n**Overall Data Quality Score**: {overall_score:.2f}%\n\n"
        
        # Quality grade
        grade = "A" if overall_score >= 90 else "B" if overall_score >= 80 else "C" if overall_score >= 70 else "D" if overall_score >= 60 else "F"
        result += f"**Data Quality Grade**: {grade}\n\n"
        
        # Recommendations
        result += "## Recommendations\n\n"
        
        if issues["Completeness"]:
            result += "### Completeness Issues\n"
            result += "- Consider handling missing values in: " + ", ".join(issues["Completeness"]) + "\n"
            result += "- Options: imputation, dropping rows with many missing values, or using algorithms that handle missing data\n\n"
        
        if issues["Accuracy"]:
            result += "### Accuracy Issues\n"
            result += "- Address potential outliers in: " + ", ".join(issues["Accuracy"]) + "\n"
            result += "- Options: investigate, cap/transform, or use robust statistical methods\n\n"
        
        if issues["Consistency"]:
            result += "### Consistency Issues\n"
            result += "- Fix consistency issues in: " + ", ".join(issues["Consistency"]) + "\n"
            result += "- Options: standardize case, trim whitespace, correct inconsistent spellings\n\n"
        
        if issues["Validity"]:
            result += "### Validity Issues\n"
            result += "- Address validity concerns in: " + ", ".join(issues["Validity"]) + "\n"
            result += "- Options: apply domain-specific validation rules, correct invalid values\n\n"
        
        if issues["Uniqueness"]:
            result += "### Uniqueness Issues\n"
            result += "- Handle duplication issues: " + ", ".join(issues["Uniqueness"]) + "\n"
            result += "- Options: remove duplicates, investigate source of duplication, establish proper key constraints\n\n"
        
        return result
    
    except Exception as e:
        return f"Error checking data quality: {str(e)}"