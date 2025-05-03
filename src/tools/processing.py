"""
Data processing tools for the Data Science MCP.
This module contains tools for data transformation, cleaning, and feature engineering.
"""

from pathlib import Path
import os
from typing import Optional, Dict, List, Any, Union, Tuple

import pandas as pd
import numpy as np
from scipy import stats

# The DATA_DIR will be initialized by the main server module
# This is a placeholder that will be overwritten
DATA_DIR = None
# The MCP instance will be initialized by the main server module
mcp = None

def initialize(mcp_instance, data_dir: Path):
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
    mcp_instance.add_tool(get_correlation)
    mcp_instance.add_tool(filter_data)
    mcp_instance.add_tool(transform_column)
    mcp_instance.add_tool(group_and_aggregate)
    mcp_instance.add_tool(handle_missing_values)
    #mcp_instance.add_tool(create_feature)
    #mcp_instance.add_tool(encode_categorical)



def get_correlation(filename: str, column1: str, column2: str) -> str:
    """
    Calculate the correlation between two numeric columns in a dataset.
    
    Args:
        filename: Name of the CSV file
        column1: First column name
        column2: Second column name
        
    Returns:
        Correlation coefficient and interpretation
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return f"Error: File {filename} not found."
    
    try:
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if column1 not in df.columns:
            return f"Error: Column '{column1}' not found in dataset."
        if column2 not in df.columns:
            return f"Error: Column '{column2}' not found in dataset."
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[column1]):
            return f"Error: Column '{column1}' is not numeric."
        if not pd.api.types.is_numeric_dtype(df[column2]):
            return f"Error: Column '{column2}' is not numeric."
        
        # Calculate correlations using different methods
        pearson_corr = df[column1].corr(df[column2], method='pearson')
        spearman_corr = df[column1].corr(df[column2], method='spearman')
        
        # Calculate p-value for Pearson correlation
        pearson_r, p_value = stats.pearsonr(df[column1].dropna(), df[column2].dropna())
        
        # Interpret correlation
        def interpret_correlation(corr):
            interpretation = ""
            if abs(corr) < 0.3:
                interpretation = "weak"
            elif abs(corr) < 0.7:
                interpretation = "moderate"
            else:
                interpretation = "strong"
            
            if corr > 0:
                interpretation += " positive"
            else:
                interpretation += " negative"
            
            return interpretation
        
        result = f"# Correlation Analysis: {column1} vs {column2}\n\n"
        
        result += f"## Pearson Correlation (Linear)\n"
        result += f"- **Correlation Coefficient**: {pearson_corr:.4f}\n"
        result += f"- **Interpretation**: This indicates a {interpret_correlation(pearson_corr)} linear relationship.\n"
        result += f"- **Statistical Significance**: p-value = {p_value:.4f} "
        
        if p_value < 0.05:
            result += "(statistically significant at α = 0.05)\n\n"
        else:
            result += "(not statistically significant at α = 0.05)\n\n"
        
        result += f"## Spearman Correlation (Monotonic)\n"
        result += f"- **Correlation Coefficient**: {spearman_corr:.4f}\n"
        result += f"- **Interpretation**: This indicates a {interpret_correlation(spearman_corr)} monotonic relationship.\n\n"
        
        # Add explanation of differences if substantial
        if abs(pearson_corr - spearman_corr) > 0.1:
            result += "## Note on Different Correlation Methods\n"
            result += "There is a notable difference between the Pearson and Spearman correlations. "
            result += "This suggests the relationship may be monotonic but not strictly linear, "
            result += "or there may be outliers affecting the Pearson correlation.\n\n"
        
        # Explanation
        result += "## Explanation\n"
        result += "- **Pearson Correlation** measures linear relationships (how well the relationship fits a straight line).\n"
        result += "- **Spearman Correlation** measures monotonic relationships (whether variables increase/decrease together, regardless of at what rate).\n"
        result += "- A p-value < 0.05 indicates we can reject the null hypothesis that there is no relationship between the variables.\n\n"
        
        result += "## Recommendation\n"
        
        if abs(pearson_corr) > 0.3 or abs(spearman_corr) > 0.3:
            result += "Consider visualizing this relationship with a scatter plot to better understand the pattern."
            if abs(pearson_corr - spearman_corr) > 0.1:
                result += " Pay attention to potential non-linear patterns or outliers in the data."
        else:
            result += "The correlation is relatively weak. There might not be a meaningful relationship between these variables."
        
        return result
    
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"



def filter_data(filename: str, condition: str, output_filename: str = None, overwrite: bool = False) -> str:
    """
    Filter data based on a condition and save as a new dataset.
    
    Args:
        filename: Name of the CSV file
        condition: Python expression for filtering (e.g., "age > 30 and income < 70000")
        output_filename: Name of the output CSV file (if None, just returns summary)
        overwrite: Whether to overwrite an existing output file
        
    Returns:
        Summary of filtered data
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return f"Error: File {filename} not found."
    
    # Handle output filename
    if output_filename:
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        
        output_path = DATA_DIR / output_filename
        
        if output_path.exists() and not overwrite:
            return f"Error: Output file {output_filename} already exists. Set overwrite=True to replace it."
    
    try:
        df = pd.read_csv(filepath)
        
        # Apply the filter condition
        try:
            filtered_df = df.query(condition)
        except Exception as e:
            return f"Error applying filter condition: {str(e)}"
        
        # Generate summary
        summary = f"# Filtered Data Summary\n\n"
        summary += f"- **Original Dataset**: {filename}\n"
        summary += f"- **Filter Condition**: {condition}\n"
        summary += f"- **Original Rows**: {len(df):,}\n"
        summary += f"- **Filtered Rows**: {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.2f}% of original)\n\n"
        
        # Save to output file if requested
        if output_filename:
            filtered_df.to_csv(output_path, index=False)
            summary += f"**Result saved to**: {output_filename}\n\n"
        
        # Add basic statistics
        if not filtered_df.empty:
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary += "## Numeric Column Statistics (Filtered Data)\n"
                summary += f"```\n{filtered_df[numeric_cols].describe().to_string()}\n```\n\n"
        
        return summary
    
    except Exception as e:
        return f"Error filtering data: {str(e)}"



def transform_column(filename: str, column: str, transformation: str, new_column: str = None, 
                     output_filename: str = None, overwrite: bool = False) -> str:
    """
    Apply a transformation to a column in the dataset.
    
    Args:
        filename: Name of the CSV file
        column: Column to transform
        transformation: Type of transformation ('log', 'sqrt', 'square', 'standardize', 'normalize', 'bin', 'abs')
        new_column: Name for the transformed column (if None, overwrites original)
        output_filename: Name of the output CSV file (if None, returns summary without saving)
        overwrite: Whether to overwrite existing columns/files
        
    Returns:
        Summary of the transformation results
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return f"Error: File {filename} not found."
    
    # Handle output filename
    if output_filename:
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        
        output_path = DATA_DIR / output_filename
        
        if output_path.exists() and not overwrite:
            return f"Error: Output file {output_filename} already exists. Set overwrite=True to replace it."
    
    # Valid transformations
    valid_transformations = ['log', 'sqrt', 'square', 'standardize', 'normalize', 'bin', 'abs']
    if transformation not in valid_transformations:
        return f"Error: Invalid transformation. Choose from: {', '.join(valid_transformations)}"
    
    try:
        df = pd.read_csv(filepath)
        
        # Check if column exists
        if column not in df.columns:
            return f"Error: Column '{column}' not found in dataset."
        
        # Check if output column already exists
        if new_column and new_column in df.columns and not overwrite:
            return f"Error: Column '{new_column}' already exists in dataset. Set overwrite=True to replace it."
        
        # Use original column name if new_column not specified
        output_column = new_column if new_column else column
        
        # Apply transformation
        if transformation == 'log':
            # Check for non-positive values
            if (df[column] <= 0).any():
                return f"Error: Log transformation cannot be applied to non-positive values in '{column}'."
            
            df[output_column] = np.log(df[column])
            transform_desc = "Natural logarithm (ln)"
            
        elif transformation == 'sqrt':
            # Check for negative values
            if (df[column] < 0).any():
                return f"Error: Square root transformation cannot be applied to negative values in '{column}'."
            
            df[output_column] = np.sqrt(df[column])
            transform_desc = "Square root"
            
        elif transformation == 'square':
            df[output_column] = df[column] ** 2
            transform_desc = "Square (x²)"
            
        elif transformation == 'standardize':
            # Z-score standardization
            df[output_column] = (df[column] - df[column].mean()) / df[column].std()
            transform_desc = "Standardization (z-score)"
            
        elif transformation == 'normalize':
            # Min-max normalization
            min_val = df[column].min()
            max_val = df[column].max()
            
            if min_val == max_val:
                return f"Error: Cannot normalize '{column}' because all values are identical."
            
            df[output_column] = (df[column] - min_val) / (max_val - min_val)
            transform_desc = "Min-max normalization [0,1]"
            
        elif transformation == 'bin':
            # Binning into 10 equal-width bins
            df[output_column] = pd.cut(df[column], bins=10, labels=False)
            transform_desc = "Binning (10 equal-width bins)"
            
        elif transformation == 'abs':
            # Absolute value
            df[output_column] = df[column].abs()
            transform_desc = "Absolute value"
        
        # Generate summary
        summary = f"# Column Transformation Summary\n\n"
        summary += f"- **Dataset**: {filename}\n"
        summary += f"- **Source Column**: {column}\n"
        summary += f"- **Transformation**: {transform_desc}\n"
        summary += f"- **Output Column**: {output_column}\n\n"
        
        # Add before/after statistics
        summary += "## Before/After Statistics\n\n"
        
        before_stats = df[column].describe()
        after_stats = df[output_column].describe()
        
        stats_table = pd.DataFrame({
            'Before': before_stats,
            'After': after_stats
        })
        
        summary += f"```\n{stats_table.to_string()}\n```\n\n"
        
        # Add skewness comparison
        try:
            before_skew = df[column].skew()
            after_skew = df[output_column].skew()
            
            summary += "## Skewness Comparison\n\n"
            summary += f"- **Before**: {before_skew:.4f}\n"
            summary += f"- **After**: {after_skew:.4f}\n\n"
            
            if abs(after_skew) < abs(before_skew):
                summary += "✓ The transformation reduced the skewness of the data.\n\n"
            else:
                summary += "Note: The transformation did not reduce the skewness of the data.\n\n"
        except Exception:
            pass
        
        # Save to output file if requested
        if output_filename:
            df.to_csv(output_path, index=False)
            summary += f"**Result saved to**: {output_filename}\n\n"
        
        return summary
    
    except Exception as e:
        return f"Error transforming column: {str(e)}"



def group_and_aggregate(filename: str, group_by: Union[str, List[str]], aggregate_cols: List[str], 
                      aggregate_funcs: List[str], output_filename: str = None, overwrite: bool = False) -> str:
    """
    Group data by column(s) and calculate aggregate statistics.
    
    Args:
        filename: Name of the CSV file
        group_by: Column(s) to group by (string or list of strings)
        aggregate_cols: Columns to aggregate
        aggregate_funcs: Aggregate functions to apply (e.g., ['mean', 'sum', 'count'])
        output_filename: Name of the output CSV file (if None, returns summary without saving)
        overwrite: Whether to overwrite an existing output file
        
    Returns:
        Summary of the aggregation results
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return f"Error: File {filename} not found."
    
    # Handle output filename
    if output_filename:
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        
        output_path = DATA_DIR / output_filename
        
        if output_path.exists() and not overwrite:
            return f"Error: Output file {output_filename} already exists. Set overwrite=True to replace it."
    
    # Valid aggregation functions
    valid_aggfuncs = ['mean', 'median', 'sum', 'min', 'max', 'count', 'std', 'var', 'first', 'last']
    invalid_funcs = [func for func in aggregate_funcs if func not in valid_aggfuncs]
    
    if invalid_funcs:
        return f"Error: Invalid aggregation function(s): {', '.join(invalid_funcs)}. " \
               f"Valid options are: {', '.join(valid_aggfuncs)}"
    
    try:
        df = pd.read_csv(filepath)
        
        # Convert single column to list if needed
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Check if columns exist
        for col in group_by:
            if col not in df.columns:
                return f"Error: Group-by column '{col}' not found in dataset."
        
        for col in aggregate_cols:
            if col not in df.columns:
                return f"Error: Aggregate column '{col}' not found in dataset."
        
        # Create aggregation dictionary
        agg_dict = {col: aggregate_funcs for col in aggregate_cols}
        
        # Perform groupby and aggregation
        grouped = df.groupby(group_by).agg(agg_dict)
        
        # Generate summary
        summary = f"# Group and Aggregate Summary\n\n"
        summary += f"- **Dataset**: {filename}\n"
        summary += f"- **Grouped by**: {', '.join(group_by)}\n"
        summary += f"- **Aggregated columns**: {', '.join(aggregate_cols)}\n"
        summary += f"- **Aggregation functions**: {', '.join(aggregate_funcs)}\n"
        summary += f"- **Result groups**: {len(grouped):,}\n\n"
        
        # Add preview of results
        summary += "## Result Preview (first 10 rows)\n"
        summary += f"```\n{grouped.head(10).to_string()}\n```\n\n"
        
        # Save to output file if requested
        if output_filename:
            grouped.to_csv(output_path)
            summary += f"**Result saved to**: {output_filename}\n\n"
        
        # Add insights
        summary += "## Quick Insights\n\n"
        
        # Count of each group
        if len(group_by) == 1 and len(grouped) <= 10:
            group_counts = df[group_by[0]].value_counts()
            summary += "### Group Counts\n"
            
            for group, count in group_counts.items():
                summary += f"- **{group}**: {count:,} rows ({count/len(df)*100:.1f}%)\n"
            
            summary += "\n"
        
        # Find groups with extreme values
        for col in aggregate_cols:
            if 'mean' in aggregate_funcs or 'sum' in aggregate_funcs:
                try:
                    metric = 'mean' if 'mean' in aggregate_funcs else 'sum'
                    series = grouped[col][metric].sort_values(ascending=False)
                    
                    highest_group = series.index[0]
                    highest_value = series.iloc[0]
                    
                    lowest_group = series.index[-1]
                    lowest_value = series.iloc[-1]
                    
                    summary += f"### {col} ({metric})\n"
                    summary += f"- **Highest**: {highest_group} ({highest_value:.2f})\n"
                    summary += f"- **Lowest**: {lowest_group} ({lowest_value:.2f})\n\n"
                except Exception:
                    pass
        
        return summary
    
    except Exception as e:
        return f"Error performing aggregation: {str(e)}"



def handle_missing_values(filename: str, columns: List[str] = None, method: str = 'mean', 
                         output_filename: str = None, overwrite: bool = False) -> str:
    """
    Handle missing values in a dataset.
    
    Args:
        filename: Name of the CSV file
        columns: List of columns to process (if None, processes all columns with missing values)
        method: Imputation method - 'mean', 'median', 'mode', 'constant', 'drop_rows', 'ffill', 'bfill'
        output_filename: Name of the output CSV file (if None, returns summary without saving)
        overwrite: Whether to overwrite an existing output file
        
    Returns:
        Summary of the missing value handling
    """
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        return f"Error: File {filename} not found."
    
    # Handle output filename
    if output_filename:
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        
        output_path = DATA_DIR / output_filename
        
        if output_path.exists() and not overwrite:
            return f"Error: Output file {output_filename} already exists. Set overwrite=True to replace it."
    
    # Valid methods
    valid_methods = ['mean', 'median', 'mode', 'constant', 'drop_rows', 'ffill', 'bfill']
    if method not in valid_methods:
        return f"Error: Invalid method. Choose from: {', '.join(valid_methods)}"
    
    try:
        df = pd.read_csv(filepath)
        
        # Check for missing values
        missing = df.isnull().sum()
        columns_with_missing = missing[missing > 0].index.tolist()
        
        if not columns_with_missing:
            return "No missing values found in the dataset."
        
        # If columns not specified, use all columns with missing values
        if not columns:
            columns = columns_with_missing
        else:
            # Check if specified columns exist
            for col in columns:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found in dataset."
            
            # Filter to only include columns with missing values
            columns = [col for col in columns if col in columns_with_missing]
            
            if not columns:
                return "None of the specified columns have missing values."
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Handle missing values based on the specified method
        if method == 'drop_rows':
            # Drop rows with missing values in specified columns
            before_count = len(result_df)
            result_df = result_df.dropna(subset=columns)
            dropped_count = before_count - len(result_df)
            method_desc = f"Dropped {dropped_count} rows with missing values"
            
        else:
            # Handle each column
            for col in columns:
                if method == 'mean':
                    # Only applicable to numeric columns
                    if not pd.api.types.is_numeric_dtype(result_df[col]):
                        return f"Error: Mean imputation not applicable to non-numeric column '{col}'."
                    
                    fill_value = result_df[col].mean()
                    result_df[col] = result_df[col].fillna(fill_value)
                    method_desc = "Mean imputation"
                    
                elif method == 'median':
                    # Only applicable to numeric columns
                    if not pd.api.types.is_numeric_dtype(result_df[col]):
                        return f"Error: Median imputation not applicable to non-numeric column '{col}'."
                    
                    fill_value = result_df[col].median()
                    result_df[col] = result_df[col].fillna(fill_value)
                    method_desc = "Median imputation"
                    
                elif method == 'mode':
                    fill_value = result_df[col].mode()[0]
                    result_df[col] = result_df[col].fillna(fill_value)
                    method_desc = "Mode imputation"
                    
                elif method == 'constant':
                    # Use 0 for numeric, 'missing' for non-numeric
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        fill_value = 0
                    else:
                        fill_value = 'missing'
                    
                    result_df[col] = result_df[col].fillna(fill_value)
                    method_desc = f"Constant imputation ('{fill_value}')"
                    
                elif method == 'ffill':
                    # Forward fill
                    result_df[col] = result_df[col].fillna(method='ffill')
                    method_desc = "Forward fill"
                    
                elif method == 'bfill':
                    # Backward fill
                    result_df[col] = result_df[col].fillna(method='bfill')
                    method_desc = "Backward fill"
        
        # Generate summary
        summary = f"# Missing Values Handling Summary\n\n"
        summary += f"- **Dataset**: {filename}\n"
        summary += f"- **Method**: {method_desc}\n"
        summary += f"- **Processed columns**: {', '.join(columns)}\n\n"
        
        # Add before/after statistics
        missing_before = df[columns].isnull().sum()
        missing_after = result_df[columns].isnull().sum()
        
        summary += "## Missing Values Before/After\n\n"
        summary += "| Column | Before | After |\n"
        summary += "|--------|--------|-------|\n"
        
        for col in columns:
            summary += f"| {col} | {missing_before[col]} | {missing_after[col]} |\n"
        
        summary += "\n"
        
        # Save to output file if requested
        if output_filename:
            result_df.to_csv(output_path, index=False)
            summary += f"**Result saved to**: {output_filename}\n\n"
        
        return summary
    
    except Exception as e:
        return f"Error handling missing values: {str(e)}"


# @mcp.tool()
# def create_feature(filename: str, new_column: str, expression: str, 
#                   output_filename: str = None, overwrite: bool = False) -> str:
#     """
#     Create a new feature/column based on a Python expression.
    
#     Args:
#         filename: Name of the CSV file
#         new_column: Name for the new feature column
#         expression: Python expression to compute the feature (e.g., "age * income / 1000")
#         output_filename: Name of the output CSV file (if None, returns summary without saving)
#         overwrite: Whether to overwrite an existing column/file
        
#     Returns:
#         Summary of the feature creation
#     """
#     # Ensure filename has .csv extension
#     if not filename.endswith('.csv'):
#         filename += '.csv'
    
#     filepath = DATA_DIR / filename
    
#     if not filepath.exists():
#         return f"Error: File {filename} not found."
    
#     # Handle output filename
#     if output_filename:
#         if not output_filename.endswith('.csv'):
#             output_filename += '.csv'
        
#         output_path = DATA_DIR / output_filename
        
#         if output_path.exists() and not overwrite:
#             return f"Error: Output file {output_filename} already exists. Set overwrite=True to replace it."
    
#     try:
#         df = pd.read_csv(filepath)
        
#         # Check if new column already exists
#         if new_column in df.columns and not overwrite:
#             return f"Error: Column '{new_column}' already exists in dataset. Set overwrite=True to replace it."
        
#         # Evaluate the expression
#         try:
#             # Use pandas eval for safety and to allow column references
#             df[new_column] = df.eval(expression)
#         except Exception as e:
#             return f"Error evaluating expression: {str(e)}"
        
#         # Generate summary
#         summary = f"# Feature Creation Summary\n\n"
#         summary += f"- **Dataset**: {filename}\n"
#         summary += f"- **New Feature**: {new_column}\n"
#         summary += f"- **Expression**: {expression}\n\n"
        
#         # Add statistics about the new feature
#         if pd.api.types.is_numeric_dtype(df[new_column]):
#             summary += "## Numeric Feature Statistics\n\n"
#             stats = df[new_column].describe()
            
#             summary += f"```\n{stats.to_string()}\n```\n\n"
            
#             # Add distribution info
#             summary += "## Distribution Information\n\n"
#             summary += f"- **Skewness**: {df[new_column].skew():.4f}\n"
#             summary += f"- **Kurtosis**: {df[new_column].kurt():.4f}\n"
#             summary += f"- **Missing Values**: {df[new_column].isnull().sum()} ({df[new_column].isnull().mean()*100:.2f}%)\n\n"
            
#         else:
#             summary += "## Categorical Feature Statistics\n\n"
#             value_counts = df[new_column].value_counts().head(10)
            
#             summary += "Top values:\n\n"
#             summary += f"```\n{value_counts.to_string()}\n```\n\n"
            
#             if len(value_counts) > 10:
#                 summary += f"(showing top 10 out of {df[new_column].nunique()} unique values)\n\n"
            
#             summary += f"- **Unique Values**: {df[new_column].nunique()}\n"
#             summary += f"- **Missing Values**: {df[new_column].isnull().sum()} ({df[new_column].isnull().mean()*100:.2f}%)\n\n"
        
#         # Add correlation with other numeric columns
#         if pd.api.types.is_numeric_dtype(df[new_column]):
#             numeric_cols = df.select_dtypes(include=[np.