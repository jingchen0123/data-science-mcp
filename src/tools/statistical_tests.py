"""
Statistical tests tools for the Data Science MCP.
This module contains tools for performing various statistical tests, hypothesis testing,
and statistical assumption checking.
"""

from mcp.server.fastmcp import FastMCP, Context, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import io
from typing import Optional, Dict, List, Any, Union, Tuple

# The DATA_DIR will be initialized by the main server module
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
    mcp_instance.add_tool(run_ttest)
    mcp_instance.add_tool(run_paired_ttest)
    mcp_instance.add_tool(run_anova)
    mcp_instance.add_tool(run_chi_square)
    mcp_instance.add_tool(run_correlation_test)
    mcp_instance.add_tool(run_regression)
    mcp_instance.add_tool(check_normality)
    mcp_instance.add_tool(check_homogeneity)
    mcp_instance.add_tool(power_analysis)
    mcp_instance.add_tool(effect_size)


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


def run_ttest(file_path: str, column: str, test_value: float = 0.0, 
             alternative: str = "two-sided", alpha: float = 0.05) -> str:
    """
    Perform a one-sample t-test to test if the mean of a sample is significantly different from a specified value.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column: Column to test
        test_value: Value to test the mean against (default: 0.0)
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
        alpha: Significance level (default: 0.05)
        
    Returns:
        T-test results and interpretation
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
        
        # Check alternative hypothesis parameter
        valid_alternatives = ['two-sided', 'less', 'greater']
        if alternative not in valid_alternatives:
            return f"Error: Alternative must be one of {', '.join(valid_alternatives)}"
        
        # Drop missing values
        data = df[column].dropna()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(data, test_value, alternative=alternative)
        
        # Calculate effect size (Cohen's d)
        cohen_d = abs(data.mean() - test_value) / data.std()
        
        # Create result
        result = f"# One-Sample T-Test Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Column**: {column}\n"
        result += f"**Test Value**: {test_value}\n"
        result += f"**Alternative Hypothesis**: {alternative}\n"
        result += f"**Significance Level (α)**: {alpha}\n\n"
        
        result += "## Sample Statistics\n\n"
        result += f"- Sample Size: {len(data)}\n"
        result += f"- Sample Mean: {data.mean():.4f}\n"
        result += f"- Sample Standard Deviation: {data.std():.4f}\n"
        result += f"- Standard Error of the Mean: {data.std() / np.sqrt(len(data)):.4f}\n\n"
        
        result += "## Test Results\n\n"
        result += f"- t-statistic: {t_stat:.4f}\n"
        result += f"- p-value: {p_value:.4f}\n"
        result += f"- Cohen's d (effect size): {cohen_d:.4f}\n\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        if p_value < alpha:
            result += f"**Result**: Reject the null hypothesis (p = {p_value:.4f} < α = {alpha})\n\n"
            
            if alternative == 'two-sided':
                result += f"The mean of '{column}' ({data.mean():.4f}) is significantly different from {test_value}.\n"
            elif alternative == 'less':
                result += f"The mean of '{column}' ({data.mean():.4f}) is significantly less than {test_value}.\n"
            elif alternative == 'greater':
                result += f"The mean of '{column}' ({data.mean():.4f}) is significantly greater than {test_value}.\n"
        else:
            result += f"**Result**: Fail to reject the null hypothesis (p = {p_value:.4f} > α = {alpha})\n\n"
            
            if alternative == 'two-sided':
                result += f"There is not enough evidence to conclude that the mean of '{column}' ({data.mean():.4f}) is different from {test_value}.\n"
            elif alternative == 'less':
                result += f"There is not enough evidence to conclude that the mean of '{column}' ({data.mean():.4f}) is less than {test_value}.\n"
            elif alternative == 'greater':
                result += f"There is not enough evidence to conclude that the mean of '{column}' ({data.mean():.4f}) is greater than {test_value}.\n"
        
        # Effect size interpretation
        result += "\n**Effect Size Interpretation**:\n"
        if cohen_d < 0.2:
            result += "The effect size is very small.\n"
        elif cohen_d < 0.5:
            result += "The effect size is small.\n"
        elif cohen_d < 0.8:
            result += "The effect size is medium.\n"
        else:
            result += "The effect size is large.\n"
            
        return result
    
    except Exception as e:
        return f"Error performing t-test: {str(e)}"


def run_paired_ttest(file_path: str, column1: str, column2: str, 
                    alternative: str = "two-sided", alpha: float = 0.05) -> str:
    """
    Perform a paired-sample t-test to compare the means of two related samples.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column1: First column name
        column2: Second column name
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
        alpha: Significance level (default: 0.05)
        
    Returns:
        Paired t-test results and interpretation
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
        
        # Check alternative hypothesis parameter
        valid_alternatives = ['two-sided', 'less', 'greater']
        if alternative not in valid_alternatives:
            return f"Error: Alternative must be one of {', '.join(valid_alternatives)}"
        
        # Drop rows with missing values in either column
        valid_data = df[[column1, column2]].dropna()
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(valid_data[column1], valid_data[column2], alternative=alternative)
        
        # Calculate effect size (Cohen's d for paired samples)
        d = (valid_data[column1] - valid_data[column2]).mean() / (valid_data[column1] - valid_data[column2]).std()
        cohen_d = abs(d)
        
        # Create result
        result = f"# Paired-Sample T-Test Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Columns**: {column1} vs {column2}\n"
        result += f"**Alternative Hypothesis**: {alternative}\n"
        result += f"**Significance Level (α)**: {alpha}\n\n"
        
        result += "## Sample Statistics\n\n"
        result += f"- Sample Size: {len(valid_data)}\n"
        result += f"- Mean of '{column1}': {valid_data[column1].mean():.4f}\n"
        result += f"- Mean of '{column2}': {valid_data[column2].mean():.4f}\n"
        result += f"- Mean Difference: {(valid_data[column1] - valid_data[column2]).mean():.4f}\n"
        result += f"- Standard Deviation of Differences: {(valid_data[column1] - valid_data[column2]).std():.4f}\n\n"
        
        result += "## Test Results\n\n"
        result += f"- t-statistic: {t_stat:.4f}\n"
        result += f"- p-value: {p_value:.4f}\n"
        result += f"- Cohen's d (effect size): {cohen_d:.4f}\n\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        if p_value < alpha:
            result += f"**Result**: Reject the null hypothesis (p = {p_value:.4f} < α = {alpha})\n\n"
            
            if alternative == 'two-sided':
                result += f"There is a significant difference between the means of '{column1}' and '{column2}'.\n"
            elif alternative == 'less':
                result += f"The mean of '{column1}' is significantly less than the mean of '{column2}'.\n"
            elif alternative == 'greater':
                result += f"The mean of '{column1}' is significantly greater than the mean of '{column2}'.\n"
        else:
            result += f"**Result**: Fail to reject the null hypothesis (p = {p_value:.4f} > α = {alpha})\n\n"
            
            if alternative == 'two-sided':
                result += f"There is not enough evidence to conclude that the means of '{column1}' and '{column2}' are different.\n"
            elif alternative == 'less':
                result += f"There is not enough evidence to conclude that the mean of '{column1}' is less than the mean of '{column2}'.\n"
            elif alternative == 'greater':
                result += f"There is not enough evidence to conclude that the mean of '{column1}' is greater than the mean of '{column2}'.\n"
        
        # Effect size interpretation
        result += "\n**Effect Size Interpretation**:\n"
        if cohen_d < 0.2:
            result += "The effect size is very small.\n"
        elif cohen_d < 0.5:
            result += "The effect size is small.\n"
        elif cohen_d < 0.8:
            result += "The effect size is medium.\n"
        else:
            result += "The effect size is large.\n"
            
        return result
    
    except Exception as e:
        return f"Error performing paired t-test: {str(e)}"


def run_anova(file_path: str, value_column: str, group_column: str, alpha: float = 0.05,
              post_hoc: bool = True) -> str:
    """
    Perform one-way ANOVA to test if there are significant differences between group means.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        value_column: Column containing the values to compare
        group_column: Column containing the group labels
        alpha: Significance level (default: 0.05)
        post_hoc: Whether to perform post-hoc tests (Tukey's HSD) if ANOVA is significant
        
    Returns:
        ANOVA results and interpretation
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if value_column not in df.columns:
            return f"Error: Value column '{value_column}' not found in {filepath}."
        if group_column not in df.columns:
            return f"Error: Group column '{group_column}' not found in {filepath}."
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return f"Error: Value column '{value_column}' is not numeric."
        
        # Drop rows with missing values in either column
        valid_data = df[[value_column, group_column]].dropna()
        
        # Get unique groups
        groups = valid_data[group_column].unique()
        
        if len(groups) < 2:
            return f"Error: Need at least two groups for ANOVA, but found only {len(groups)} in '{group_column}'."
        
        # Split data by groups
        group_data = [valid_data[valid_data[group_column] == group][value_column].values for group in groups]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Calculate effect size (eta-squared)
        # Between-group sum of squares
        grand_mean = valid_data[value_column].mean()
        ss_between = sum(len(data) * ((data.mean() - grand_mean) ** 2) for data in group_data)
        # Total sum of squares
        ss_total = sum((valid_data[value_column] - grand_mean) ** 2)
        eta_squared = ss_between / ss_total
        
        # Create result
        result = f"# One-Way ANOVA Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Value Column**: {value_column}\n"
        result += f"**Group Column**: {group_column}\n"
        result += f"**Significance Level (α)**: {alpha}\n\n"
        
        result += "## Sample Statistics\n\n"
        result += f"- Number of Groups: {len(groups)}\n"
        result += f"- Total Sample Size: {len(valid_data)}\n\n"
        
        result += "| Group | Count | Mean | Std Dev | Min | Max |\n"
        result += "|-------|-------|------|---------|-----|-----|\n"
        
        for group, data in zip(groups, group_data):
            result += f"| {group} | {len(data)} | {data.mean():.4f} | {data.std():.4f} | {data.min():.4f} | {data.max():.4f} |\n"
        
        result += f"\n- Grand Mean: {grand_mean:.4f}\n\n"
        
        result += "## Test Results\n\n"
        result += f"- F-statistic: {f_stat:.4f}\n"
        result += f"- p-value: {p_value:.4f}\n"
        result += f"- Eta-squared (effect size): {eta_squared:.4f}\n\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        if p_value < alpha:
            result += f"**Result**: Reject the null hypothesis (p = {p_value:.4f} < α = {alpha})\n\n"
            result += f"There are significant differences in the means of '{value_column}' across different groups in '{group_column}'.\n"
            
            # Effect size interpretation
            if eta_squared < 0.01:
                result += "\nThe effect size is very small (η² < 0.01).\n"
            elif eta_squared < 0.06:
                result += "\nThe effect size is small (η² < 0.06).\n"
            elif eta_squared < 0.14:
                result += "\nThe effect size is medium (η² < 0.14).\n"
            else:
                result += "\nThe effect size is large (η² ≥ 0.14).\n"
            
            # Perform post-hoc tests if requested
            if post_hoc and len(groups) > 2:
                result += "\n## Post-Hoc Tests (Tukey's HSD)\n\n"
                
                # Convert to format needed for Tukey's test
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                posthoc = pairwise_tukeyhsd(
                    valid_data[value_column],
                    valid_data[group_column],
                    alpha=alpha
                )
                
                # Convert to string and format as table
                result += "| Group 1 | Group 2 | Mean Diff | Lower CI | Upper CI | p-value | Significant |\n"
                result += "|---------|---------|-----------|----------|----------|---------|-------------|\n"
                
                # Extract data from the posthoc results
                data = posthoc.summary().data[1:]  # Skip header
                for row in data:
                    group1, group2, meandiff, lower, upper, reject = row
                    p_adjusted = 'N/A'  # Adjusted p-values not directly available from pairwise_tukeyhsd
                    significant = 'Yes' if reject else 'No'
                    result += f"| {group1} | {group2} | {float(meandiff):.4f} | {float(lower):.4f} | {float(upper):.4f} | {p_adjusted} | {significant} |\n"
        else:
            result += f"**Result**: Fail to reject the null hypothesis (p = {p_value:.4f} > α = {alpha})\n\n"
            result += f"There is not enough evidence to conclude that there are differences in the means of '{value_column}' across different groups in '{group_column}'.\n"
        
        return result
    
    except Exception as e:
        return f"Error performing ANOVA: {str(e)}"


def run_chi_square(file_path: str, column1: str, column2: str, alpha: float = 0.05) -> str:
    """
    Perform a chi-square test of independence to test if there is a relationship between two categorical variables.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column1: First categorical column
        column2: Second categorical column
        alpha: Significance level (default: 0.05)
        
    Returns:
        Chi-square test results and interpretation
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
        
        # Check if either column has too many unique values
        # For chi-square, we generally want categorical variables, not continuous ones
        max_categories = 20  # Arbitrary limit to prevent trying to use continuous variables
        if df[column1].nunique() > max_categories:
            return f"Error: Column '{column1}' has too many unique values ({df[column1].nunique()}). Consider binning it first."
        if df[column2].nunique() > max_categories:
            return f"Error: Column '{column2}' has too many unique values ({df[column2].nunique()}). Consider binning it first."
        
        # Create contingency table
        contingency_table = pd.crosstab(df[column1], df[column2])
        
        # Check if the table has enough data for a chi-square test
        if contingency_table.size <= 1:
            return f"Error: Cannot perform chi-square test. The contingency table has only one cell."
        
        # Check for cells with expected frequencies less than 5
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        low_expected_count = (expected < 5).sum()
        total_cells = expected.size
        percent_low_expected = (low_expected_count / total_cells) * 100
        
        # Calculate Cramer's V for effect size
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        # Create result
        result = f"# Chi-Square Test of Independence Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Columns**: {column1} vs {column2}\n"
        result += f"**Significance Level (α)**: {alpha}\n\n"
        
        result += "## Contingency Table\n\n"
        result += contingency_table.to_markdown() + "\n\n"
        
        result += "## Test Results\n\n"
        result += f"- Chi-square statistic: {chi2:.4f}\n"
        result += f"- Degrees of freedom: {dof}\n"
        result += f"- p-value: {p:.4f}\n"
        result += f"- Cramer's V (effect size): {cramers_v:.4f}\n\n"
        
        # Warning about expected frequencies
        if low_expected_count > 0:
            result += f"**Note**: {low_expected_count} cells ({percent_low_expected:.1f}%) have expected frequencies less than 5.\n"
            if percent_low_expected > 20:
                result += "This violates an assumption of the chi-square test. Consider collapsing categories or using Fisher's exact test.\n\n"
            else:
                result += "This is within acceptable limits (<20% of cells), but be cautious in interpretation.\n\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        if p < alpha:
            result += f"**Result**: Reject the null hypothesis (p = {p:.4f} < α = {alpha})\n\n"
            result += f"There is a significant association between '{column1}' and '{column2}'.\n"
            
            # Effect size interpretation
            if cramers_v < 0.1:
                result += "\nThe effect size is very small (Cramer's V < 0.1).\n"
            elif cramers_v < 0.3:
                result += "\nThe effect size is small (Cramer's V < 0.3).\n"
            elif cramers_v < 0.5:
                result += "\nThe effect size is medium (Cramer's V < 0.5).\n"
            else:
                result += "\nThe effect size is large (Cramer's V ≥ 0.5).\n"
        else:
            result += f"**Result**: Fail to reject the null hypothesis (p = {p:.4f} > α = {alpha})\n\n"
            result += f"There is not enough evidence to conclude that there is an association between '{column1}' and '{column2}'.\n"
        
        return result
    
    except Exception as e:
        return f"Error performing chi-square test: {str(e)}"


def run_correlation_test(file_path: str, column1: str, column2: str, 
                        method: str = "pearson", alpha: float = 0.05) -> str:
    """
    Perform a correlation test between two numeric variables.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column1: First numeric column
        column2: Second numeric column
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        alpha: Significance level (default: 0.05)
        
    Returns:
        Correlation test results and interpretation
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
        
        # Check correlation method
        valid_methods = ['pearson', 'spearman', 'kendall']
        if method not in valid_methods:
            return f"Error: Method must be one of {', '.join(valid_methods)}"
        
        # Drop rows with missing values in either column
        valid_data = df[[column1, column2]].dropna()
        
        if len(valid_data) < 3:
            return f"Error: Not enough valid data points for correlation analysis. Need at least 3 but found {len(valid_data)}."
        
        # Calculate correlation based on the specified method
        if method == 'pearson':
            corr, p_value = stats.pearsonr(valid_data[column1], valid_data[column2])
            method_name = "Pearson's correlation"
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(valid_data[column1], valid_data[column2])
            method_name = "Spearman's rank correlation"
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(valid_data[column1], valid_data[column2])
            method_name = "Kendall's tau correlation"
        
        # Create result
        result = f"# Correlation Test Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Columns**: {column1} vs {column2}\n"
        result += f"**Method**: {method_name}\n"
        result += f"**Significance Level (α)**: {alpha}\n\n"
        
        result += "## Sample Statistics\n\n"
        result += f"- Sample Size: {len(valid_data)}\n"
        result += f"- Mean of '{column1}': {valid_data[column1].mean():.4f}\n"
        result += f"- Mean of '{column2}': {valid_data[column2].mean():.4f}\n"
        result += f"- Standard Deviation of '{column1}': {valid_data[column1].std():.4f}\n"
        result += f"- Standard Deviation of '{column2}': {valid_data[column2].std():.4f}\n\n"
        
        result += "## Test Results\n\n"
        result += f"- Correlation coefficient: {corr:.4f}\n"
        result += f"- p-value: {p_value:.4f}\n"
        result += f"- Coefficient of determination (r²): {corr**2:.4f}\n\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        if p_value < alpha:
            result += f"**Result**: Reject the null hypothesis (p = {p_value:.4f} < α = {alpha})\n\n"
            
            if corr > 0:
                result += f"There is a significant positive correlation between '{column1}' and '{column2}'.\n"
            else:
                result += f"There is a significant negative correlation between '{column1}' and '{column2}'.\n"
            
            # Effect size interpretation based on absolute correlation
            abs_corr = abs(corr)
            if abs_corr < 0.1:
                result += "\nThe correlation is very weak.\n"
            elif abs_corr < 0.3:
                result += "\nThe correlation is weak.\n"
            elif abs_corr < 0.5:
                result += "\nThe correlation is moderate.\n"
            elif abs_corr < 0.7:
                result += "\nThe correlation is moderately strong.\n"
            elif abs_corr < 0.9:
                result += "\nThe correlation is strong.\n"
            else:
                result += "\nThe correlation is very strong.\n"
            
            result += f"\nThe coefficient of determination (r² = {corr**2:.4f}) indicates that {corr**2*100:.1f}% of the variance in one variable can be explained by the other variable.\n"
        else:
            result += f"**Result**: Fail to reject the null hypothesis (p = {p_value:.4f} > α = {alpha})\n\n"
            result += f"There is not enough evidence to conclude that there is a correlation between '{column1}' and '{column2}'.\n"
        
        # Create a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_data[column1], valid_data[column2], alpha=0.7)
        plt.title(f'{method_name} between {column1} and {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        
        # Add regression line if using Pearson's correlation
        if method == 'pearson':
            # Calculate regression line
            from scipy.stats import linregress
            slope, intercept, r, p, stderr = linregress(valid_data[column1], valid_data[column2])
            x = np.array([valid_data[column1].min(), valid_data[column1].max()])
            y = intercept + slope * x
            plt.plot(x, y, 'r-', label=f'y = {slope:.4f}x + {intercept:.4f}')
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Add scatter plot image directly into the result
        img_data = img_buf.getvalue()
        
        # Create image object
        img = Image(data=img_data, format="png")
        
        # Return both text result and image
        return result
    
    except Exception as e:
        return f"Error performing correlation test: {str(e)}"


def run_regression(file_path: str, dependent_var: str, independent_vars: List[str],
                 alpha: float = 0.05) -> str:
    """
    Perform multiple linear regression analysis.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        dependent_var: Dependent variable (y)
        independent_vars: List of independent variables (x)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Regression results and interpretation
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if dependent_var not in df.columns:
            return f"Error: Dependent variable '{dependent_var}' not found in {filepath}."
        
        missing_cols = [col for col in independent_vars if col not in df.columns]
        if missing_cols:
            return f"Error: Independent variables not found: {', '.join(missing_cols)}"
        
        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(df[dependent_var]):
            return f"Error: Dependent variable '{dependent_var}' is not numeric."
        
        non_numeric = [col for col in independent_vars if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            return f"Error: Non-numeric independent variables: {', '.join(non_numeric)}"
        
        # Drop rows with missing values in any of the variables
        variables = [dependent_var] + independent_vars
        valid_data = df[variables].dropna()
        
        if len(valid_data) < len(independent_vars) + 2:
            return f"Error: Not enough valid data points for regression analysis. Need at least {len(independent_vars) + 2} but found {len(valid_data)}."
        
        # Perform multiple linear regression using statsmodels
        import statsmodels.api as sm
        
        # Add constant for intercept
        X = sm.add_constant(valid_data[independent_vars])
        y = valid_data[dependent_var]
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Get regression results
        summary = model.summary()
        
        # Create result
        result = f"# Multiple Linear Regression Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Dependent Variable**: {dependent_var}\n"
        result += f"**Independent Variables**: {', '.join(independent_vars)}\n"
        result += f"**Significance Level (α)**: {alpha}\n\n"
        
        result += "## Model Summary\n\n"
        result += f"- Sample Size: {len(valid_data)}\n"
        result += f"- R-squared: {model.rsquared:.4f}\n"
        result += f"- Adjusted R-squared: {model.rsquared_adj:.4f}\n"
        result += f"- F-statistic: {model.fvalue:.4f}\n"
        result += f"- F-test p-value: {model.f_pvalue:.4f}\n\n"
        
        result += "## Coefficients\n\n"
        result += "| Variable | Coefficient | Std Error | t-value | p-value | Significant |\n"
        result += "|----------|-------------|-----------|---------|---------|-------------|\n"
        
        for idx, var_name in enumerate(['const'] + independent_vars):
            coef = model.params[idx]
            std_err = model.bse[idx]
            t_value = model.tvalues[idx]
            p_value = model.pvalues[idx]
            significant = 'Yes' if p_value < alpha else 'No'
            
            result += f"| {var_name} | {coef:.4f} | {std_err:.4f} | {t_value:.4f} | {p_value:.4f} | {significant} |\n"
        
        result += "\n## Regression Equation\n\n"
        equation = f"{dependent_var} = {model.params[0]:.4f}"
        
        for idx, var in enumerate(independent_vars):
            coef = model.params[idx + 1]
            if coef >= 0:
                equation += f" + {coef:.4f} × {var}"
            else:
                equation += f" - {abs(coef):.4f} × {var}"
        
        result += equation + "\n\n"
        
        # Check for multicollinearity using VIF
        if len(independent_vars) > 1:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            result += "## Multicollinearity Check\n\n"
            result += "| Variable | VIF | Multicollinearity |\n"
            result += "|----------|-----|------------------|\n"
            
            for idx, var in enumerate(independent_vars):
                vif = variance_inflation_factor(X.values, idx + 1)
                if vif < 5:
                    multicollinearity = "Low"
                elif vif < 10:
                    multicollinearity = "Moderate"
                else:
                    multicollinearity = "High"
                
                result += f"| {var} | {vif:.4f} | {multicollinearity} |\n"
            
            result += "\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        # Overall model significance
        if model.f_pvalue < alpha:
            result += f"**Model Significance**: The overall model is statistically significant (F-test p-value = {model.f_pvalue:.4f} < α = {alpha}).\n\n"
        else:
            result += f"**Model Significance**: The overall model is not statistically significant (F-test p-value = {model.f_pvalue:.4f} > α = {alpha}).\n\n"
        
        # Model fit
        result += f"**Model Fit**: The model explains {model.rsquared*100:.1f}% of the variance in {dependent_var} (R² = {model.rsquared:.4f}).\n\n"
        
        # Significant predictors
        sig_predictors = [var_name for idx, var_name in enumerate(independent_vars) if model.pvalues[idx + 1] < alpha]
        
        if sig_predictors:
            result += f"**Significant Predictors**: The following variables are significant predictors of {dependent_var} at α = {alpha}:\n"
            for var in sig_predictors:
                coef = model.params[independent_vars.index(var) + 1]
                direction = "positively" if coef > 0 else "negatively"
                result += f"- {var} ({direction} associated, p = {model.pvalues[independent_vars.index(var) + 1]:.4f})\n"
            result += "\n"
        else:
            result += f"**Significant Predictors**: None of the independent variables are significant predictors of {dependent_var} at α = {alpha}.\n\n"
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y, model.predict(), alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.title(f'Actual vs. Predicted Values for {dependent_var}')
        plt.xlabel(f'Actual {dependent_var}')
        plt.ylabel(f'Predicted {dependent_var}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Add scatter plot image directly into the result
        img_data = img_buf.getvalue()
        
        # Create image object
        img = Image(data=img_data, format="png")
        
        # Return both text result and image
        return result
    
    except Exception as e:
        return f"Error performing regression: {str(e)}"


def check_normality(file_path: str, column: str, test: str = "shapiro") -> str:
    """
    Test if a column follows a normal distribution.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column: Column to test for normality
        test: Type of normality test ('shapiro', 'ks', 'anderson')
        
    Returns:
        Normality test results and interpretation
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
        
        # Check test type
        valid_tests = ['shapiro', 'ks', 'anderson']
        if test not in valid_tests:
            return f"Error: Test must be one of {', '.join(valid_tests)}"
        
        # Drop missing values
        data = df[column].dropna().values
        
        if len(data) < 3:
            return f"Error: Not enough valid data points for normality testing. Need at least 3 but found {len(data)}."
        
        if len(data) > 5000 and test == 'shapiro':
            return f"Error: Shapiro-Wilk test is not recommended for sample sizes > 5000. Try 'ks' instead."
        
        # Perform the specified normality test
        if test == 'shapiro':
            # Shapiro-Wilk test
            statistic, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
            
        elif test == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            test_name = "Kolmogorov-Smirnov"
            
        elif test == 'anderson':
            # Anderson-Darling test
            result = stats.anderson(data, dist='norm')
            statistic = result.statistic
            test_name = "Anderson-Darling"
            
            # Get the appropriate critical value and significance level
            critical_values = result.critical_values
            significance_levels = [15, 10, 5, 2.5, 1]
            
            # Find the highest significance level where we can reject normality
            for sig_level, critical_value in zip(significance_levels, critical_values):
                if statistic > critical_value:
                    p_value = sig_level / 100  # Convert percentage to probability
                    break
            else:
                # If we get here, we couldn't reject at any level
                p_value = 0.15  # Default to the highest significance level
        
        # Calculate descriptive statistics
        mean = data.mean()
        median = np.median(data)
        std = data.std()
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Create result
        result = f"# Normality Test Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Column**: {column}\n"
        result += f"**Test**: {test_name}\n\n"
        
        result += "## Descriptive Statistics\n\n"
        result += f"- Sample Size: {len(data)}\n"
        result += f"- Mean: {mean:.4f}\n"
        result += f"- Median: {median:.4f}\n"
        result += f"- Standard Deviation: {std:.4f}\n"
        result += f"- Skewness: {skewness:.4f}\n"
        result += f"- Kurtosis: {kurtosis:.4f}\n\n"
        
        result += "## Test Results\n\n"
        result += f"- Test Statistic: {statistic:.4f}\n"
        
        if test == 'anderson':
            result += "- Critical Values:\n"
            for sig_level, critical_value in zip(significance_levels, critical_values):
                result += f"  - {sig_level}%: {critical_value:.4f}\n"
        else:
            result += f"- p-value: {p_value:.4f}\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        alpha = 0.05  # Standard significance level
        
        if p_value < alpha:
            result += f"**Result**: Reject the null hypothesis (p = {p_value:.4f} < α = {alpha})\n\n"
            result += f"The data in column '{column}' does not follow a normal distribution.\n"
        else:
            result += f"**Result**: Fail to reject the null hypothesis (p = {p_value:.4f} > α = {alpha})\n\n"
            result += f"There is not enough evidence to conclude that the data in column '{column}' deviates from a normal distribution.\n"
        
        # Description of skewness and kurtosis
        result += "\n**Distribution Shape**:\n"
        
        # Skewness interpretation
        if abs(skewness) < 0.5:
            result += "- The data is approximately symmetric (skewness near 0).\n"
        elif skewness < 0:
            result += "- The data is negatively skewed (left-tailed).\n"
        else:
            result += "- The data is positively skewed (right-tailed).\n"
        
        # Kurtosis interpretation
        if abs(kurtosis) < 0.5:
            result += "- The data has a normal kurtosis (similar to normal distribution).\n"
        elif kurtosis < 0:
            result += "- The data is platykurtic (flatter than normal distribution).\n"
        else:
            result += "- The data is leptokurtic (more peaked than normal distribution).\n"
        
        # Create visual assessment of normality
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram with normal curve
        ax1.hist(data, bins=30, density=True, alpha=0.7, label='Data')
        
        # Add normal curve
        x = np.linspace(min(data), max(data), 100)
        y = stats.norm.pdf(x, mean, std)
        ax1.plot(x, y, 'r-', label='Normal Curve')
        
        ax1.set_title(f'Histogram of {column} with Normal Curve')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Add plot image directly into the result
        img_data = img_buf.getvalue()
        
        # Create image object
        img = Image(data=img_data, format="png")
        
        # Return both text result and image
        return result
    
    except Exception as e:
        return f"Error checking normality: {str(e)}"


def check_homogeneity(file_path: str, value_column: str, group_column: str, test: str = "levene") -> str:
    """
    Test for homogeneity of variance across groups.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        value_column: Column containing the values to compare
        group_column: Column containing the group labels
        test: Type of test ('levene' or 'bartlett')
        
    Returns:
        Homogeneity test results and interpretation
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check if columns exist
        if value_column not in df.columns:
            return f"Error: Value column '{value_column}' not found in {filepath}."
        if group_column not in df.columns:
            return f"Error: Group column '{group_column}' not found in {filepath}."
        
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return f"Error: Value column '{value_column}' is not numeric."
        
        # Check test type
        valid_tests = ['levene', 'bartlett']
        if test not in valid_tests:
            return f"Error: Test must be one of {', '.join(valid_tests)}"
        
        # Get groups
        groups = df[group_column].unique()
        
        if len(groups) < 2:
            return f"Error: Need at least two groups for homogeneity testing, but found only {len(groups)}."
        
        # Split data by groups
        group_data = []
        group_sizes = []
        
        for group in groups:
            data = df[df[group_column] == group][value_column].dropna().values
            if len(data) < 3:
                return f"Error: Group '{group}' has insufficient data points ({len(data)}). Each group needs at least 3 data points."
            group_data.append(data)
            group_sizes.append(len(data))
        
        # Perform the homogeneity test
        if test == 'levene':
            # Levene's test
            statistic, p_value = stats.levene(*group_data)
            test_name = "Levene's"
            
        elif test == 'bartlett':
            # Bartlett's test
            statistic, p_value = stats.bartlett(*group_data)
            test_name = "Bartlett's"
        
        # Create result
        result = f"# Homogeneity of Variance Test Results\n\n"
        result += f"**File**: {filepath}\n"
        result += f"**Value Column**: {value_column}\n"
        result += f"**Group Column**: {group_column}\n"
        result += f"**Test**: {test_name} test\n\n"
        
        result += "## Group Statistics\n\n"
        result += "| Group | Count | Mean | Std Dev | Variance |\n"
        result += "|-------|-------|------|---------|----------|\n"
        
        for group, data in zip(groups, group_data):
            result += f"| {group} | {len(data)} | {data.mean():.4f} | {data.std():.4f} | {data.var():.4f} |\n"
        
        result += "\n## Test Results\n\n"
        result += f"- Test Statistic: {statistic:.4f}\n"
        result += f"- p-value: {p_value:.4f}\n\n"
        
        # Interpretation
        result += "## Interpretation\n\n"
        
        alpha = 0.05  # Standard significance level
        
        if p_value < alpha:
            result += f"**Result**: Reject the null hypothesis (p = {p_value:.4f} < α = {alpha})\n\n"
            result += f"The variances of '{value_column}' are significantly different across groups in '{group_column}'.\n"
            result += "\nThis indicates heterogeneity of variance (unequal variances), which violates an assumption of tests like ANOVA.\nConsider using non-parametric alternatives or tests that don't assume equal variances.\n"
        else:
            result += f"**Result**: Fail to reject the null hypothesis (p = {p_value:.4f} > α = {alpha})\n\n"
            result += f"There is not enough evidence to conclude that the variances of '{value_column}' differ across groups in '{group_column}'.\n"
            result += "\nThis suggests homogeneity of variance (equal variances), which satisfies an assumption of tests like ANOVA.\n"
        
        # Create a visual comparison of variances
        plt.figure(figsize=(10, 6))
        group_labels = [str(group) for group in groups]
        group_variances = [data.var() for data in group_data]
        
        plt.bar(group_labels, group_variances, alpha=0.7)
        plt.axhline(y=np.mean(group_variances), color='r', linestyle='--', label='Mean Variance')
        
        plt.title(f'Variance Comparison of {value_column} Across Groups')
        plt.xlabel(group_column)
        plt.ylabel('Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Add plot image directly into the result
        img_data = img_buf.getvalue()
        
        # Create image object
        img = Image(data=img_data, format="png")
        
        # Return both text result and image
        return result
    
    except Exception as e:
        return f"Error checking homogeneity of variance: {str(e)}"


def power_analysis(test_type: str, effect_size: float, alpha: float = 0.05, 
                  power: float = 0.8, **kwargs) -> str:
    """
    Perform statistical power analysis to determine required sample size.
    
    Args:
        test_type: Type of test ('t_test', 'paired_t', 'ind_t', 'anova', 'chi2', 'correlation')
        effect_size: Expected effect size (Cohen's d, f, w, or r)
        alpha: Significance level (default: 0.05)
        power: Desired statistical power (default: 0.8)
        **kwargs: Additional parameters for specific tests
        
    Returns:
        Power analysis results
    """
    try:
        # Import statsmodels for power analysis
        from statsmodels.stats.power import TTestPower, TTestIndPower, FTestAnovaPower, GofChisquarePower, NormalIndPower
        
        # Check test type
        valid_tests = ['t_test', 'paired_t', 'ind_t', 'anova', 'chi2', 'correlation']
        if test_type not in valid_tests:
            return f"Error: Test type must be one of {', '.join(valid_tests)}"
        
        # Check parameters
        if effect_size <= 0:
            return "Error: Effect size must be positive."
        if not (0 < alpha < 1):
            return "Error: Alpha must be between 0 and 1."
        if not (0 < power < 1):
            return "Error: Power must be between 0 and 1."
        
        # Perform power analysis based on test type
        if test_type in ['t_test', 'paired_t']:
            # One-sample or paired t-test
            power_analysis = TTestPower()
            sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
            test_name = "One-sample t-test" if test_type == 't_test' else "Paired t-test"
            effect_size_name = "Cohen's d"
            
        elif test_type == 'ind_t':
            # Independent samples t-test
            ratio = kwargs.get('ratio', 1.0)  # Ratio of sample sizes (N1/N2)
            power_analysis = TTestIndPower()
            sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio, alternative='two-sided')
            test_name = "Independent samples t-test"
            effect_size_name = "Cohen's d"
            
            # Calculate sample sizes for each group
            n1 = sample_size
            n2 = sample_size * ratio
            
        elif test_type == 'anova':
            # One-way ANOVA
            k = kwargs.get('k', 3)  # Number of groups
            if k < 2:
                return "Error: Number of groups (k) must be at least 2 for ANOVA."
            
            power_analysis = FTestAnovaPower()
            sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k)
            test_name = "One-way ANOVA"
            effect_size_name = "Cohen's f"
            
        elif test_type == 'chi2':
            # Chi-square test
            df = kwargs.get('df', 1)  # Degrees of freedom
            if df < 1:
                return "Error: Degrees of freedom must be at least 1 for chi-square test."
            
            power_analysis = GofChisquarePower()
            sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, df=df)
            test_name = "Chi-square test"
            effect_size_name = "Cohen's w"
            
        elif test_type == 'correlation':
            # Correlation test
            power_analysis = NormalIndPower()
            sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
            test_name = "Correlation test"
            effect_size_name = "Correlation coefficient (r)"
        
        # Round up to the nearest integer
        sample_size = int(np.ceil(sample_size))
        
        # Create result
        result = f"# Statistical Power Analysis Results\n\n"
        result += f"**Test Type**: {test_name}\n"
        result += f"**Effect Size ({effect_size_name})**: {effect_size:.4f}\n"
        result += f"**Significance Level (α)**: {alpha}\n"
        result += f"**Desired Power**: {power}\n"
        
        if test_type == 'anova':
            result += f"**Number of Groups**: {k}\n"
        elif test_type == 'chi2':
            result += f"**Degrees of Freedom**: {df}\n"
        elif test_type == 'ind_t':
            result += f"**Group Size Ratio (N1/N2)**: {ratio}\n"
        
        result += "\n## Required Sample Size\n\n"
        
        if test_type == 'ind_t':
            result += f"- Group 1: {int(np.ceil(n1))}\n"
            result += f"- Group 2: {int(np.ceil(n2))}\n"
            result += f"- Total: {int(np.ceil(n1 + n2))}\n"
        elif test_type == 'anova':
            result += f"- Per Group: {sample_size}\n"
            result += f"- Total: {sample_size * k}\n"
        else:
            result += f"- {sample_size}\n"
        
        # Interpretation
        result += "\n## Interpretation\n\n"
        
        if test_type == 't_test':
            result += f"To detect an effect size of {effect_size:.4f} with {power*100}% power at a significance level of {alpha}, you need a sample size of at least {sample_size} observations.\n"
        elif test_type == 'paired_t':
            result += f"To detect an effect size of {effect_size:.4f} with {power*100}% power at a significance level of {alpha}, you need at least {sample_size} pairs of observations.\n"
        elif test_type == 'ind_t':
            result += f"To detect an effect size of {effect_size:.4f} with {power*100}% power at a significance level of {alpha}, you need at least {int(np.ceil(n1))} observations in group 1 and {int(np.ceil(n2))} observations in group 2.\n"
        elif test_type == 'anova':
            result += f"To detect an effect size of {effect_size:.4f} with {power*100}% power at a significance level of {alpha}, you need at least {sample_size} observations in each of the {k} groups, for a total of {sample_size * k} observations.\n"
        elif test_type == 'chi2':
            result += f"To detect an effect size of {effect_size:.4f} with {power*100}% power at a significance level of {alpha}, you need a total sample size of at least {sample_size} observations.\n"
        elif test_type == 'correlation':
            result += f"To detect a correlation of {effect_size:.4f} with {power*100}% power at a significance level of {alpha}, you need a sample size of at least {sample_size} observations.\n"
        
        # Effect size interpretation
        result += "\n## Effect Size Guide\n\n"
        
        if test_type in ['t_test', 'paired_t', 'ind_t']:
            # Cohen's d
            result += "**Cohen's d (for t-tests)**:\n"
            result += "- 0.2: Small effect\n"
            result += "- 0.5: Medium effect\n"
            result += "- 0.8: Large effect\n"
        elif test_type == 'anova':
            # Cohen's f
            result += "**Cohen's f (for ANOVA)**:\n"
            result += "- 0.1: Small effect\n"
            result += "- 0.25: Medium effect\n"
            result += "- 0.4: Large effect\n"
        elif test_type == 'chi2':
            # Cohen's w
            result += "**Cohen's w (for chi-square)**:\n"
            result += "- 0.1: Small effect\n"
            result += "- 0.3: Medium effect\n"
            result += "- 0.5: Large effect\n"
        elif test_type == 'correlation':
            # Correlation coefficient
            result += "**Correlation coefficient (r)**:\n"
            result += "- 0.1: Small effect\n"
            result += "- 0.3: Medium effect\n"
            result += "- 0.5: Large effect\n"
        
        # Power curve plot
        plt.figure(figsize=(10, 6))
        
        # Get power analysis object based on test type
        if test_type in ['t_test', 'paired_t']:
            power_analysis = TTestPower()
            effect_sizes = np.linspace(0.1, 1.0, 10)
            power_values = [power_analysis.solve_power(effect_size=es, nobs=sample_size, alpha=alpha, alternative='two-sided') for es in effect_sizes]
            
        elif test_type == 'ind_t':
            power_analysis = TTestIndPower()
            effect_sizes = np.linspace(0.1, 1.0, 10)
            power_values = [power_analysis.solve_power(effect_size=es, nobs1=n1, ratio=ratio, alpha=alpha, alternative='two-sided') for es in effect_sizes]
            
        elif test_type == 'anova':
            power_analysis = FTestAnovaPower()
            effect_sizes = np.linspace(0.1, 0.5, 10)
            power_values = [power_analysis.solve_power(effect_size=es, nobs=sample_size*k, alpha=alpha, k_groups=k) for es in effect_sizes]
            
        elif test_type == 'chi2':
            power_analysis = GofChisquarePower()
            effect_sizes = np.linspace(0.1, 0.5, 10)
            power_values = [power_analysis.solve_power(effect_size=es, nobs=sample_size, alpha=alpha, df=df) for es in effect_sizes]
            
        elif test_type == 'correlation':
            power_analysis = NormalIndPower()
            effect_sizes = np.linspace(0.1, 0.7, 10)
            power_values = [power_analysis.solve_power(effect_size=es, nobs=sample_size, alpha=alpha, alternative='two-sided') for es in effect_sizes]
        
        # Plot power curve
        plt.plot(effect_sizes, power_values, 'bo-')
        plt.axhline(y=power, color='r', linestyle='--', label=f'Target Power ({power})')
        plt.axvline(x=effect_size, color='g', linestyle='--', label=f'Target Effect Size ({effect_size:.2f})')
        
        plt.title('Statistical Power vs. Effect Size')
        plt.xlabel(f'Effect Size ({effect_size_name})')
        plt.ylabel('Statistical Power')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        
        # Add plot image directly into the result
        img_data = img_buf.getvalue()
        
        # Create image object
        img = Image(data=img_data, format="png")
        
        # Return both text result and image
        return result
    
    except Exception as e:
        return f"Error performing power analysis: {str(e)}"


def effect_size(file_path: str, column1: str, column2: str = None, group_column: str = None,
               test_type: str = 'mean_diff', pooled: bool = True) -> str:
    """
    Calculate effect size for various statistical tests.
    
    Args:
        file_path: Path to the CSV file (absolute or relative to DATA_DIR)
        column1: First data column
        column2: Second data column (for paired or independent tests)
        group_column: Column with group labels (for ANOVA)
        test_type: Type of effect size ('mean_diff', 'paired', 'ind_t', 'anova', 'correlation')
        pooled: Whether to use pooled standard deviation (for independent t-test)
        
    Returns:
        Effect size calculation and interpretation
    """
    filepath = resolve_file_path(file_path, '.csv')
    
    if not filepath.exists():
        return f"Error: File {filepath} not found."
    
    try:
        # Load the data
        df = pd.read_csv(filepath)
        
        # Check test type
        valid_tests = ['mean_diff', 'paired', 'ind_t', 'anova', 'correlation']
        if test_type not in valid_tests:
            return f"Error: Test type must be one of {', '.join(valid_tests)}"
        
        # Create result
        result = f"# Effect Size Calculation\n\n"
        result += f"**File**: {filepath}\n"
        
        # Calculate effect size based on test type
        if test_type == 'mean_diff':
            # Cohen's d for one-sample t-test (against constant or population mean)
            
            # Check if column exists
            if column1 not in df.columns:
                return f"Error: Column '{column1}' not found in {filepath}."
            
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(df[column1]):
                return f"Error: Column '{column1}' is not numeric."
            
            # Drop missing values
            data = df[column1].dropna()
            
            # Use 0 as the reference value if not specified
            reference_value = 0
            
            # Calculate Cohen's d
            d = (data.mean() - reference_value) / data.std()
            
            result += f"**Test Type**: One-sample t-test (Cohen's d)\n"
            result += f"**Column**: {column1}\n"
            result += f"**Reference Value**: {reference_value}\n\n"
            
            result += "## Sample Statistics\n\n"
            result += f"- Sample Size: {len(data)}\n"
            result += f"- Sample Mean: {data.mean():.4f}\n"
            result += f"- Sample Standard Deviation: {data.std():.4f}\n\n"
            
            result += "## Effect Size Results\n\n"
            result += f"- Cohen's d: {d:.4f}\n\n"
            
            # Interpretation
            result += "## Interpretation\n\n"
            
            abs_d = abs(d)
            if abs_d < 0.2:
                result += "The effect size is very small.\n"
            elif abs_d < 0.5:
                result += "The effect size is small.\n"
            elif abs_d < 0.8:
                result += "The effect size is medium.\n"
            else:
                result += "The effect size is large.\n"
                
        elif test_type == 'paired':
            # Cohen's d for paired t-test
            
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
            
            # Drop rows with missing values in either column
            valid_data = df[[column1, column2]].dropna()
            
            # Calculate differences
            diff = valid_data[column1] - valid_data[column2]
            
            # Calculate Cohen's d for paired data
            d = diff.mean() / diff.std()
            
            result += f"**Test Type**: Paired t-test (Cohen's d)\n"
            result += f"**Columns**: {column1} vs {column2}\n\n"
            
            result += "## Sample Statistics\n\n"
            result += f"- Sample Size: {len(valid_data)}\n"
            result += f"- Mean of '{column1}': {valid_data[column1].mean():.4f}\n"
            result += f"- Mean of '{column2}': {valid_data[column2].mean():.4f}\n"
            result += f"- Mean Difference: {diff.mean():.4f}\n"
            result += f"- Standard Deviation of Differences: {diff.std():.4f}\n\n"
            
            result += "## Effect Size Results\n\n"
            result += f"- Cohen's d: {d:.4f}\n\n"
            
            # Interpretation
            result += "## Interpretation\n\n"
            
            abs_d = abs(d)
            if abs_d < 0.2:
                result += "The effect size is very small.\n"
            elif abs_d < 0.5:
                result += "The effect size is small.\n"
            elif abs_d < 0.8:
                result += "The effect size is medium.\n"
            else:
                result += "The effect size is large.\n"
                
        elif test_type == 'ind_t':
            # Cohen's d for independent samples t-test
            
            # Check if columns exist
            if column1 not in df.columns:
                return f"Error: Column '{column1}' not found in {filepath}."
            if column2 is None and group_column is None:
                return "Error: Either column2 or group_column must be specified for independent t-test."
            
            if group_column is not None:
                # Using group_column to define groups
                if group_column not in df.columns:
                    return f"Error: Group column '{group_column}' not found in {filepath}."
                
                # Check if value column is numeric
                if not pd.api.types.is_numeric_dtype(df[column1]):
                    return f"Error: Value column '{column1}' is not numeric."
                
                # Get unique groups
                groups = df[group_column].unique()
                
                if len(groups) != 2:
                    return f"Error: Independent t-test requires exactly 2 groups, but found {len(groups)}."
                
                # Get data for each group
                group1_data = df[df[group_column] == groups[0]][column1].dropna()
                group2_data = df[df[group_column] == groups[1]][column1].dropna()
                
                result += f"**Test Type**: Independent samples t-test (Cohen's d)\n"
                result += f"**Value Column**: {column1}\n"
                result += f"**Group Column**: {group_column}\n"
                result += f"**Groups**: {groups[0]} vs {groups[1]}\n"
                result += f"**Pooled Standard Deviation**: {pooled}\n\n"
                
                result += "## Sample Statistics\n\n"
                result += f"- Group '{groups[0]}' Size: {len(group1_data)}\n"
                result += f"- Group '{groups[1]}' Size: {len(group2_data)}\n"
                result += f"- Group '{groups[0]}' Mean: {group1_data.mean():.4f}\n"
                result += f"- Group '{groups[1]}' Mean: {group2_data.mean():.4f}\n"
                result += f"- Mean Difference: {group1_data.mean() - group2_data.mean():.4f}\n"
                result += f"- Group '{groups[0]}' Standard Deviation: {group1_data.std():.4f}\n"
                result += f"- Group '{groups[1]}' Standard Deviation: {group2_data.std():.4f}\n\n"
                
            else:
                # Using two separate columns
                if column2 not in df.columns:
                    return f"Error: Column '{column2}' not found in {filepath}."
                
                # Check if columns are numeric
                if not pd.api.types.is_numeric_dtype(df[column1]):
                    return f"Error: Column '{column1}' is not numeric."
                if not pd.api.types.is_numeric_dtype(df[column2]):
                    return f"Error: Column '{column2}' is not numeric."
                
                # Get data for each group
                group1_data = df[column1].dropna()
                group2_data = df[column2].dropna()
                
                result += f"**Test Type**: Independent samples t-test (Cohen's d)\n"
                result += f"**Columns**: {column1} vs {column2}\n"
                result += f"**Pooled Standard Deviation**: {pooled}\n\n"
                
                result += "## Sample Statistics\n\n"
                result += f"- '{column1}' Size: {len(group1_data)}\n"
                result += f"- '{column2}' Size: {len(group2_data)}\n"
                result += f"- '{column1}' Mean: {group1_data.mean():.4f}\n"
                result += f"- '{column2}' Mean: {group2_data.mean():.4f}\n"
                result += f"- Mean Difference: {group1_data.mean() - group2_data.mean():.4f}\n"
                result += f"- '{column1}' Standard Deviation: {group1_data.std():.4f}\n"
                result += f"- '{column2}' Standard Deviation: {group2_data.std():.4f}\n\n"
            
            # Calculate Cohen's d
            mean_diff = group1_data.mean() - group2_data.mean()
            
            if pooled:
                # Pooled standard deviation
                n1, n2 = len(group1_data), len(group2_data)
                s1, s2 = group1_data.std(), group2_data.std()
                pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                d = mean_diff / pooled_std
            else:
                # Simple average of standard deviations
                d = mean_diff / ((group1_data.std() + group2_data.std()) / 2)
            
            result += "## Effect Size Results\n\n"
            result += f"- Cohen's d: {d:.4f}\n\n"
            
            # Interpretation
            result += "## Interpretation\n\n"
            
            abs_d = abs(d)
            if abs_d < 0.2:
                result += "The effect size is very small.\n"
            elif abs_d < 0.5:
                result += "The effect size is small.\n"
            elif abs_d < 0.8:
                result += "The effect size is medium.\n"
            else:
                result += "The effect size is large.\n"
                
        elif test_type == 'anova':
            # Eta-squared or Cohen's f for one-way ANOVA
            
            # Check if columns exist
            if column1 not in df.columns:
                return f"Error: Value column '{column1}' not found in {filepath}."
            if group_column not in df.columns:
                return f"Error: Group column '{group_column}' not found in {filepath}."
            
            # Check if value column is numeric
            if not pd.api.types.is_numeric_dtype(df[column1]):
                return f"Error: Value column '{column1}' is not numeric."
            
            # Get unique groups
            groups = df[group_column].unique()
            
            if len(groups) < 2:
                return f"Error: Need at least two groups for ANOVA, but found only {len(groups)}."
            
            # Drop rows with missing values
            valid_data = df[[column1, group_column]].dropna()
            
            # Calculate grand mean
            grand_mean = valid_data[column1].mean()
            
            # Split data by groups
            group_data = {}
            group_stats = {}
            
            for group in groups:
                group_data[group] = valid_data[valid_data[group_column] == group][column1].values
                group_stats[group] = {
                    'n': len(group_data[group]),
                    'mean': group_data[group].mean(),
                    'std': group_data[group].std()
                }
            
            # Calculate between-group sum of squares
            ss_between = sum(group_stats[group]['n'] * (group_stats[group]['mean'] - grand_mean)**2 for group in groups)
            
            # Calculate total sum of squares
            ss_total = sum((valid_data[column1] - grand_mean)**2)
            
            # Calculate eta-squared
            eta_squared = ss_between / ss_total
            
            # Calculate Cohen's f
            f = np.sqrt(eta_squared / (1 - eta_squared))
            
            result += f"**Test Type**: One-way ANOVA (Eta-squared and Cohen's f)\n"
            result += f"**Value Column**: {column1}\n"
            result += f"**Group Column**: {group_column}\n\n"
            
            result += "## Sample Statistics\n\n"
            result += f"- Number of Groups: {len(groups)}\n"
            result += f"- Total Sample Size: {len(valid_data)}\n"
            result += f"- Grand Mean: {grand_mean:.4f}\n\n"
            
            result += "| Group | Count | Mean | Std Dev |\n"
            result += "|-------|-------|------|--------|\n"
            
            for group in groups:
                stats = group_stats[group]
                result += f"| {group} | {stats['n']} | {stats['mean']:.4f} | {stats['std']:.4f} |\n"
            
            result += "\n## Effect Size Results\n\n"
            result += f"- Eta-squared (η²): {eta_squared:.4f}\n"
            result += f"- Cohen's f: {f:.4f}\n\n"
            
            # Interpretation
            result += "## Interpretation\n\n"
            
            # Eta-squared interpretation
            result += "**Eta-squared (η²) Interpretation**:\n"
            if eta_squared < 0.01:
                result += "- η² < 0.01: Very small effect\n"
            elif eta_squared < 0.06:
                result += "- 0.01 ≤ η² < 0.06: Small effect\n"
            elif eta_squared < 0.14:
                result += "- 0.06 ≤ η² < 0.14: Medium effect\n"
            else:
                result += "- η² ≥ 0.14: Large effect\n"
            
            # Cohen's f interpretation
            result += "\n**Cohen's f Interpretation**:\n"
            if f < 0.1:
                result += "- f < 0.1: Very small effect\n"
            elif f < 0.25:
                result += "- 0.1 ≤ f < 0.25: Small effect\n"
            elif f < 0.4:
                result += "- 0.25 ≤ f < 0.4: Medium effect\n"
            else:
                result += "- f ≥ 0.4: Large effect\n"
                
        elif test_type == 'correlation':
            # Correlation coefficient (r) as effect size
            
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
            
            # Drop rows with missing values in either column
            valid_data = df[[column1, column2]].dropna()
            
            # Calculate Pearson correlation coefficient
            r, p_value = stats.pearsonr(valid_data[column1], valid_data[column2])
            
            # Calculate coefficient of determination (r²)
            r_squared = r**2
            
            result += f"**Test Type**: Correlation (Pearson's r)\n"
            result += f"**Columns**: {column1} vs {column2}\n\n"
            
            result += "## Sample Statistics\n\n"
            result += f"- Sample Size: {len(valid_data)}\n"
            result += f"- Mean of '{column1}': {valid_data[column1].mean():.4f}\n"
            result += f"- Mean of '{column2}': {valid_data[column2].mean():.4f}\n"
            result += f"- Standard Deviation of '{column1}': {valid_data[column1].std():.4f}\n"
            result += f"- Standard Deviation of '{column2}': {valid_data[column2].std():.4f}\n\n"
            
            result += "## Effect Size Results\n\n"
            result += f"- Correlation Coefficient (r): {r:.4f}\n"
            result += f"- Coefficient of Determination (r²): {r_squared:.4f}\n"
            result += f"- p-value: {p_value:.4f}\n\n"
            
            # Interpretation
            result += "## Interpretation\n\n"
            
            result += "**Correlation Coefficient (r) Interpretation**:\n"
            
            # Strength interpretation
            abs_r = abs(r)
            if abs_r < 0.1:
                strength = "negligible"
            elif abs_r < 0.3:
                strength = "weak"
            elif abs_r < 0.5:
                strength = "moderate"
            elif abs_r < 0.7:
                strength = "moderately strong"
            elif abs_r < 0.9:
                strength = "strong"
            else:
                strength = "very strong"
            
            # Direction
            direction = "positive" if r > 0 else "negative"
            
            if abs_r < 0.1:
                result += f"- The correlation is {strength}.\n"
            else:
                result += f"- The correlation is {strength} and {direction}.\n"
            
            # Coefficient of determination
            result += f"- The coefficient of determination (r² = {r_squared:.4f}) indicates that {r_squared*100:.1f}% of the variance in one variable can be explained by the other variable.\n"
            
            # Statistical significance
            alpha = 0.05  # Standard significance level
            if p_value < alpha:
                result += f"- The correlation is statistically significant (p = {p_value:.4f} < α = {alpha}).\n"
            else:
                result += f"- The correlation is not statistically significant (p = {p_value:.4f} > α = {alpha}).\n"
        
        return result
    
    except Exception as e:
        return f"Error calculating effect size: {str(e)}"