"""
Code generation and execution tools for the Data Science MCP.
This module contains tools for dynamically generating and executing Python code
for data analysis based on user requests.
"""

from mcp.server.fastmcp import FastMCP, Context, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

# The DATA_DIR will be initialized by the main server module
DATA_DIR = None
# The MCP instance will be initialized by the main server module
mcp = None

# Global dictionary to store generated code
_code_store = {}

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
    mcp_instance.add_tool(generate_analysis_code)
    mcp_instance.add_tool(execute_code)
    mcp_instance.add_tool(get_code)
    mcp_instance.add_tool(save_code)

def generate_analysis_code(request: str, dataset_name: str, ctx: Context) -> str:
    """
    Generate Python code for data analysis based on a natural language request.
    
    Args:
        request: Natural language description of the analysis to perform
        dataset_name: Name of the dataset to analyze
        
    Returns:
        Generated Python code as a string
    """
    # Ensure dataset_name has .csv extension
    if not dataset_name.endswith('.csv'):
        dataset_name += '.csv'
    
    filepath = DATA_DIR / dataset_name
    
    if not filepath.exists():
        return f"Error: Dataset '{dataset_name}' not found."
    
    try:
        # Load the dataset to inspect columns
        df = pd.read_csv(filepath)
        columns = list(df.columns)
        
        # Create code header with imports and loading dataset
        code = f"""# Analysis code for: {request}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the dataset
DATA_DIR = Path("{DATA_DIR}")
df = pd.read_csv(DATA_DIR / "{dataset_name}")

"""
        
        # Generate type-specific code based on the request
        if "correlation" in request.lower() or "relationship between" in request.lower():
            code += f"""# Analyze correlations between numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1, cmap='coolwarm')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()
"""

        elif "distribution" in request.lower() or "histogram" in request.lower():
            code += f"""# Plot histograms for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    plt.hist(df[col].dropna(), bins=20, alpha=0.7)
    plt.title(f'Distribution of {{col}}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
"""

        elif "summary" in request.lower() or "statistics" in request.lower():
            code += f"""# Generate summary statistics
print("Basic Information:")
print(f"Shape: {{df.shape[0]}} rows × {{df.shape[1]}} columns")
print("\\nColumn Data Types:")
print(df.dtypes)

print("\\nSummary Statistics:")
print(df.describe().T)

print("\\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found.")
"""

        elif "outlier" in request.lower():
            code += f"""# Detect outliers in numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"Outliers in {{col}}:")
    print(f"  Count: {{len(outliers)}}")
    print(f"  Percentage: {{len(outliers) / len(df) * 100:.2f}}%")
    print(f"  Range for normal values: [{{lower_bound:.2f}}, {{upper_bound:.2f}}]")
    
    # Plot boxplot to visualize outliers
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[col].dropna())
    plt.title(f'Boxplot for {{col}} showing outliers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
"""

        elif "missing" in request.lower() or "null" in request.lower():
            code += f"""# Analyze missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

print("Missing Values Analysis:")
if missing.sum() > 0:
    print(missing[missing > 0])
    print("\\nMissing Percentages:")
    print(missing_pct[missing > 0])
    
    # Visualize missing values pattern
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(missing[missing > 0])), missing[missing > 0])
    plt.xticks(range(len(missing[missing > 0])), missing[missing > 0].index, rotation=90)
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Count')
    plt.tight_layout()
else:
    print("No missing values found in the dataset.")
"""

        elif "group by" in request.lower() or "aggregation" in request.lower():
            # Try to identify categorical columns for grouping
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                group_col = cat_cols[0]  # Use the first categorical column as an example
                code += f"""# Group by analysis
# Using '{group_col}' as the grouping column - modify as needed
groupby_col = '{group_col}'

# Select numeric columns to aggregate
numeric_cols = df.select_dtypes(include=['number']).columns

# Perform groupby aggregation
grouped = df.groupby(groupby_col)[numeric_cols].agg(['mean', 'median', 'std', 'count'])
print("Groupby Analysis:")
print(grouped)

# Visualize one of the aggregated metrics
if len(numeric_cols) > 0:
    metric_col = numeric_cols[0]  # Use the first numeric column as an example
    plt.figure(figsize=(12, 6))
    df.groupby(groupby_col)[metric_col].mean().sort_values().plot(kind='bar')
    plt.title(f'Mean {metric_col} by {group_col}')
    plt.ylabel(f'Mean {metric_col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
"""
            else:
                code += f"""# Note: No categorical columns found for groupby analysis
# You may need to create categories from numeric data

# Example: Creating bins from a numeric column
if len(df.select_dtypes(include=['number']).columns) > 0:
    numeric_col = df.select_dtypes(include=['number']).columns[0]
    df['bins'] = pd.cut(df[numeric_col], bins=5)
    
    # Select another numeric column to aggregate
    other_numeric_cols = [col for col in df.select_dtypes(include=['number']).columns if col != numeric_col]
    if other_numeric_cols:
        agg_col = other_numeric_cols[0]
        grouped = df.groupby('bins')[agg_col].agg(['mean', 'median', 'std', 'count'])
        print("Groupby Analysis:")
        print(grouped)
    else:
        print("Not enough numeric columns for meaningful groupby analysis")
else:
    print("No numeric columns found for groupby analysis")
"""

        elif "regression" in request.lower() or "predict" in request.lower():
            # Generate regression analysis code
            code += f"""# Regression analysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Identify potential target variable (last numeric column by default)
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) < 2:
    print("Need at least 2 numeric columns for regression analysis")
else:
    # Use the last numeric column as the target by default
    target_col = numeric_cols[-1]
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    print(f"Performing regression analysis to predict {{target_col}}")
    print(f"Using features: {{feature_cols}}")
    
    # Prepare data
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\\nModel Results:")
    print(f"Mean Squared Error: {{mse:.4f}}")
    print(f"R² Score: {{r2:.4f}}")
    
    # Print coefficients
    print("\\nCoefficients:")
    for feature, coef in zip(feature_cols, model.coef_):
        print(f"{{feature}}: {{coef:.4f}}")
    print(f"Intercept: {{model.intercept_:.4f}}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs. Predicted {{target_col}}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
"""

        else:
            # Default to a general exploratory analysis
            code += f"""# General exploratory analysis
print("Dataset Overview:")
print(df.head())

print("\\nData Types:")
print(df.dtypes)

print("\\nSummary Statistics:")
print(df.describe())

# Plot a few key visualizations
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) >= 2:
    # Scatter plot of first two numeric columns
    plt.figure(figsize=(10, 6))
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5)
    plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

if len(numeric_cols) > 0:
    # Histogram of first numeric column
    plt.figure(figsize=(10, 6))
    plt.hist(df[numeric_cols[0]].dropna(), bins=20, alpha=0.7)
    plt.title(f'Distribution of {numeric_cols[0]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
"""

        # Add code to handle matplotlib output
        code += """
# Show all plots
plt.show()
"""
        
        # Store the generated code with a unique ID
        code_id = f"analysis_{len(_code_store) + 1}"
        _code_store[code_id] = code
        
        return f"Generated analysis code (ID: {code_id}):\n\n```python\n{code}\n```\n\nYou can execute this code using the `execute_code` tool with the code_id."
    
    except Exception as e:
        return f"Error generating analysis code: {str(e)}"


def execute_code(code_id: str = None, code: str = None, ctx: Context = None) -> str:
    """
    Execute Python code and return the results.
    
    Args:
        code_id: ID of stored code to execute
        code: Raw Python code to execute (alternative to code_id)
        
    Returns:
        Results of code execution
    """
    if code_id is None and code is None:
        return "Error: Either code_id or code must be provided."
    
    # Get the code to execute
    if code_id is not None:
        if code_id not in _code_store:
            return f"Error: Code with ID '{code_id}' not found."
        code_to_execute = _code_store[code_id]
    else:
        code_to_execute = code
    
    # Execute the code
    try:
        # Redirect stdout and stderr to capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        # Create a local namespace with access to necessary libraries
        local_namespace = {
            'pd': pd, 
            'np': np, 
            'plt': plt,
            'Path': Path,
            'DATA_DIR': DATA_DIR
        }
        
        # *** NEW CODE: Fix file path references ***
        import re
        
        # Patterns to detect and fix
        fix_patterns = [
            (r"pd\.read_csv\(['\"]([^/\\][^'\"]+\.csv)['\"]", r"pd.read_csv(DATA_DIR / '\1'"),
            (r"pd\.read_excel\(['\"]([^/\\][^'\"]+\.xlsx)['\"]", r"pd.read_excel(DATA_DIR / '\1'"),
            (r"pd\.read_parquet\(['\"]([^/\\][^'\"]+\.parquet)['\"]", r"pd.read_parquet(DATA_DIR / '\1'"),
            (r"open\(['\"]([^/\\][^'\"]+\.[^'\"]+)['\"]", r"open(DATA_DIR / '\1'"),
            
            (r"plt\.savefig\(['\"]([^/\\][^'\"]+\.[^'\"]+)['\"]", r"plt.savefig(DATA_DIR / '\1'"),
            (r"\.to_csv\(['\"]([^/\\][^'\"]+\.csv)['\"]", r".to_csv(DATA_DIR / '\1'"),
            (r"\.to_excel\(['\"]([^/\\][^'\"]+\.xlsx)['\"]", r".to_excel(DATA_DIR / '\1'"),
            (r"\.to_parquet\(['\"]([^/\\][^'\"]+\.parquet)['\"]", r".to_parquet(DATA_DIR / '\1'"),
            (r"np\.save\(['\"]([^/\\][^'\"]+\.npy)['\"]", r"np.save(DATA_DIR / '\1'"),
            (r"joblib\.dump\(.*?,\s*['\"]([^/\\][^'\"]+\.[^'\"]+)['\"]", r"joblib.dump(\1, DATA_DIR / '\2'"),
            (r"pickle\.dump\(.*?,\s*open\(['\"]([^/\\][^'\"]+\.[^'\"]+)['\"]", r"pickle.dump(\1, open(DATA_DIR / '\2'")
        ]
        
        modified_code = code_to_execute
        for pattern, replacement in fix_patterns:
            modified_code = re.sub(pattern, replacement, modified_code)
        
        # Log the code transformations for debugging
        print("### Code Transformation Applied ###")
        if modified_code != code_to_execute:
            print("Original file references were automatically corrected to use DATA_DIR.")
        # *** END NEW CODE ***
        
        # Replace plt.show() with code to save figures
        modified_code = modified_code.replace("plt.show()", "")
        
        # Add code to save figures
        modified_code += """
# Save all figures to files
import matplotlib.pyplot as plt
figure_paths = []
for i, fig in enumerate(plt.get_fignums()):
    figure = plt.figure(fig)
    path = f"figure_{i+1}.png"
    figure.savefig(path)
    figure_paths.append(path)
print("\\nGenerated figures:")
for path in figure_paths:
    print(f"- {path}")
"""
        
        # Execute the modified code
        exec(modified_code, local_namespace)
        
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # Get output
        output = stdout_buffer.getvalue()
        errors = stderr_buffer.getvalue()
        
        # Check if there are any figures generated
        figure_paths = local_namespace.get('figure_paths', [])
        figure_output = ""
        
        if figure_paths:
            figure_output = "\n\nGenerated figures:\n"
            for path in figure_paths:
                figure_output += f"- {path}\n"
        
        if errors:
            return f"Code executed with errors:\n\n{errors}\n\nOutput:\n{output}{figure_output}"
        else:
            return f"Code executed successfully.\n\nOutput:\n{output}{figure_output}"
    
    except Exception as e:
        # Restore stdout and stderr in case of exception
        sys.stdout = sys.stdout if 'old_stdout' not in locals() else old_stdout
        sys.stderr = sys.stderr if 'old_stderr' not in locals() else old_stderr
        
        # Get the traceback
        error_traceback = traceback.format_exc()
        return f"Error executing code:\n\n{error_traceback}"


def get_code(code_id: str, ctx: Context) -> str:
    """
    Retrieve previously generated code by ID.
    
    Args:
        code_id: ID of the code to retrieve
        
    Returns:
        The code as a string
    """
    if code_id not in _code_store:
        return f"Error: Code with ID '{code_id}' not found."
    
    code = _code_store[code_id]
    return f"Code (ID: {code_id}):\n\n```python\n{code}\n```"


def save_code(code_id: str, filename: str, ctx: Context) -> str:
    """
    Save generated code to a file.
    
    Args:
        code_id: ID of the code to save
        filename: Name of the file to save the code to
        
    Returns:
        Confirmation message
    """
    if code_id not in _code_store:
        return f"Error: Code with ID '{code_id}' not found."
    
    code = _code_store[code_id]
    
    # Ensure filename has .py extension
    if not filename.endswith('.py'):
        filename += '.py'
    
    filepath = DATA_DIR / filename
    
    try:
        with open(filepath, 'w') as f:
            f.write(code)
        
        return f"Code successfully saved to {filepath}"
    
    except Exception as e:
        return f"Error saving code to file: {str(e)}"