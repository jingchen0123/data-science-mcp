# Data Science MCP Agent

A comprehensive Model Context Protocol (MCP) server for data science workflows, providing tools for data loading, exploration, visualization, processing, statistical analysis, and code generation.

## Features

### 🔍 Data Exploration & Analysis
- Load and inspect CSV datasets
- Comprehensive data quality assessment
- Exploratory data analysis with statistics and insights
- Column-specific analysis and profiling
- Missing value and outlier detection

### 📊 Data Visualization  
- Create scatter plots, histograms, bar charts, and line plots
- Generate correlation matrices and box plots
- Customizable plot titles and styling

### 🔧 Data Processing
- Filter datasets with complex conditions
- Transform columns (log, sqrt, standardize, normalize, etc.)
- Group and aggregate data with multiple functions
- Handle missing values with various imputation methods

### 📈 Statistical Analysis
- Correlation analysis between variables
- One-sample and paired t-tests
- ANOVA for group comparisons
- Chi-square tests for categorical associations
- Normality and homogeneity testing
- Power analysis and effect size calculations

### 💻 Code Generation & Execution
- Generate Python analysis code from natural language descriptions
- Execute generated code safely in controlled environment
- Store and retrieve code snippets
- Automatic file path resolution

### 🎯 Guided Workflows
- Pre-built prompt templates for common tasks
- Step-by-step data cleaning workflows
- Feature engineering guidance
- Statistical test recommendations

## Installation

### Prerequisites
- Python 3.8 or higher
- `uv` package manager (recommended) or `pip`

### Setup

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd data-science-mcp
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv add "mcp[cli]" pandas numpy matplotlib seaborn scikit-learn scipy

   # Or using pip
   pip install "mcp[cli]" pandas numpy matplotlib seaborn scikit-learn scipy
   ```

3. **Configure data directory**
   Edit the `DATA_DIR` path in `server.py`:
   ```python
   DATA_DIR = Path("/path/to/your/data/directory")
   ```

4. **Run the server**
   ```bash
   python server.py
   ```

## Configuration for Claude Desktop

Add this configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "data-science": {
      "command": "python",
      "args": ["/absolute/path/to/data-science-mcp/server.py"],
      "env": {}
    }
  }
}
```

**Configuration file locations:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

## Tools Reference

### Data Loading & Management
- `load_csv(file_path)` - Load CSV files into the system
- `list_datasets()` - List all available datasets
- `get_dataset_info(file_path)` - Get detailed dataset information
- `save_csv(data, file_path)` - Save data to CSV format
- `upload_csv(content, file_path)` - Upload CSV content as string

### Data Exploration
- `explore_data(file_path, sample_rows=5)` - Comprehensive data exploration
- `describe_dataset(file_path)` - Generate descriptive statistics
- `get_columns_info(file_path, columns=None)` - Detailed column analysis
- `detect_outliers(file_path, columns=None, method="iqr")` - Outlier detection
- `check_data_quality(file_path)` - Complete data quality assessment

### Data Visualization
- `plot_scatter(file_path, x_column, y_column, color_column=None)` - Scatter plots
- `plot_histogram(file_path, column, bins=20, kde=False)` - Histograms
- `plot_bar(file_path, x_column, y_column=None, aggfunc="count")` - Bar charts
- `plot_line(file_path, x_column, y_columns)` - Line plots
- `plot_correlation_matrix(file_path, columns=None)` - Correlation heatmaps
- `plot_box(file_path, columns, by_column=None)` - Box plots
- `get_correlation(file_path, column1, column2)` - Correlation analysis

### Data Processing
- `filter_data(file_path, condition, output_file_path=None)` - Filter datasets
- `transform_column(file_path, column, transformation, new_column=None)` - Transform columns
- `group_and_aggregate(file_path, group_by, aggregate_cols, aggregate_funcs)` - Group and aggregate
- `handle_missing_values(file_path, columns=None, method="mean")` - Handle missing data

### Statistical Analysis
- `run_ttest(file_path, column, test_value=0)` - One-sample t-test
- `run_paired_ttest(file_path, column1, column2)` - Paired t-test
- `run_anova(file_path, value_column, group_column)` - One-way ANOVA
- `run_chi_square(file_path, column1, column2)` - Chi-square test
- `run_correlation_test(file_path, column1, column2, method="pearson")` - Correlation test
- `run_regression(file_path, dependent_var, independent_vars)` - Linear regression
- `check_normality(file_path, column, test="shapiro")` - Normality testing
- `check_homogeneity(file_path, value_column, group_column)` - Homogeneity testing
- `power_analysis(test_type, effect_size, alpha=0.05, power=0.8)` - Power analysis
- `effect_size(file_path, column1, column2=None, test_type="mean_diff")` - Effect size calculation

### Code Generation & Execution
- `generate_analysis_code(request, dataset_path)` - Generate Python code from descriptions
- `execute_code(code_id=None, code=None)` - Execute Python code
- `get_code(code_id)` - Retrieve stored code
- `save_code(code_id, file_path)` - Save code to file

## Resources

The server exposes CSV data through the resource system:
- `csv://{file_path}` - Access CSV file content with preview and statistics

## Prompt Templates

Pre-built prompts for common workflows:
- `analyze_dataset` - Comprehensive dataset analysis
- `explore_relationship` - Guided variable relationship exploration
- `data_science_assistant` - General assistance
- `data_cleaning_workflow` - Step-by-step data cleaning
- `feature_engineering_guide` - Feature engineering guidance
- `explain_correlation` - Correlation explanation
- `interpret_visualization` - Visualization interpretation
- `statistical_test_advisor` - Statistical test recommendations
- `modeling_workflow` - Guided modeling workflow

## Usage Examples

### Basic Data Analysis
```
1. Load your dataset: "Please load the file customer_data.csv"
2. Explore the data: "Can you explore this dataset and tell me what you find?"
3. Visualize relationships: "Create a scatter plot of age vs income"
4. Generate insights: "What correlations do you see in this data?"
```

### Advanced Workflows
```
1. Data quality check: "Please assess the quality of my dataset"
2. Handle missing values: "How should I handle the missing data?"
3. Statistical analysis: "Test if there's a significant difference between groups"
4. Generate report: "Create a comprehensive analysis report"
```

### Code Generation
```
1. "Generate code to analyze the correlation between all numeric variables"
2. "Create a regression model predicting sales from the other variables"
3. "Write code to detect and visualize outliers in the dataset"
```

## File Path Handling

The server supports both absolute and relative paths:
- **Relative paths**: `"data.csv"` → resolves to `{DATA_DIR}/data.csv`
- **Absolute paths**: `"/full/path/to/data.csv"` → used as-is
- **Auto-extension**: `"data"` → automatically becomes `"data.csv"`

## Error Handling

- Comprehensive input validation
- Clear error messages with suggestions
- Graceful handling of missing files or invalid data
- Safe code execution with proper error reporting

## Data Directory

By default, the server uses a data directory for file operations:
- **Location**: Configurable in `server.py`
- **Purpose**: Centralized data storage and file resolution
- **Sample data**: Example dataset created automatically

## Dependencies

- **Core**: `mcp`, `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Statistics**: `scipy`, `scikit-learn`
- **System**: `pathlib`, `os`, `io`

## Architecture

The server is organized into modular components:
- `data_loading.py` - Data management and loading
- `exploration.py` - Data exploration and quality assessment
- `visualization.py` - Plotting and visualization tools
- `processing.py` - Data transformation and manipulation
- `statistical_tests.py` - Statistical analysis functions
- `code_generation.py` - Dynamic code generation and execution
- `templates.py` - Prompt templates for guided workflows

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new tools or improve existing ones
4. Test with various datasets
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or feature requests:
1. Check the error messages for specific guidance
2. Ensure your data files are accessible and properly formatted
3. Verify your Python environment has all required dependencies
4. Review the tool documentation for proper usage

## Changelog

### Version 1.0.0
- Initial release with comprehensive data science toolkit
- Full MCP protocol compliance
- Automatic output capture and reporting
- Guided workflow templates
- Statistical analysis suite