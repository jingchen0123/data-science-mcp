"""
Prompt templates for the Data Science MCP.
This module contains reusable prompts and conversation flows for common data science tasks.
"""

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base, Prompt
from typing import List, Dict, Any, Union

# The MCP instance will be initialized by the main server module
mcp = None

def analyze_dataset(dataset_name: str) -> str:
    """
    Create a prompt for comprehensive dataset analysis.
    
    Args:
        dataset_name: Name of the dataset to analyze
    """
    return f"""I would like you to help me analyze the dataset '{dataset_name}'. 
Please help me understand:

1. The basic structure and content of the dataset
2. Key statistics and distributions
3. Interesting patterns, correlations, or anomalies
4. Potential insights and next steps for analysis

You can use tools like describe_dataset, explore_data, get_columns_info, plot_histogram, 
and plot_scatter to explore the data. Feel free to suggest specific analyses that would
be valuable for this dataset."""


def explore_relationship(dataset_name: str, variable1: str, variable2: str) -> List[base.Message]:
    """
    Create a guided conversation to explore the relationship between two variables.
    
    Args:
        dataset_name: Name of the dataset
        variable1: First variable name
        variable2: Second variable name
    """
    return [
        base.UserMessage(f"I want to understand the relationship between {variable1} and {variable2} in my dataset {dataset_name}."),
        base.AssistantMessage(f"I'll help you explore the relationship between {variable1} and {variable2}. Let me first check the correlation between them."),
        base.UserMessage("Please also create a scatter plot and explain what you observe.")
    ]


def data_science_assistant() -> str:
    """
    Main prompt template for general data science assistance.
    """
    return """I'm your data science assistant. I can help you with various data analysis tasks:

1. **Data Exploration**: Understand your dataset's structure, distributions, and patterns
2. **Data Quality Assessment**: Identify issues with missing values, outliers, or inconsistencies
3. **Visualization**: Create insightful plots and charts to reveal patterns
4. **Statistical Analysis**: Calculate correlations, run statistical tests, and interpret results
5. **Feature Engineering**: Develop new features to improve analysis and modeling
6. **Data Cleaning**: Suggest and implement data cleaning techniques

What would you like to do with your data today? If you have a specific dataset in mind, 
please let me know its name (e.g., "customer_data.csv") so I can help you analyze it effectively."""


def data_cleaning_workflow(dataset_name: str) -> List[base.Message]:
    """
    Create a guided conversation for data cleaning.
    
    Args:
        dataset_name: Name of the dataset to clean
    """
    return [
        base.UserMessage(f"I need to clean the dataset {dataset_name} before analysis. Please help me with this process."),
        
        base.AssistantMessage(f"""I'll help you clean the dataset {dataset_name}. Let's start with a thorough data quality assessment to identify issues that need addressing.

I'll use the `check_data_quality` tool to get a comprehensive assessment."""),
        
        base.UserMessage("Thanks for the quality assessment. What specific cleaning steps do you recommend based on these issues?"),
        
        base.AssistantMessage("""Based on the quality assessment, here are recommended cleaning steps we can take:

1. Handle missing values
2. Address potential outliers
3. Fix consistency issues
4. Correct invalid data
5. Remove duplicate records

Would you like me to help implement any of these steps? We can take them one by one."""),
        
        base.UserMessage("Yes, let's start with handling the missing values. What approaches do you recommend?")
    ]


def feature_engineering_guide(dataset_name: str) -> str:
    """
    Create a prompt for feature engineering guidance.
    
    Args:
        dataset_name: Name of the dataset
    """
    return f"""I'd like to perform feature engineering on my dataset {dataset_name} to improve its analytical value.

Please help me with:

1. Identifying which features could benefit from transformation or encoding
2. Creating interaction terms between relevant features
3. Developing new features that might capture important patterns
4. Normalizing or scaling features as needed
5. Reducing dimensionality if appropriate

Before suggesting specific feature engineering steps, please analyze the current state
of the dataset to understand what would be most valuable."""


def explain_correlation(dataset_name: str, variable1: str, variable2: str) -> str:
    """
    Create a prompt for explaining correlations between variables.
    
    Args:
        dataset_name: Name of the dataset
        variable1: First variable name
        variable2: Second variable name
    """
    return f"""I've noticed a correlation between {variable1} and {variable2} in my dataset {dataset_name}.

Could you help me understand:

1. The strength and direction of this correlation
2. Whether this correlation implies causation
3. Potential confounding variables that might explain this relationship
4. How to visualize this relationship effectively
5. Any statistical tests I should run to further investigate this relationship

Please use appropriate tools to calculate and visualize the correlation."""


def interpret_visualization(visualization_type: str, variables: List[str], dataset_name: str) -> str:
    """
    Create a prompt for interpreting a visualization.
    
    Args:
        visualization_type: Type of visualization (e.g., histogram, scatter plot)
        variables: Variables included in the visualization
        dataset_name: Name of the dataset
    """
    variables_str = ", ".join(variables)
    
    return f"""I've created a {visualization_type} of {variables_str} from my dataset {dataset_name}.

Could you help me interpret what this visualization shows? Specifically:

1. What are the key patterns or trends visible in this {visualization_type}?
2. Are there any outliers or anomalies I should pay attention to?
3. What insights can I draw about {variables_str} from this visualization?
4. What additional analyses might complement this visualization?
5. How might I improve this visualization to make it more informative?

Please generate the {visualization_type} first using the appropriate tool so you can provide a specific interpretation."""


def statistical_test_advisor(dataset_name: str) -> str:
    """
    Create a prompt for recommending appropriate statistical tests.
    
    Args:
        dataset_name: Name of the dataset
    """
    return f"""I'm working with the dataset {dataset_name} and would like advice on which statistical tests would be appropriate for my analysis.

Please help me understand:

1. Which statistical tests would be suitable given the data types and distributions in my dataset
2. The assumptions of each recommended test
3. How to interpret the results of these tests
4. How to implement these tests using the available tools
5. Common pitfalls to avoid when running these tests

Before recommending specific tests, please analyze the dataset to understand its characteristics."""


def modeling_workflow(dataset_name: str, target_variable: str) -> List[base.Message]:
    """
    Create a guided conversation for a modeling workflow.
    
    Args:
        dataset_name: Name of the dataset
        target_variable: Target variable for modeling
    """
    return [
        base.UserMessage(f"I want to build a predictive model for {target_variable} using my dataset {dataset_name}. Please guide me through the process."),
        
        base.AssistantMessage(f"""I'll help you build a predictive model for {target_variable} using {dataset_name}. Let's start by exploring the dataset and understanding the target variable.

First, I'll use the `explore_data` tool to get a comprehensive overview of the dataset."""),
        
        base.UserMessage("Thanks for the exploration. What preprocessing steps should I take before modeling?"),
        
        base.AssistantMessage(f"""Based on the exploration of {dataset_name}, here are the preprocessing steps I recommend before modeling {target_variable}:

1. Handle missing values in the dataset
2. Encode categorical variables appropriately
3. Scale or normalize numeric features
4. Handle outliers in the data
5. Create any useful new features
6. Split the data into training and validation sets

Would you like me to help implement any of these preprocessing steps?"""),
        
        base.UserMessage("Yes, please help with these preprocessing steps. After that, what modeling approach would you recommend?")
    ]


def initialize(mcp_instance: FastMCP):
    """
    Initialize this module with the MCP instance.
    
    Args:
        mcp_instance: The FastMCP instance
    """
    global mcp
    mcp = mcp_instance
    
    # Register all prompts with the MCP instance
    # Create proper Prompt objects including argument definitions
    mcp_instance.add_prompt(Prompt(
        name="analyze_dataset",
        fn=analyze_dataset,
        description="Create a prompt for comprehensive dataset analysis",
        arguments=[{
            "name": "dataset_name",
            "description": "Name of the dataset to analyze",
            "required": True
        }]
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="explore_relationship",
        fn=explore_relationship,
        description="Create a guided conversation to explore the relationship between two variables",
        arguments=[
            {
                "name": "dataset_name",
                "description": "Name of the dataset",
                "required": True
            },
            {
                "name": "variable1",
                "description": "First variable name",
                "required": True
            },
            {
                "name": "variable2",
                "description": "Second variable name",
                "required": True
            }
        ]
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="data_science_assistant",
        fn=data_science_assistant,
        description="Main prompt template for general data science assistance"
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="data_cleaning_workflow",
        fn=data_cleaning_workflow,
        description="Create a guided conversation for data cleaning",
        arguments=[{
            "name": "dataset_name",
            "description": "Name of the dataset to clean",
            "required": True
        }]
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="feature_engineering_guide",
        fn=feature_engineering_guide,
        description="Create a prompt for feature engineering guidance",
        arguments=[{
            "name": "dataset_name",
            "description": "Name of the dataset",
            "required": True
        }]
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="explain_correlation",
        fn=explain_correlation,
        description="Create a prompt for explaining correlations between variables",
        arguments=[
            {
                "name": "dataset_name",
                "description": "Name of the dataset",
                "required": True
            },
            {
                "name": "variable1",
                "description": "First variable name",
                "required": True
            },
            {
                "name": "variable2",
                "description": "Second variable name",
                "required": True
            }
        ]
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="interpret_visualization",
        fn=interpret_visualization,
        description="Create a prompt for interpreting a visualization",
        arguments=[
            {
                "name": "visualization_type",
                "description": "Type of visualization (e.g., histogram, scatter plot)",
                "required": True
            },
            {
                "name": "variables",
                "description": "Variables included in the visualization",
                "required": True
            },
            {
                "name": "dataset_name",
                "description": "Name of the dataset",
                "required": True
            }
        ]
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="statistical_test_advisor",
        fn=statistical_test_advisor,
        description="Create a prompt for recommending appropriate statistical tests",
        arguments=[{
            "name": "dataset_name",
            "description": "Name of the dataset",
            "required": True
        }]
    ))
    
    mcp_instance.add_prompt(Prompt(
        name="modeling_workflow",
        fn=modeling_workflow,
        description="Create a guided conversation for a modeling workflow",
        arguments=[
            {
                "name": "dataset_name",
                "description": "Name of the dataset",
                "required": True
            },
            {
                "name": "target_variable",
                "description": "Target variable for modeling",
                "required": True
            }
        ]
    ))