import pandas as pd
import streamlit as st
import re

def parse_db_condition(condition_str: str) -> str:
    """Standardizes database condition strings to Python operators.

    This function takes a condition string from the PRISM database (e.g.,
    "Equal To", "<") and converts it into a standardized Python comparison
    operator (e.g., "==", "<").

    Args:
        condition_str (str): The condition string from the database.

    Returns:
        str: The standardized Python operator.
    """
    c = condition_str.lower().strip()
    if "equal" in c or c == "=": return "=="
    if "less" in c or c == "<": return "<"
    if "greater" in c or c == ">": return ">"
    return c

def fetch_model_constraints(db_connection, model_name: str) -> pd.DataFrame:
    """Fetches active filter constraints for a model from the PRISM database.

    This function queries the PRISM database to retrieve the active filter
    constraints for a given model. It returns a structured DataFrame containing
    the metric name, point name, operator, and value for each constraint.

    Args:
        db_connection: An active database connection object.
        model_name (str): The name of the model to fetch constraints for.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Metric Name', 'Point Name',
        'Operator', 'Value'], or an empty DataFrame if no constraints are found
        or an error occurs.
    """
    if not db_connection:
        return pd.DataFrame()

    # Query to fetch Metric Name, Point Name, Condition, and Value for a specific Project
    # We join Projects -> ProjectPoints -> PointFilters
    query = f"""
    SELECT 
        TM.Description AS [Metric Name],
        PP.Name AS [Point Name],
        PF.PointCondition AS [Condition],
        PF.PointValue AS [Value]
    FROM prismdb.dbo.Projects T
    LEFT JOIN prismdb.dbo.Projects P ON P.ParentTemplateID = T.ProjectID
    LEFT JOIN prismdb.dbo.ProjectPoints TP ON T.ProjectID = TP.ProjectID
    INNER JOIN prismdb.dbo.PointFilters PF ON TP.ProjectPointID = PF.PROJECTPOINTID
    INNER JOIN prismdb.dbo.PointTypeMetric TM ON TP.PointTypeMetricID = TM.PointTypeMetricID 
    INNER JOIN prismdb.dbo.ProjectPoints PP ON PP.PointTypeMetricID = TM.PointTypeMetricID AND PP.ProjectID = P.ProjectID
    WHERE 
        P.Name = '{model_name}'
        AND PF.FilterActive = 1
        AND T.ProjectTypeID = 2 
        AND PP.PointTypeID = 1
    """
    
    try:
        df = db_connection.run_query(query)
        
        if df.empty:
            return pd.DataFrame(columns=["Metric Name", "Point Name", "Operator", "Value"])
            
        # Parse the results into the format expected by the plotting tool
        parsed_data = []
        for _, row in df.iterrows():
            parsed_data.append({
                "Metric Name": row['Metric Name'],
                "Point Name": row['Point Name'],
                "Operator": parse_db_condition(row['Condition']),
                "Value": float(row['Value'])
            })
            
        return pd.DataFrame(parsed_data)
        
    except Exception as e:
        st.error(f"Error fetching constraints: {e}")
        return pd.DataFrame(columns=["Metric Name", "Point Name", "Operator", "Value"])