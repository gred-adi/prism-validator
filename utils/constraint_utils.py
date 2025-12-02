import pandas as pd
import streamlit as st
import re

def parse_db_condition(condition_str: str) -> str:
    """
    Standardizes database condition strings to Python operators.
    Example: "Equal To" -> "==", "<" -> "<"
    """
    c = condition_str.lower().strip()
    if "equal" in c or c == "=": return "=="
    if "less" in c or c == "<": return "<"
    if "greater" in c or c == ">": return ">"
    return c

def fetch_model_constraints(db_connection, model_name: str) -> pd.DataFrame:
    """
    Fetches active filter constraints for a specific model from the PRISM database.
    Returns a DataFrame with columns: [Column, Operator, Value]
    """
    if not db_connection:
        return pd.DataFrame()

    # Query to fetch Point Name, Condition, and Value for a specific Project
    # We join Projects -> ProjectPoints -> PointFilters
    query = f"""
    SELECT 
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
            return pd.DataFrame(columns=["Point Name", "Operator", "Value"])
            
        # Parse the results into the format expected by the plotting tool
        parsed_data = []
        for _, row in df.iterrows():
            parsed_data.append({
                "Point Name": row['Point Name'],
                "Operator": parse_db_condition(row['Condition']),
                "Value": float(row['Value'])
            })
            
        return pd.DataFrame(parsed_data)
        
    except Exception as e:
        st.error(f"Error fetching constraints: {e}")
        return pd.DataFrame(columns=["Point Name", "Operator", "Value"])