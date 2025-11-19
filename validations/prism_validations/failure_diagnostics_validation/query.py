def get_query(tdt_names=None):
    """
    Returns the SQL query for Failure Diagnostics Validation.
    Optionally filters by a list of TDT names to optimize performance.
    """
    # Base filtering condition
    where_clause = "p.Name LIKE 'AP-%'"
    
    # Add dynamic TDT filtering if names are provided
    if tdt_names:
        # Escape single quotes in names just in case, though TDT names are usually safe
        sanitized_names = [name.replace("'", "''") for name in tdt_names]
        formatted_names = ", ".join([f"'{name}'" for name in sanitized_names])
        where_clause += f" AND p.Name IN ({formatted_names})"

    return f"""
    SELECT
        TEMPLATE_FAULT.TemplateID AS [FORM ID],
        p.Name AS [FORM NAME],
        TEMPLATE_FAULT_DETAIL.PointTypeMetricID AS [METRIC ID],
        ptm.Description AS [METRIC NAME],
        TEMPLATE_FAULT.Description AS [FAILURE MODE],
        TEMPLATE_FAULT.Notes AS [FAILURE DESCRIPTION],
        TEMPLATE_FAULT.NextSteps AS [NEXT STEPS],
        -- Inlined logic from SINE_FAULT_DIAGNOSTIC view for DIRECTION
        CASE
            WHEN TEMPLATE_FAULT_DETAIL.updownvalue = 1 THEN NCHAR(8593) -- ↑
            WHEN TEMPLATE_FAULT_DETAIL.updownvalue = -1 THEN NCHAR(8595) -- ↓
            WHEN TEMPLATE_FAULT_DETAIL.updownvalue = 0 THEN NCHAR(8594) -- →
            WHEN TEMPLATE_FAULT_DETAIL.updownvalue = 2 THEN NCHAR(8593)+NCHAR(8595) -- ↕
            ELSE NULL
        END AS [DIRECTION],
        TEMPLATE_FAULT_DETAIL.PriorityWeight AS [WEIGHT]
    FROM
        -- Inlined logic from SINE_FAULT_DIAGNOSTIC view
        prismdb.dbo.FaultDiagnostic TEMPLATE_FAULT
        LEFT JOIN prismdb.dbo.FaultSignatureDev TEMPLATE_FAULT_DETAIL ON TEMPLATE_FAULT.FaultDiagnosticID = TEMPLATE_FAULT_DETAIL.FaultDiagnosticID
        -- Original Joins from your query
        INNER JOIN prismdb.dbo.Projects p ON TEMPLATE_FAULT.TemplateID = p.ProjectID
        INNER JOIN prismdb.dbo.PointTypeMetric ptm ON TEMPLATE_FAULT_DETAIL.PointTypeMetricID = ptm.PointTypeMetricID
    WHERE
        {where_clause}
    ORDER BY
        p.Name,
        TEMPLATE_FAULT.Description,
        ptm.Description;
    """