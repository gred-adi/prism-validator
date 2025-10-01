def get_query():
    """
    Constructs and returns the SQL query for failure diagnostics validation.

    This query retrieves the configuration of failure diagnostics from the
    PRISM database. It joins tables to link templates (projects), fault
    diagnostics, and their detailed signatures, including the metric,
    direction, and priority weight.

    A key feature of this query is the use of the `NCHAR()` function to
    translate a numerical direction value (`updownvalue`) into Unicode arrow
    characters (e.g., ↑, ↓, →). This is crucial for matching against the
    TDT reference data.

    The query filters for project names that start with 'AP-'.

    Returns:
        str: A multi-line string containing the complete SQL query for
             failure diagnostics.
    """
    return """
    SELECT
        TEMPLATE_FAULT.TemplateID AS [FORM ID],
        p.Name AS [FORM NAME],
        TEMPLATE_FAULT_DETAIL.PointTypeMetricID AS [METRIC ID],
        ptm.Description AS [METRIC NAME],
        TEMPLATE_FAULT.Description AS [FAILURE MODE],
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
        p.Name LIKE 'AP-%'
    ORDER BY
        p.Name,
        TEMPLATE_FAULT.Description,
        ptm.Description;
    """