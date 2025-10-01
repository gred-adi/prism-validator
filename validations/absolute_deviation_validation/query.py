def get_query():
    """
    Constructs and returns the SQL query for absolute deviation validation.

    This query retrieves threshold data for project-level models from the PRISM
    database. It specifically fetches high/low alert and warning thresholds
    for points configured with 'Absolute Signal Deviation'. The query joins
    several tables to link projects, points, metrics, and their corresponding
    threshold values.

    The query filters for:
    - Projects of type 'PROJECT' (ProjectTypeID = 0).
    - Points with a threshold type of 'Absolute Signal Deviation' (PointTypeID = 8).
    - Deployed projects (DeployedProfileID IS NOT NULL).
    - Project names starting with 'AP-'.
    - Metrics that have at least a 'HIGH WARNING' threshold defined.

    Returns:
        str: A multi-line string containing the complete SQL query.
    """
    return """
    SELECT
        FORM.ProjectID AS [FORM ID],
        FORM.Name AS [FORM NAME],
        FORM_METRIC.PointTypeMetricID AS [METRIC ID],
        FORM_METRIC.Description AS [METRIC NAME],
        FORM_POINTS.name AS [POINT NAME],
        MAX(CASE WHEN FORM_POINTS_DETAIL_THRES.AlarmThresholdTypeID = 2 THEN FORM_POINTS_DETAIL_THRES.ThresholdValue END) AS [HIGH ALERT],
        MAX(CASE WHEN FORM_POINTS_DETAIL_THRES.AlarmThresholdTypeID = 3 THEN FORM_POINTS_DETAIL_THRES.ThresholdValue END) AS [HIGH WARNING],
        MIN(CASE WHEN FORM_POINTS_DETAIL_THRES.AlarmThresholdTypeID = 6 THEN FORM_POINTS_DETAIL_THRES.ThresholdValue END) AS [LOW WARNING],
        MIN(CASE WHEN FORM_POINTS_DETAIL_THRES.AlarmThresholdTypeID = 5 THEN FORM_POINTS_DETAIL_THRES.ThresholdValue END) AS [LOW ALERT]
    FROM
        prismdb.dbo.Projects FORM
        LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS ON FORM.ProjectID = FORM_POINTS.ProjectID AND FORM_POINTS.PointTypeID IN (1, 2)
        LEFT JOIN prismdb.dbo.PointTypeMetric FORM_METRIC ON FORM_POINTS.PointTypeMetricID = FORM_METRIC.PointTypeMetricID
        LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS_DETAIL ON FORM_POINTS.ProjectID = FORM_POINTS_DETAIL.ProjectID
            AND FORM_POINTS.OrderIndex = FORM_POINTS_DETAIL.OrderIndex
        LEFT JOIN prismdb.dbo.AlarmThresholds FORM_POINTS_DETAIL_THRES ON FORM_POINTS_DETAIL.ProjectPointID = FORM_POINTS_DETAIL_THRES.ProjectPointID
    WHERE
        FORM.PROJECTTYPEID = 0 -- Corresponds to x.[FORM TYPE] = 'PROJECT'
        AND FORM_POINTS_DETAIL.PointTypeID = 8 -- Corresponds to x.[THRESHOLD TYPE] = 'Absolute Signal Deviation'
        AND FORM.DeployedProfileID IS NOT NULL -- Corresponds to x.[DEPLOYED] = 'Y'
        AND FORM.Name LIKE 'AP-%'
    GROUP BY
        FORM.ProjectID,
        FORM.Name,
        FORM_METRIC.PointTypeMetricID,
        FORM_METRIC.Description,
        FORM_POINTS.name
    HAVING
        -- This check is applied after grouping, corresponding to the original x.[HIGH WARNING] IS NOT NULL
        MAX(CASE WHEN FORM_POINTS_DETAIL_THRES.AlarmThresholdTypeID = 3 THEN FORM_POINTS_DETAIL_THRES.ThresholdValue END) IS NOT NULL
    ORDER BY
        [FORM NAME],
        [METRIC NAME];
    """