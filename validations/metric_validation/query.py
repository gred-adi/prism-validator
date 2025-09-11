def get_query():
    """Returns the SQL query for Metric Validation."""
    return """
    SELECT
        FORM.ProjectID AS [FORM ID],
        FORM.Name AS [FORM NAME],
        FORM_METRIC.PointTypeMetricID AS [METRIC ID],
        FORM_METRIC.Description AS [METRIC NAME],
        CASE
            WHEN FORM_POINTS.ConstrainedPt = 1 THEN 'OPERATIONAL STATE'
            WHEN COUNT(FAULT_DETAIL.UpDownValue) >= 1 THEN 'FAULT DETECTION'
            ELSE 'NON-MODELED'
        END AS [FUNCTION],
        CASE
            WHEN FORM_POINTS_SYS.DigitalGroupID IS NOT NULL THEN 'DIGITAL'
            WHEN FORM_POINTS_CALC.PointCalcID IS NOT NULL THEN 'PRISM CALC'
            ELSE 'ANALOG'
        END AS [POINT TYPE],
        'Input Signal' AS [THRESHOLD TYPE] -- This is fixed by the WHERE clause
    FROM
        prismdb.dbo.Projects FORM
        LEFT JOIN prismdb.dbo.Assets ASSET ON FORM.AssetID = ASSET.AssetID
        LEFT JOIN prismdb.dbo.Projects PARENT ON FORM.ParentTemplateID = PARENT.ProjectID
        LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS ON FORM.ProjectID = FORM_POINTS.ProjectID AND FORM_POINTS.PointTypeID IN (1, 2)
        LEFT JOIN prismdb.dbo.PointTypeMetric FORM_METRIC ON FORM_POINTS.PointTypeMetricID = FORM_METRIC.PointTypeMetricID
        LEFT JOIN prismdb.dbo.SystemPoints FORM_POINTS_SYS ON FORM_POINTS.SystemPointId = FORM_POINTS_SYS.Id
        LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS_DETAIL ON FORM_POINTS.ProjectID = FORM_POINTS_DETAIL.ProjectID
            AND FORM_POINTS.OrderIndex = FORM_POINTS_DETAIL.OrderIndex
            AND ((FORM_POINTS.PointTypeID = 1 AND FORM_POINTS_DETAIL.PointTypeID <> 2) OR (FORM_POINTS.PointTypeID = 2 AND FORM_POINTS_DETAIL.PointTypeID = 2))
        LEFT JOIN prismdb.dbo.PointCalc FORM_POINTS_CALC ON FORM_POINTS.ProjectPointID = FORM_POINTS_CALC.ProjectPointID
        LEFT JOIN prismdb.dbo.FaultDiagnostic FAULT ON (FORM.PROJECTTYPEID = 2 AND FORM.ProjectID = FAULT.TemplateID)
        LEFT JOIN prismdb.dbo.FaultSignatureDev FAULT_DETAIL ON FAULT.FaultDiagnosticID = FAULT_DETAIL.FaultDiagnosticID AND FORM_POINTS.PointTypeMetricID = FAULT_DETAIL.PointTypeMetricID
    WHERE
        FORM.PROJECTTYPEID = 2 -- This corresponds to [FORM TYPE] = 'TEMPLATE'
        AND FORM_POINTS_DETAIL.PointTypeID = 1 -- This corresponds to [THRESHOLD TYPE] = 'Input Signal'
        AND FORM.Name LIKE '%TVI%'
    GROUP BY
        FORM.ProjectID,
        FORM.Name,
        FORM_METRIC.PointTypeMetricID,
        FORM_METRIC.Description,
        FORM_POINTS.ConstrainedPt,
        FORM_POINTS_SYS.DigitalGroupID,
        FORM_POINTS_CALC.PointCalcID
    """