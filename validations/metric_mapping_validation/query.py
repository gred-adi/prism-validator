def get_query():
    """
    Returns the SQL query for Metric Mapping Validation.
    The hardcoded ProjectID filter has been removed to make it adaptive.
    """
    return """
    WITH
    EXIST_REV_LOGIC AS (
        SELECT
            ASSET.Description AS [ASSET],
            CASE WHEN FORM.PROJECTTYPEID = 0 THEN 'PROJECT' ELSE 'TEMPLATE' END AS [FORM TYPE],
            IIF(FORM.DeployedProfileID IS NULL, 'N', 'Y') AS [DEPLOYED],
            FORM.Name AS [FORM NAME],
            PARENT.Name AS [PARENT TEMPLATE NAME],
            FORM.PollingInterval [INTERVAL TIME (SEC)],
            CASE WHEN FORM_POINTS_SYS.DigitalGroupID IS NOT NULL THEN 'DIGITAL' WHEN FORM_POINTS_CALC.PointCalcID IS NOT NULL THEN 'PRISM CALC' ELSE 'ANALOG' END AS [POINT TYPE],
            IIF(FORM_POINTS.ConstrainedPt = 1, 'Y', 'N') AS [CONSTRAINED POINT],
            CASE WHEN FORM_POINTS.PointTypeID = 2 THEN 'OMR' ELSE FORM_METRIC.Description END AS [METRIC NAME],
            CASE WHEN FORM_POINTS.PointTypeID = 2 THEN 'OMR' ELSE CASE WHEN FORM_POINTS.ConstrainedPt = 1 THEN 'OPERATIONAL STATE' WHEN COUNT(FAULT_DETAIL.UpDownValue) >= 1 THEN 'FAULT DETECTION' ELSE 'NON-MODELED' END END AS [FUNCTION],
            FORM_POINTS.name AS [POINT NAME],
            FORM_POINTS.Description AS [POINT DESCRIPTION],
            FORM_POINTS.Units AS [POINT UNIT],
            CASE WHEN FORM_POINTS_DETAIL.PointTypeID IN (1,2,4,5,7,8,9,11,12) THEN CASE FORM_POINTS_DETAIL.PointTypeID WHEN 1 THEN 'Input Signal' WHEN 2 THEN 'Overall Model Residual' WHEN 4 THEN 'Predicted Upper Bound' WHEN 5 THEN 'Predicted Lower Bound' WHEN 7 THEN 'Relative Signal Deviation' WHEN 8 THEN 'Absolute Signal Deviation' WHEN 9 THEN 'Deviation Contribution' WHEN 11 THEN 'Actual Value' WHEN 12 THEN 'Predicted Value' END ELSE 'CONTACT SINE' END AS [THRESHOLD TYPE],
            FORM.ProjectID AS [FORM ID],
            FORM_METRIC.PointTypeMetricID AS [METRIC ID]
        FROM
            prismdb.dbo.Projects FORM
            LEFT JOIN prismdb.dbo.Assets ASSET ON FORM.AssetID = ASSET.AssetID LEFT JOIN prismdb.dbo.Projects PARENT ON FORM.ParentTemplateID = PARENT.ProjectID LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS ON FORM.ProjectID = FORM_POINTS.ProjectID AND FORM_POINTS.PointTypeID IN (1, 2) LEFT JOIN prismdb.dbo.PointTypeMetric FORM_METRIC ON FORM_POINTS.PointTypeMetricID = FORM_METRIC.PointTypeMetricID LEFT JOIN prismdb.dbo.SystemPoints FORM_POINTS_SYS ON FORM_POINTS.SystemPointId = FORM_POINTS_SYS.Id LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS_DETAIL ON FORM_POINTS.ProjectID = FORM_POINTS_DETAIL.ProjectID AND FORM_POINTS.OrderIndex = FORM_POINTS_DETAIL.OrderIndex AND ((FORM_POINTS.PointTypeID = 1 AND FORM_POINTS_DETAIL.PointTypeID <> 2) OR (FORM_POINTS.PointTypeID = 2 AND FORM_POINTS_DETAIL.PointTypeID = 2)) LEFT JOIN prismdb.dbo.PointCalc FORM_POINTS_CALC ON FORM_POINTS.ProjectPointID = FORM_POINTS_CALC.ProjectPointID LEFT JOIN prismdb.dbo.FaultDiagnostic FAULT ON (FORM.PROJECTTYPEID = 0 AND FORM.ParentTemplateID = FAULT.TemplateID) OR (FORM.PROJECTTYPEID = 2 AND FORM.ProjectID = FAULT.TemplateID) LEFT JOIN prismdb.dbo.FaultSignatureDev FAULT_DETAIL ON FAULT.FaultDiagnosticID = FAULT_DETAIL.FaultDiagnosticID AND FORM_POINTS.PointTypeMetricID = FAULT_DETAIL.PointTypeMetricID LEFT JOIN prismdb.dbo.ProfilePoints FORM_DEPLOYED_PROFILE ON FORM.DeployedProfileID = FORM_DEPLOYED_PROFILE.ProfileID AND FORM_POINTS.ProjectPointID = FORM_DEPLOYED_PROFILE.ProjectPointID LEFT JOIN prismdb.dbo.PointFilters FORM_FILTER ON FORM_POINTS.ProjectPointID = FORM_FILTER.PROJECTPOINTID
        GROUP BY
            ASSET.Description, FORM.PROJECTTYPEID, FORM.Name, PARENT.Name, FORM_POINTS_SYS.DigitalGroupID, FORM_POINTS_CALC.PointCalcID, FORM_POINTS.ConstrainedPt, FORM_METRIC.Description, FORM_POINTS.name, FORM_POINTS.Description, FORM_POINTS.ExtendedID, FORM_POINTS.ExtendedDescription, FORM_POINTS.Units, FORM_POINTS.PointTypeID, FORM_POINTS_DETAIL.PointTypeID, FORM.DeployedProfileID, FORM.ProjectID, FORM_POINTS.ProjectPointID, PARENT.ProjectID, FORM_DEPLOYED_PROFILE.ProjectPointID, FORM_METRIC.PointTypeMetricID, FORM.PollingInterval, FORM_POINTS_DETAIL.ProjectPointID
    ),
    NON_EXIST_REV_LOGIC AS (
        SELECT
            PROJECT_ASSET.Description AS [ASSET],
            'PROJECT' AS [FORM TYPE],
            IIF(PROJECT.DeployedProfileID IS NULL, 'N', 'Y') AS [DEPLOYED],
            MissingMetrics.ProjectName AS [FORM NAME],
            TEMPLATE.Name AS [PARENT TEMPLATE NAME],
            X.[INTERVAL TIME (SEC)],
            X.[POINT TYPE],
            'N' AS [CONSTRAINED POINT],
            X.[METRIC NAME],
            X.[FUNCTION],
            '' AS [POINT NAME],
            '' AS [POINT DESCRIPTION],
            '' AS [POINT UNIT],
            X.[THRESHOLD TYPE],
            MissingMetrics.ProjectID AS [FORM ID],
            MissingMetrics.[METRIC ID] AS [METRIC ID]
        FROM (
            SELECT P.Name AS ProjectName, PTM.PointTypeMetricID AS [METRIC ID], PTM.Description AS [METRIC NAME], P.ProjectID AS ProjectID, T.ProjectID AS TemplateID, T.Name AS TemplateName FROM prismdb.dbo.Projects T LEFT JOIN prismdb.dbo.ProjectPoints TP ON T.ProjectID = TP.ProjectID AND TP.PointTypeID = 1 LEFT JOIN prismdb.dbo.PointTypeMetric PTM ON TP.PointTypeMetricID = PTM.PointTypeMetricID LEFT JOIN prismdb.dbo.Projects P ON T.ProjectID = P.ParentTemplateID WHERE T.PROJECTTYPEID = 2 AND P.PROJECTTYPEID = 0
            EXCEPT
            SELECT P.Name AS ProjectName, PM.PointTypeMetricID AS [METRIC ID], PM.Description AS [METRIC NAME], P.ProjectID AS ProjectID, T.ProjectID AS TemplateID, T.Name AS TemplateName FROM prismdb.dbo.Projects P LEFT JOIN prismdb.dbo.ProjectPoints PP ON P.ProjectID = PP.ProjectID AND PP.PointTypeID = 1 LEFT JOIN prismdb.dbo.PointTypeMetric PM ON PP.PointTypeMetricID = PM.PointTypeMetricID LEFT JOIN prismdb.dbo.Projects T ON P.ParentTemplateID = T.ProjectID WHERE P.PROJECTTYPEID = 0
        ) AS MissingMetrics
        LEFT JOIN EXIST_REV_LOGIC X ON MissingMetrics.TemplateID = X.[FORM ID] AND MissingMetrics.[METRIC NAME] = X.[METRIC NAME] AND X.[FORM TYPE] = 'TEMPLATE'
        LEFT JOIN prismdb.dbo.Projects PROJECT ON MissingMetrics.ProjectID = PROJECT.ProjectID
        LEFT JOIN prismdb.dbo.Projects TEMPLATE ON MissingMetrics.TemplateID = TEMPLATE.ProjectID
        LEFT JOIN prismdb.dbo.Assets PROJECT_ASSET ON PROJECT.AssetID = PROJECT_ASSET.AssetID
    ),
    -- Combine the results of the two CTEs using UNION
    COMBINED_LOGIC AS (
        SELECT [FORM ID], [FORM NAME], [METRIC ID], [METRIC NAME], [POINT NAME], [POINT DESCRIPTION], [POINT UNIT], [CONSTRAINED POINT], [FUNCTION], [POINT TYPE], [FORM TYPE], [DEPLOYED], [THRESHOLD TYPE] FROM EXIST_REV_LOGIC
        UNION
        SELECT [FORM ID], [FORM NAME], [METRIC ID], [METRIC NAME], [POINT NAME], [POINT DESCRIPTION], [POINT UNIT], [CONSTRAINED POINT], [FUNCTION], [POINT TYPE], [FORM TYPE], [DEPLOYED], [THRESHOLD TYPE] FROM NON_EXIST_REV_LOGIC
    )
    SELECT
        x.[FORM NAME], x.[METRIC NAME], x.[POINT NAME], x.[POINT DESCRIPTION], x.[POINT UNIT], x.[FUNCTION], x.[POINT TYPE]
    FROM COMBINED_LOGIC x
    WHERE
        x.[FORM TYPE] = 'PROJECT'
        AND x.[THRESHOLD TYPE] = 'Input Signal'
        AND x.[POINT NAME] IS NOT NULL
        -- The specific ProjectID list has been removed from the WHERE clause
    ORDER BY
        x.[FORM NAME],
        x.[METRIC NAME];
    """

