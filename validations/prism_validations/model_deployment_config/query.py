def get_query(asset_descriptions: list):
    """
    Generates the SQL query for Model Deployment Config based on a dynamic list of assets.
    
    Args:
        asset_descriptions: A list of asset description strings provided by the user.

    Returns:
        A formatted SQL query string.
    """
    if not asset_descriptions:
        # Return a query that safely yields no results if the list is empty.
        return "SELECT 'No asset descriptions were provided.' AS Status WHERE 1=0;"

    # Safely format the list of assets for the SQL IN clause to prevent injection.
    formatted_assets = ", ".join([f"'{asset}'" for asset in asset_descriptions])

    # The base query with the dynamic IN clause.
    return f"""
-- CHECK IF THERE ARE DEPLOYED MODELS IN TVI WITH PA ARCHIVE are DISABLED
-- CONSOLIDATED PROJECTS TABLE
-- Use Common Table Expressions (CTEs) to separate the logic for each table
WITH
-- CTE for SourceData (previously nested inside PivotData)
SourceData AS (
    SELECT
        ASSETS.Description AS [ASSET],
        TEMPLATES.ProjectID AS [TEMPLATE ID],
        TEMPLATES.Name AS [TEMPLATE_NAME],
        PROJECTS.ProjectID AS [PROJECT_ID],
        PROJECTS.Name AS [PROJECT_NAME],
        PROJECT_METRICS.PointTypeMetricID AS [METRIC_ID],
        PROJECT_METRICS.Description AS [METRIC_NAME],
        PROJECT_POINTS.Name AS [POINT_NAME],
        CASE
            WHEN PROJECT_POINTS.PointTypeID = '1' THEN 'ODBC'
            WHEN PROJECT_POINTS.PointTypeID = '2' THEN 'PA Archive (OMR)'
            WHEN PROJECT_POINTS.PointTypeID = '4' THEN 'PA Archive (Predicted - Upper Bound)'
            WHEN PROJECT_POINTS.PointTypeID = '5' THEN 'PA Archive (Predicted - Lower Bound)'
            WHEN PROJECT_POINTS.PointTypeID = '7' THEN 'PA Archive (Deviation - Relative)'
            WHEN PROJECT_POINTS.PointTypeID = '8' THEN 'PA Archive (Deviation - Absolute)'
            WHEN PROJECT_POINTS.PointTypeID = '9' THEN 'PA Archive (Deviation - Contribution)'
            WHEN PROJECT_POINTS.PointTypeID = '11' THEN 'PA Archive (Actual Value)'
            WHEN PROJECT_POINTS.PointTypeID = '12' THEN 'PA Archive (Predicted Value)'
            ELSE NULL
        END AS [POINT_TYPE]
    FROM
        prismdb.dbo.Projects PROJECTS
        LEFT JOIN prismdb.dbo.Assets ASSETS ON PROJECTS.AssetID = ASSETS.AssetID
        LEFT JOIN prismdb.dbo.Projects TEMPLATES ON PROJECTS.ParentTemplateID = TEMPLATES.ProjectID
        LEFT JOIN prismdb.dbo.ProjectPoints PROJECT_POINTS ON PROJECTS.ProjectID = PROJECT_POINTS.ProjectID AND PROJECT_POINTS.PointTypeID <> '2'
        LEFT JOIN prismdb.dbo.PointTypeMetric PROJECT_METRICS ON PROJECT_POINTS.PointTypeMetricID = PROJECT_METRICS.PointTypeMetricID
    WHERE
        PROJECTS.ProjectTypeID = 0
        --AND PROJECTS.ProjectID IN ('833') -- Sample
),
-- CTE 1: PROJECT POINTS with ODBC and PA Data (from your Table 1)
PivotData AS (
    SELECT
        [ASSET],
        [TEMPLATE ID],
        [TEMPLATE_NAME],
        [PROJECT_ID],
        [PROJECT_NAME],
        [METRIC_ID],
        [METRIC_NAME],
        [ODBC],
        [PA Archive (Predicted - Upper Bound)],
        [PA Archive (Predicted - Lower Bound)],
        [PA Archive (Deviation - Relative)],
        [PA Archive (Deviation - Absolute)],
        [PA Archive (Deviation - Contribution)],
        [PA Archive (Actual Value)],
        [PA Archive (Predicted Value)]
    FROM
        SourceData
    PIVOT (
        MAX([POINT_NAME])
        FOR [POINT_TYPE] IN (
            [ODBC],
            [PA Archive (Predicted - Upper Bound)],
            [PA Archive (Predicted - Lower Bound)],
            [PA Archive (Deviation - Relative)],
            [PA Archive (Deviation - Absolute)],
            [PA Archive (Deviation - Contribution)],
            [PA Archive (Actual Value)],
            [PA Archive (Predicted Value)]
        )
    ) AS PivotTable
),
-- CTE 2: PROJECT POINTS DETAILS (from your Table 2)
DetailsData AS (
    SELECT
        ASSETS.Description AS [ASSET],
        TEMPLATES.ProjectID AS [TEMPLATE ID],
        TEMPLATES.Name AS [TEMPLATE_NAME],
        PROJECTS.ProjectID AS [PROJECT_ID],
        PROJECTS.Name AS [PROJECT_NAME],
        IIF(PROJECTS.DeployedProfileID IS NULL, 'NO', 'YES') AS [DEPLOYED],
        PROJECT_METRICS.PointTypeMetricID AS [METRIC_ID],
        PROJECT_METRICS.Description AS [METRIC_NAME],
        PROJECT_POINTS.ProjectPointID AS [POINT_ID],
        PROJECT_POINTS.Name AS [POINT_NAME],
        PROJECT_POINTS.Description AS [POINT_DESCRIPTION],
        IIF(PROJECT_POINTS.RealTimeServiceID = 4, 'YES','NO') AS [PRISM_CALC],
        IIF(PROJECT_POINTS.ConstrainedPt = 1, 'YES', 'NO') AS [CONSTRAIN],
        IIF(COUNT(FAULT_DETAIL.UpDownValue) >= 1, 'YES', 'NO') AS [FAULT_DETECTION],
        IIF(PROJECT_DEPLOYED_PROFILE.ProjectPointID IS NOT NULL, 'YES', 'NO') AS [MODELED],
        STUFF((
            SELECT ', ' + 
            CASE 
                WHEN PF.FilterActive = 1 THEN CONCAT(PF.PointCondition, ' ', PF.PointValue) 
                WHEN PF.FilterActive = 0 THEN CONCAT(PF.PointCondition, ' ', PF.PointValue, ' (DISABLED)') 
                ELSE '' END 
            FROM prismdb.dbo.PointFilters PF 
            LEFT JOIN prismdb.dbo.ProjectPoints PF_POINTS_DETAIL ON PF.PROJECTPOINTID = PF_POINTS_DETAIL.ProjectPointID 
            LEFT JOIN prismdb.dbo.PointTypeMetric PF_METRIC ON PF_POINTS_DETAIL.PointTypeMetricID = PF_METRIC.PointTypeMetricID 
            WHERE 
            (
                (PF.PROJECTPOINTID = PROJECT_POINTS.ProjectPointID AND PF.ProjectID = PROJECTS.ProjectID) OR 
                (PF_METRIC.PointTypeMetricID = PROJECT_METRICS.PointTypeMetricID AND PF.ProjectID = TEMPLATES.ProjectID)
            ) 
            AND PF.FILTERID NOT IN (SELECT PF2.TemplateParentFilterID FROM prismdb.dbo.PointFilters PF2 WHERE PF2.ProjectID = PROJECTS.ProjectID) FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'), 1, 2, '') AS [FILTER]
    FROM
        prismdb.dbo.Projects PROJECTS
        LEFT JOIN prismdb.dbo.Assets ASSETS ON PROJECTS.AssetID = ASSETS.AssetID
        LEFT JOIN prismdb.dbo.Projects TEMPLATES ON PROJECTS.ParentTemplateID = TEMPLATES.ProjectID
        -- Note: This query specifically targets PointTypeID = 1, which corresponds to the [ODBC] column in the pivot
        LEFT JOIN prismdb.dbo.ProjectPoints PROJECT_POINTS ON PROJECTS.ProjectID = PROJECT_POINTS.ProjectID AND PROJECT_POINTS.PointTypeID = 1
        LEFT JOIN prismdb.dbo.PointTypeMetric PROJECT_METRICS ON PROJECT_POINTS.PointTypeMetricID = PROJECT_METRICS.PointTypeMetricID
        LEFT JOIN prismdb.dbo.FaultDiagnostic FAULT ON PROJECTS.ParentTemplateID = FAULT.TemplateID
        LEFT JOIN prismdb.dbo.FaultSignatureDev FAULT_DETAIL ON FAULT.FaultDiagnosticID = FAULT_DETAIL.FaultDiagnosticID AND PROJECT_POINTS.PointTypeMetricID = FAULT_DETAIL.PointTypeMetricID
        LEFT JOIN prismdb.dbo.ProfilePoints PROJECT_DEPLOYED_PROFILE ON PROJECTS.DeployedProfileID = PROJECT_DEPLOYED_PROFILE.ProfileID AND PROJECT_POINTS.ProjectPointID = PROJECT_DEPLOYED_PROFILE.ProjectPointID
        LEFT JOIN prismdb.dbo.PointFilters PROJECT_FILTER ON PROJECT_POINTS.ProjectPointID = PROJECT_FILTER.PROJECTPOINTID
    --WHERE
    --    PROJECTS.ProjectID IN ('833') -- Sample
    GROUP BY
        ASSETS.Description,
        TEMPLATES.ProjectID,
        TEMPLATES.Name,
        PROJECTS.ProjectID,
        PROJECTS.Name,
        PROJECTS.DeployedProfileID,
        PROJECT_METRICS.PointTypeMetricID,
        PROJECT_METRICS.Description,
        PROJECT_POINTS.ProjectPointID,
        PROJECT_POINTS.Name,
        PROJECT_POINTS.Description,
        PROJECT_POINTS.RealTimeServiceID,
        PROJECT_POINTS.ConstrainedPt,
        PROJECT_DEPLOYED_PROFILE.ProjectPointID,
        PROJECT_FILTER.PROJECTPOINTID
),
-- CTE 4: Aggregate Summary table
ConsolidatedProject AS (
	SELECT
	    -- Grouping Keys (from user selection)
	    T1.[ASSET],
	    T1.[TEMPLATE ID],
	    T1.[TEMPLATE_NAME],
	    T1.[PROJECT_ID],
	    T1.[PROJECT_NAME],
	    T2.[DEPLOYED],
	    -- Aggregations
	    COUNT(DISTINCT T1.[METRIC_ID]) AS [METRIC_COUNT],
	    -- Counts of PIVOT columns (where not null)
	    COUNT(T1.[ODBC]) AS [ODBC_COUNT],
	    COUNT(T1.[PA Archive (Predicted - Upper Bound)]) AS [PA_UPPER_BOUND_COUNT],
	    COUNT(T1.[PA Archive (Predicted - Lower Bound)]) AS [PA_LOWER_BOUND_COUNT],
	    COUNT(T1.[PA Archive (Deviation - Relative)]) AS [PA_DEV_RELATIVE_COUNT],
	    COUNT(T1.[PA Archive (Deviation - Absolute)]) AS [PA_DEV_ABSOLUTE_COUNT],
	    COUNT(T1.[PA Archive (Deviation - Contribution)]) AS [PA_DEV_CONTRIB_COUNT],
	    COUNT(T1.[PA Archive (Actual Value)]) AS [PA_ACTUAL_COUNT],
	    COUNT(T1.[PA Archive (Predicted Value)]) AS [PA_PREDICTED_COUNT],
	    -- Counts of Detail columns (where not null)
	    COUNT(T2.[POINT_ID]) AS [POINT_COUNT],
	    --COUNT(T2.[POINT_NAME]) AS [POINT_NAME_COUNT],
	    --COUNT(T2.[POINT_DESCRIPTION]) AS [POINT_DESC_COUNT],
	    COUNT(T2.[FILTER]) AS [FILTER_COUNT],
	    -- Counts of Detail columns (where 'YES')
	    SUM(CASE WHEN T2.[PRISM_CALC] = 'YES' THEN 1 ELSE 0 END) AS [PRISM_CALC_COUNT],
	    SUM(CASE WHEN T2.[CONSTRAIN] = 'YES' THEN 1 ELSE 0 END) AS [CONSTRAIN_COUNT],
	    SUM(CASE WHEN T2.[FAULT_DETECTION] = 'YES' THEN 1 ELSE 0 END) AS [FAULT_DETECTION_COUNT],
	    SUM(CASE WHEN T2.[MODELED] = 'YES' THEN 1 ELSE 0 END) AS [MODELED_COUNT]
	FROM
	    PivotData T1
	    -- Use a LEFT JOIN to keep all records from PivotData (T1)
	    LEFT JOIN DetailsData T2
	        -- Join on all common identifying columns for accuracy
	        ON T1.[ASSET] = T2.[ASSET]
	        AND T1.[TEMPLATE ID] = T2.[TEMPLATE ID]
	        AND T1.[TEMPLATE_NAME] = T2.[TEMPLATE_NAME]
	        AND T1.[PROJECT_ID] = T2.[PROJECT_ID]
	        AND T1.[PROJECT_NAME] = T2.[PROJECT_NAME]
	        AND T1.[METRIC_ID] = T2.[METRIC_ID]
	        AND T1.[METRIC_NAME] = T2.[METRIC_NAME]
	GROUP BY
	    T1.[ASSET],
	    T1.[TEMPLATE ID],
	    T1.[TEMPLATE_NAME],
	    T1.[PROJECT_ID],
	    T1.[PROJECT_NAME],
	    T2.[DEPLOYED]
),
-- CTE 5: Validation Table:
Validation AS (
SELECT
    CP.[ASSET],
    CP.[TEMPLATE ID],
    CP.[TEMPLATE_NAME],
    CP.[PROJECT_ID],
    CP.[PROJECT_NAME],
    CP.[DEPLOYED],
    CASE WHEN CP.[METRIC_COUNT] = CP.[PA_ACTUAL_COUNT] THEN N'✅' ELSE N'❌' END AS [PA_ACTUAL_ARCHIVE],
    CASE WHEN CP.[METRIC_COUNT] = CP.[PA_PREDICTED_COUNT] THEN N'✅' ELSE N'❌' END AS [PA_PREDICTED_ARCHIVE],
    CASE WHEN CP.[METRIC_COUNT] = CP.[PA_DEV_RELATIVE_COUNT] THEN N'✅' ELSE N'❌' END AS [PA_DEV_RELATIVE_ARCHIVE],
    CASE WHEN CP.[METRIC_COUNT] = CP.[PA_DEV_ABSOLUTE_COUNT] THEN N'✅' ELSE N'❌' END AS [PA_DEV_ABSOLUTE_ARCHIVE],
    CASE WHEN CP.[METRIC_COUNT] = CP.[PA_DEV_CONTRIB_COUNT] THEN N'✅' ELSE N'❌' END AS [PA_DEV_CONTRIB_ARCHIVE]
FROM ConsolidatedProject CP
),
ModelDeploymentConfig AS (
SELECT DISTINCT
	PROJECTS.ProjectID AS [PROJECT ID],
	PROJECTS.Name AS [MODEL],
	PROJECTS.PollingInterval AS [INTERVAL TIME (SEC)],
	CASE
		WHEN PROJECT_POINT_DETAIL_THRES.AlarmThresholdDenominationID = 1 THEN 'Percent of Time'
		WHEN PROJECT_POINT_DETAIL_THRES.AlarmThresholdDenominationID = 2 THEN 'Percent of Count'
		ELSE CAST(PROJECT_POINT_DETAIL_THRES.AlarmThresholdDenominationID AS VARCHAR)
	 END AS [WINDOW TYPE],
	'Y' AS [THRESHOLD ACTIVE], -- This is fixed by the WHERE clause
	PROJECT_POINT_DETAIL_THRES.ThresholdPercent AS [THRESHOLD PERCENT],
	PROJECT_POINT_DETAIL_THRES.ThresholdTimeWindowSeconds AS [THRESHOLD TIME WINDOW SECONDS],
	CASE 
	    WHEN PROJECT_POINT_DETAIL_THRES.ThresholdCountMinimum IS NOT NULL AND PROJECT_POINT_DETAIL_THRES.ThresholdCountWindow IS NOT NULL 
		    THEN CONCAT(PROJECT_POINT_DETAIL_THRES.ThresholdCountMinimum,'/',PROJECT_POINT_DETAIL_THRES.ThresholdCountWindow) 
	    ELSE NULL 
    END AS [THRESHOLD WINDOW],
	'Overall Model Residual' AS [THRESHOLD TYPE] -- This is fixed by the WHERE clause
FROM
    prismdb.dbo.Projects PROJECTS
    LEFT JOIN prismdb.dbo.Assets ASSETS ON PROJECTS.AssetID = ASSETS.AssetID
    LEFT JOIN prismdb.dbo.ProjectPoints PROJECT_POINTS ON PROJECTS.ProjectID = PROJECT_POINTS.ProjectID
    LEFT JOIN prismdb.dbo.ProjectPoints PROJECT_POINT_DETAILS ON PROJECT_POINTS.ProjectID = PROJECT_POINT_DETAILS.ProjectID
        AND PROJECT_POINTS.OrderIndex = PROJECT_POINT_DETAILS.OrderIndex
    LEFT JOIN prismdb.dbo.AlarmThresholds PROJECT_POINT_DETAIL_THRES ON PROJECT_POINT_DETAILS.ProjectPointID = PROJECT_POINT_DETAIL_THRES.ProjectPointID
WHERE
    PROJECTS.DeployedProfileID IS NOT NULL -- Corresponds to x.[DEPLOYED] = 'Y'
    AND PROJECT_POINT_DETAIL_THRES.Active = 1 -- Corresponds to x.[THRESHOLD ACTIVE] = 'Y'
    AND PROJECT_POINT_DETAILS.PointTypeID = 2 -- Corresponds to x.[THRESHOLD TYPE] LIKE '%Overall%'
)
-- Query Table
SELECT
	MDC.*,
	V.[PA_ACTUAL_ARCHIVE] AS [PA Actual Value Archive],
	[PA_PREDICTED_ARCHIVE] AS [PA Predicted Value Archive],
	[PA_DEV_ABSOLUTE_ARCHIVE] AS [PA Absolute Deviation Archive],
	[PA_DEV_RELATIVE_ARCHIVE] AS [PA Relative Deviation Archive],
	[PA_DEV_CONTRIB_ARCHIVE] AS [PA Deviation Contribution Archive]
FROM 
	Validation V
	INNER JOIN ModelDeploymentConfig MDC ON MDC.[PROJECT ID] = V.[PROJECT_ID]
WHERE
    V.ASSET IN ({formatted_assets}) -- Corresponds to V.[ASSET]
    """
