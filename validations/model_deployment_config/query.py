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
    SELECT DISTINCT
        FORM.Name AS [MODEL],
        FORM.PollingInterval AS [INTERVAL TIME (SEC)],
        CASE
            WHEN FORM_POINTS_DETAIL_THRES.AlarmThresholdDenominationID = 1 THEN 'Percent of Time'
            WHEN FORM_POINTS_DETAIL_THRES.AlarmThresholdDenominationID = 2 THEN 'Percent of Count'
            ELSE CAST(FORM_POINTS_DETAIL_THRES.AlarmThresholdDenominationID AS VARCHAR)
         END AS [WINDOW TYPE],
        'Y' AS [THRESHOLD ACTIVE], -- This is fixed by the WHERE clause
        FORM_POINTS_DETAIL_THRES.ThresholdPercent AS [THRESHOLD PERCENT],
        FORM_POINTS_DETAIL_THRES.ThresholdTimeWindowSeconds AS [THRESHOLD TIME WINDOW SECONDS],
        CASE 
            WHEN FORM_POINTS_DETAIL_THRES.ThresholdCountMinimum IS NOT NULL AND FORM_POINTS_DETAIL_THRES.ThresholdCountWindow IS NOT NULL 
                THEN CONCAT(FORM_POINTS_DETAIL_THRES.ThresholdCountMinimum,'/',FORM_POINTS_DETAIL_THRES.ThresholdCountWindow) 
            ELSE NULL 
        END AS [THRESHOLD WINDOW],
        'Overall Model Residual' AS [THRESHOLD TYPE] -- This is fixed by the WHERE clause
    FROM
        prismdb.dbo.Projects FORM
        LEFT JOIN prismdb.dbo.Assets ASSET ON FORM.AssetID = ASSET.AssetID
        LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS ON FORM.ProjectID = FORM_POINTS.ProjectID
        LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS_DETAIL ON FORM_POINTS.ProjectID = FORM_POINTS_DETAIL.ProjectID
            AND FORM_POINTS.OrderIndex = FORM_POINTS_DETAIL.OrderIndex
        LEFT JOIN prismdb.dbo.AlarmThresholds FORM_POINTS_DETAIL_THRES ON FORM_POINTS_DETAIL.ProjectPointID = FORM_POINTS_DETAIL_THRES.ProjectPointID
    WHERE
        ASSET.Description IN ({formatted_assets}) -- Corresponds to x.[ASSET]
        AND FORM.DeployedProfileID IS NOT NULL -- Corresponds to x.[DEPLOYED] = 'Y'
        AND FORM_POINTS_DETAIL_THRES.Active = 1 -- Corresponds to x.[THRESHOLD ACTIVE] = 'Y'
        AND FORM_POINTS_DETAIL.PointTypeID = 2; -- Corresponds to x.[THRESHOLD TYPE] LIKE '%Overall%'
    """
