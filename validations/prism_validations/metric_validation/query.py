def get_query(tdt_names=None):
    """
    Returns the SQL query for validatng point types (Analog/Digital).
    Optionally filters by a list of TDT names (Template names) to optimize performance.
    """
    # Base filtering condition
    where_clause = "P.ProjectTypeID = 2 AND P.Name LIKE 'AP-%' AND PP.PointTypeID = 1"

    # Add dynamic TDT filtering if names are provided
    if tdt_names:
        # Sanitize and format names for SQL IN clause
        sanitized_names = [name.replace("'", "''") for name in tdt_names]
        formatted_names = ", ".join([f"'{name}'" for name in sanitized_names])
        where_clause += f" AND P.Name IN ({formatted_names})"

    return f"""
    SELECT
        P.Name AS [FORM NAME],
        M.Description AS [METRIC NAME],
        CASE
            WHEN PP.SystemPointId IS NOT NULL THEN 'Digital'
            WHEN PC.PointCalcID IS NOT NULL THEN 'PRiSM Calc'
            ELSE 'Analog'
        END AS [POINT_TYPE_PRISM]
    FROM
        prismdb.dbo.Projects P
        JOIN prismdb.dbo.ProjectPoints PP ON P.ProjectID = PP.ProjectID
        JOIN prismdb.dbo.PointTypeMetric M ON PP.PointTypeMetricID = M.PointTypeMetricID
        LEFT JOIN prismdb.dbo.PointCalc PC ON PP.ProjectPointID = PC.ProjectPointID
    WHERE
        {where_clause}
    """

def get_calc_query(tdt_names=None):
    """
    Returns the SQL query for fetching PRISM Calculations details.
    Optionally filters by a list of TDT names to optimize performance.
    """
    # Base filtering condition
    where_clause = "FORM.PROJECTTYPEID = 2 AND FORM_POINTS_DETAIL.PointTypeID = 1 AND FORM.Name LIKE 'AP-%'"

    # Add dynamic TDT filtering if names are provided
    if tdt_names:
        # Sanitize and format names for SQL IN clause
        sanitized_names = [name.replace("'", "''") for name in tdt_names]
        formatted_names = ", ".join([f"'{name}'" for name in sanitized_names])
        where_clause += f" AND FORM.Name IN ({formatted_names})"

    return f"""
-- Calculations (PRISM)
WITH
Templates_Base AS (
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
	    'Input Signal' AS [THRESHOLD TYPE], -- This is fixed by the WHERE clause
	    FORM_POINTS_CALC.CalcScript,
		LTRIM(RTRIM(
	    	CASE
			    WHEN CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript) > 0
			     AND CHARINDEX('// END USER CODE //', FORM_POINTS_CALC.CalcScript) > 0
			     AND CHARINDEX('// END USER CODE //', FORM_POINTS_CALC.CalcScript) > CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript)
			    THEN SUBSTRING(
			        FORM_POINTS_CALC.CalcScript,
			        CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript) + LEN('// BEGIN USER CODE //'),
			        CHARINDEX('// END USER CODE //', FORM_POINTS_CALC.CalcScript) - (CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript) + LEN('// BEGIN USER CODE //'))
			    )
			    ELSE NULL
			END
		)) AS [CALC_CODE],
		LTRIM(RTRIM(
			CASE
				WHEN CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) > 0
				 AND CHARINDEX('return UserFunction();', FORM_POINTS_CALC.CalcScript) > 0
				 AND CHARINDEX('return UserFunction();', FORM_POINTS_CALC.CalcScript) > (CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) + LEN('Points = ProjectPoints;'))
				THEN SUBSTRING(
					FORM_POINTS_CALC.CalcScript,
					CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) + LEN('Points = ProjectPoints;'),
					CHARINDEX('return UserFunction();', FORM_POINTS_CALC.CalcScript) - (CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) + LEN('Points = ProjectPoints;'))
				)
				ELSE NULL
			END
		)) AS [CALC_VARIABLES_RAW]
	FROM
	    prismdb.dbo.Projects FORM
	    LEFT JOIN prismdb.dbo.Assets ASSET ON FORM.AssetID = ASSET.AssetID
	    LEFT JOIN prismdb.dbo.Projects PARENT ON FORM.ParentTemplateID = PARENT.ProjectID
	    LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS ON FORM.ProjectID = FORM_POINTS.ProjectID AND FORM_POINTS.PointTypeID IN (1, 2)
	    LEFT JOIN prismdb.dbo.PointTypeMetric FORM_METRIC ON FORM_POINTS.PointTypeMetricID = FORM_METRIC.PointTypeMetricID
	    LEFT JOIN prismdb.dbo.SystemPoints FORM_POINTS_SYS ON FORM_POINTS.SystemPointId = FORM_POINTS_SYS.Id
	    LEFT JOIN prismdb.dbo.ProjectPoints FORM_POINTS_DETAIL ON FORM.ProjectID = FORM_POINTS_DETAIL.ProjectID
	        AND FORM_POINTS.OrderIndex = FORM_POINTS_DETAIL.OrderIndex
	        AND ((FORM_POINTS.PointTypeID = 1 AND FORM_POINTS_DETAIL.PointTypeID <> 2) OR (FORM_POINTS.PointTypeID = 2 AND FORM_POINTS_DETAIL.PointTypeID = 2))
	    LEFT JOIN prismdb.dbo.PointCalc FORM_POINTS_CALC ON FORM_POINTS.ProjectPointID = FORM_POINTS_CALC.ProjectPointID
	    LEFT JOIN prismdb.dbo.FaultDiagnostic FAULT ON (FORM.PROJECTTYPEID = 2 AND FORM.ProjectID = FAULT.TemplateID)
	    LEFT JOIN prismdb.dbo.FaultSignatureDev FAULT_DETAIL ON FAULT.FaultDiagnosticID = FAULT_DETAIL.FaultDiagnosticID AND FORM_POINTS.PointTypeMetricID = FAULT_DETAIL.PointTypeMetricID
	WHERE
	    {where_clause}
	GROUP BY
		FORM.ProjectID,
	    FORM.Name,
	    FORM_METRIC.PointTypeMetricID,
	    FORM_METRIC.Description,
	    FORM_POINTS.ConstrainedPt,
	    FORM_POINTS_SYS.DigitalGroupID,
	    FORM_POINTS_CALC.PointCalcID,
	    FORM_POINTS_CALC.CalcScript,
		-- Added the new column expression to the GROUP BY to ensure compatibility
		LTRIM(RTRIM(
	    	CASE
			    WHEN CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript) > 0
			     AND CHARINDEX('// END USER CODE //', FORM_POINTS_CALC.CalcScript) > 0
			     AND CHARINDEX('// END USER CODE //', FORM_POINTS_CALC.CalcScript) > CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript)
			    THEN SUBSTRING(
			        FORM_POINTS_CALC.CalcScript,
			        CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript) + LEN('// BEGIN USER CODE //'),
			        CHARINDEX('// END USER CODE //', FORM_POINTS_CALC.CalcScript) - (CHARINDEX('// BEGIN USER CODE //', FORM_POINTS_CALC.CalcScript) + LEN('// BEGIN USER CODE //'))
			    )
			    ELSE NULL
			END
		)),
		-- Added the new column expression to the GROUP BY to ensure compatibility
		LTRIM(RTRIM(
			CASE
				WHEN CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) > 0
				 AND CHARINDEX('return UserFunction();', FORM_POINTS_CALC.CalcScript) > 0
				 AND CHARINDEX('return UserFunction();', FORM_POINTS_CALC.CalcScript) > (CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) + LEN('Points = ProjectPoints;'))
				THEN SUBSTRING(
					FORM_POINTS_CALC.CalcScript,
					CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) + LEN('Points = ProjectPoints;'),
					CHARINDEX('return UserFunction();', FORM_POINTS_CALC.CalcScript) - (CHARINDEX('Points = ProjectPoints;', FORM_POINTS_CALC.CalcScript) + LEN('Points = ProjectPoints;'))
				)
				ELSE NULL
			END
		))
),
Templates AS (
    SELECT
        T_Base.*, -- Select all columns from Templates_Base
        LTRIM(RTRIM(
            CASE
                WHEN T_Base.[CALC_CODE] IS NOT NULL
                 AND CHARINDEX('return', T_Base.[CALC_CODE]) > 0
                 AND CHARINDEX(';', T_Base.[CALC_CODE], CHARINDEX('return', T_Base.[CALC_CODE])) > 0
                THEN SUBSTRING(
                    T_Base.[CALC_CODE],
                    CHARINDEX('return', T_Base.[CALC_CODE]) + LEN('return'),
                    CHARINDEX(';', T_Base.[CALC_CODE], CHARINDEX('return', T_Base.[CALC_CODE])) - (CHARINDEX('return', T_Base.[CALC_CODE]) + LEN('return'))
                )
                ELSE NULL
            END
        )) AS [CALC_LOGIC],
		-- New column to transform CALC_VARIABLES_RAW
        REPLACE(
            REPLACE(
                T_Base.[CALC_VARIABLES_RAW],
                'Points.GetByPointMetricIDAndOperation(', 
                ''
            ),
            ', 0)', 
            ''
        ) AS [CALC_VARIABLES],
		-- New column with metric names. Replaced STRING_SPLIT and STRING_AGG with XML methods for compatibility.
		Aggregated.NamedVariables AS [CALC_VARIABLES_NAMES]
    FROM
        Templates_Base AS T_Base
	-- This OUTER APPLY splits, parses, joins, and re-aggregates the variable strings
	-- This version uses XML methods for string splitting and aggregation to support older SQL Server versions (pre-2017)
	OUTER APPLY (
		SELECT 
			-- Replacement for STRING_AGG: STUFF + FOR XML PATH
			STUFF(
				(
					-- Use CHAR(10) (Line Feed) only to avoid XML encoding of CHAR(13)
					SELECT CHAR(10) + -- Re-aggregate with newlines
						-- Reconstruct the string: "   A = " + " " + "Metric Name" + ";"
						SUBSTRING(s_trimmed.value, 1, CHARINDEX('=', s_trimmed.value)) -- Gets "   A = "
						+ ' ' + ISNULL(METRIC.Description, 'METRIC NOT FOUND')
						+ ';'
					FROM 
						-- Replacement for STRING_SPLIT: XML nodes method
						(
							SELECT 
								CAST(Split.a.value('.', 'VARCHAR(1000)') AS VARCHAR(1000)) AS value
							FROM
							(
								-- Convert the string to XML, replacing ';' with node boundaries
								SELECT CAST ('<M>' + REPLACE(T_Base.[CALC_VARIABLES_RAW], ';', '</M><M>') + '</M>' AS XML) AS Data
							) AS A
							CROSS APPLY Data.nodes ('/M') AS Split(a)
						) AS s
					-- Clean up each split line
					CROSS APPLY ( SELECT LTRIM(RTRIM(s.value)) AS value ) AS s_trimmed
					-- Parse the Metric ID from the cleaned-up line
					CROSS APPLY (
						SELECT 
							TRY_CAST(
								LTRIM(RTRIM(
									SUBSTRING(
										s_trimmed.value, 
										CHARINDEX('(', s_trimmed.value) + 1, 
										CHARINDEX(',', s_trimmed.value) - CHARINDEX('(', s_trimmed.value) - 1
									)
								))
							AS INT) AS MetricID
					) AS Parsed
					-- Join to the metrics table to get the name
					LEFT JOIN prismdb.dbo.PointTypeMetric METRIC 
						ON METRIC.PointTypeMetricID = Parsed.MetricID
					WHERE
						-- Ensure the line is not empty and contains a variable assignment
						s_trimmed.value <> '' AND s_trimmed.value LIKE '%=%'
					ORDER BY 
						s_trimmed.value -- Order for FOR XML PATH
					FOR XML PATH('')
				), 1, 1, '' -- STUFF to remove the leading newline (1 char now)
			) AS NamedVariables
	) AS Aggregated
)
SELECT
	T.[FORM ID],
	T.[FORM NAME],
	T.[METRIC ID],
	T.[METRIC NAME],
	T.[FUNCTION],
	T.[POINT TYPE],
	T.[CALC_LOGIC],
	T.[CALC_VARIABLES_NAMES]
FROM Templates T
WHERE
	T.[FORM NAME] LIKE 'AP-%'
	AND T.[POINT TYPE] = 'PRISM CALC'
ORDER BY
	T.[FORM NAME],
	T.[METRIC NAME]
    """