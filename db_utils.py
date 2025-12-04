import pandas as pd
import pyodbc
import streamlit as st

# This class encapsulates all database logic.
# It makes the main app.py cleaner and easier to manage.
class PrismDB:
    """A class to manage the connection and queries to the PRISM database.

    This class handles the database connection, caching, and query execution.
    It uses Streamlit's caching mechanisms to avoid reconnecting to the
    database on every rerun and to cache the results of queries.

    Attributes:
        _conn: The database connection object.
    """
    def __init__(self, host, database, user, password):
        """Initializes the PrismDB class and establishes a database connection.

        Args:
            host (str): The database host.
            database (str): The database name.
            user (str): The database user.
            password (str): The database password.

        Raises:
            ConnectionError: If the database connection fails.
        """
        # We use @st.cache_resource to cache the connection itself.
        # This means Streamlit creates the connection object once and reuses it
        # across reruns, avoiding the overhead of reconnecting every time.
        @st.cache_resource(ttl=3600) # Cache connection for 1 hour
        def get_connection(h, db, u, p):
            conn_str = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={h};DATABASE={db};UID={u};PWD={p};Encrypt=yes;TrustServerCertificate=yes"
            return pyodbc.connect(conn_str, timeout=5)

        try:
            self._conn = get_connection(host, database, user, password)
        except Exception as e:
            # Reraise a more specific error to be caught by the UI
            raise ConnectionError(f"Database connection failed. Please check credentials and network. Details: {e}")

    def test_connection(self):
        """Runs a simple query to confirm the connection is alive.

        This method executes a simple 'SELECT 1' query to verify that the
        database connection is active and responsive.
        """
        # A simple, fast query to ensure we can communicate with the DB.
        self.run_query("SELECT 1")

    # We use @st.cache_data to cache the *result* of a query.
    # If the same query string is passed again, Streamlit returns the
    # cached DataFrame instead of hitting the database.
    @st.cache_data
    def run_query(_self, query: str) -> pd.DataFrame:
        """Executes a SQL query and returns the results as a DataFrame.

        This method runs a SQL query against the connected database and returns
        the results in a pandas DataFrame. The results are cached using
        Streamlit's `@st.cache_data` to avoid re-running the same query.

        Args:
            _self: The instance of the class. The '_self' convention is used
                   for methods within cached functions in Streamlit.
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the query.
        """
        return pd.read_sql(query, _self._conn)