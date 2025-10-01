import pandas as pd
import pyodbc
import streamlit as st

# This class encapsulates all database logic.
# It makes the main app.py cleaner and easier to manage.
class PrismDB:
    """
    Manages the connection and query execution for the PRISM database.

    This class uses Streamlit's caching mechanisms to optimize performance.
    The database connection is cached as a resource, while query results are
    cached as data. This avoids reconnecting on every script rerun and
    prevents re-fetching data for identical queries.

    Attributes:
        _conn: The cached database connection object.
    """
    def __init__(self, host, database, user, password):
        """
        Initializes the PrismDB object and establishes a database connection.

        The connection is established using Streamlit's caching to ensure it
        is created only once and reused across multiple runs.

        Args:
            host (str): The database server host.
            database (str): The name of the database.
            user (str): The username for database access.
            password (str): The password for database access.

        Raises:
            ConnectionError: If the database connection fails.
        """
        # We use @st.cache_resource to cache the connection itself.
        # This means Streamlit creates the connection object once and reuses it
        # across reruns, avoiding the overhead of reconnecting every time.
        @st.cache_resource(ttl=3600) # Cache connection for 1 hour
        def get_connection(h, db, u, p):
            """Establishes and returns a pyodbc database connection."""
            conn_str = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={h};DATABASE={db};UID={u};PWD={p};Encrypt=yes;TrustServerCertificate=yes"
            return pyodbc.connect(conn_str, timeout=5)

        try:
            self._conn = get_connection(host, database, user, password)
        except Exception as e:
            # Reraise a more specific error to be caught by the UI
            raise ConnectionError(f"Database connection failed. Please check credentials and network. Details: {e}")

    def test_connection(self):
        """
        Runs a simple query to confirm the connection is alive.

        This method executes a `SELECT 1` query, which is a lightweight way
        to verify that the database is reachable and responsive.
        """
        # A simple, fast query to ensure we can communicate with the DB.
        self.run_query("SELECT 1")

    # We use @st.cache_data to cache the *result* of a query.
    # If the same query string is passed again, Streamlit returns the
    # cached DataFrame instead of hitting the database.
    @st.cache_data
    def run_query(_self, query: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the results as a pandas DataFrame.

        This method is decorated with `@st.cache_data`, meaning that Streamlit
        will cache the returned DataFrame. If this method is called again with
        the exact same query string, Streamlit serves the cached result
        instead of re-executing the query against the database.

        The '_self' convention is used because `@st.cache_data` hashes the
        function object itself, and including 'self' would make the hash
        dependent on the object's memory address, defeating the cache.

        Args:
            _self: The instance of the class (by convention).
            query (str): The SQL query string to be executed.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the query.
        """
        print(f"--- Running SQL Query: {query[:50]}... ---") # For debugging
        return pd.read_sql(query, _self._conn)