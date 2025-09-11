import pandas as pd
import pyodbc
import streamlit as st

# This class encapsulates all database logic.
# It makes the main app.py cleaner and easier to manage.
class PrismDB:
    def __init__(self, host, database, user, password):
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
        """Runs a simple query to confirm the connection is alive."""
        # A simple, fast query to ensure we can communicate with the DB.
        self.run_query("SELECT 1")

    # We use @st.cache_data to cache the *result* of a query.
    # If the same query string is passed again, Streamlit returns the
    # cached DataFrame instead of hitting the database.
    @st.cache_data
    def run_query(_self, query: str) -> pd.DataFrame:
        """
        Executes a SQL query against the database.
        The '_self' convention is used for methods within cached functions.
        """
        print(f"--- Running SQL Query: {query[:50]}... ---") # For debugging
        return pd.read_sql(query, _self._conn)