import os
import logging
from typing import Dict, Any, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class PostgresConnector:
    """Connection handler for PostgreSQL database"""
    
    def __init__(self):
        """Initialize PostgreSQL connection"""
        self.conn = self._create_connection()
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
    
    def _create_connection(self):
        """Create a new PostgreSQL connection"""
        try:
            user = os.getenv("PG_USER")
            password = os.getenv("PG_PASSWORD")
            host = os.getenv("PG_HOST")
            port = os.getenv("PG_PORT")
            db = os.getenv("PG_DATABASE")
            
            connection_params = {
                "user": user, 
                "password": password, 
                "host": host, 
                "port": port, 
                "dbname": db
            }
            
            logger.info(f"Connecting to PostgreSQL database {db} on {host}:{port}")
            return psycopg2.connect(**connection_params)
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            raise
    
    def ensure_connection(self):
        """Ensure the connection is still alive, reconnect if needed"""
        try:
            # Check if connection is closed or in error state
            if self.conn.closed:
                logger.warning("PostgreSQL connection is closed. Reconnecting...")
                self.conn = self._create_connection()
                self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                return
                
            # Test connection with a simple query
            self.cursor.execute("SELECT 1")
        except Exception as e:
            logger.warning(f"Connection test failed: {str(e)}. Reconnecting...")
            try:
                # Close if still open to avoid leaks
                if not self.conn.closed:
                    self.conn.close()
                
                # Reconnect
                self.conn = self._create_connection()
                self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect: {str(reconnect_error)}")
                raise
    
    def connect(self):
        """Connect to PostgreSQL database using environment variables"""
        try:
            self.conn = self._create_connection()
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            raise
    
    def disconnect(self):
        """Close the database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("PostgreSQL database connection closed")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, fetch: bool = True) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        try:
            # Ensure connection is valid before executing
            self.ensure_connection()
            
            # Execute the query
            self.cursor.execute(query, params or {})
            self.conn.commit()
            
            if fetch and self.cursor.description:
                return self.cursor.fetchall()
            return []
        except Exception as e:
            # Try to rollback if possible
            try:
                if not self.conn.closed:
                    self.conn.rollback()
            except:
                pass  # If rollback fails, just continue to error handling
                
            logger.error(f"Error executing query: {e}\nQuery: {query}\nParams: {params}")
            raise
    
    def call_procedure(self, procedure: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Call a stored procedure and return results"""
        try:
            # Ensure connection is valid before executing
            self.ensure_connection()
            
            query = f"CALL {procedure}(%({')::, %('.join(params.keys()) if params else ''})::)"
            self.cursor.execute(query, params or {})
            self.conn.commit()
            
            if self.cursor.description:
                return self.cursor.fetchall()
            return []
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error calling procedure: {e}\nProcedure: {procedure}\nParams: {params}")
            raise
    
    def call_function(self, function: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Call a database function and return result"""
        try:
            # Ensure connection is valid before executing
            self.ensure_connection()
            
            if params:
                param_str = ', '.join([f'%({k})s::{self._get_pg_type(v)}' for k, v in params.items()])
                query = f"SELECT * FROM {function}({param_str})"
            else:
                query = f"SELECT * FROM {function}()"
                
            self.cursor.execute(query, params or {})
            self.conn.commit()
            
            if self.cursor.description:
                return self.cursor.fetchone()
            return None
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error calling function: {e}\nFunction: {function}\nParams: {params}")
            raise
    
    def _get_pg_type(self, value: Any) -> str:
        """Get PostgreSQL type for Python value"""
        if isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, dict) or isinstance(value, list):
            return 'jsonb'
        return 'text'  # Default to text for strings and unknown types

# Singleton instance
_pg_db = None

def get_pg_db():
    """Get singleton database connection"""
    global _pg_db
    if _pg_db is None:
        _pg_db = PostgresConnector()
    return _pg_db
