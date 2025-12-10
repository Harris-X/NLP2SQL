import json
import os
import time
import logging
from typing import Dict, Any
import pymysql
from pymysql.err import ProgrammingError, DatabaseError  # 细分异常所需

logger = logging.getLogger(__name__)

TIMEOUT = 60
MAX_CSV_CHARS = 4000

def get_mysql_credentials() -> Dict[str, Any]:
    credentials_path = "credentials/mysql_credential.json"
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"MySQL credentials file not found at {credentials_path}. Please create it.")
    try:
        with open(credentials_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in mysql_credential.json")


def execute_mysql_sql(sql: str, **kwargs) -> Dict[str, Any]:
    logger.info(f"Executing MySQL SQL: {sql}")
    timeout = kwargs.get('timeout', TIMEOUT)
    start_time = time.time()
    content = ""
    conn = None

    try:
        creds = get_mysql_credentials()
        conn = pymysql.connect(
            host=creds.get('host', '127.0.0.1'),
            port=int(creds.get('port', 3306)),
            user=creds['user'],
            password=creds.get('password', ''),
            database=creds.get('database'),
            connect_timeout=timeout,
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        cursor.execute(sql)

        if cursor.description:  # 有结果集结构
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]
            if not rows:  # 空结果
                content = "Query executed successfully, but no rows returned."
            else:
                # 构造 CSV
                csv_lines = [','.join(headers)]
                for r in rows:
                    csv_lines.append(','.join('' if v is None else str(v) for v in r))
                full_csv = '\n'.join(csv_lines)
                total_rows = len(rows)
                total_length = len(full_csv)
                if total_length > MAX_CSV_CHARS:
                    truncated = full_csv[:MAX_CSV_CHARS]
                    last_newline = truncated.rfind('\n')
                    if last_newline > 0:
                        truncated = truncated[:last_newline]
                    content = (
                        "Query executed successfully\n\n```csv\n"
                        f"{truncated}\n```\n\n"
                        f"Note: The result has been truncated to {MAX_CSV_CHARS} characters for display purposes. "
                        f"The complete result set contains {total_rows} rows and {total_length} characters."
                    )
                else:
                    content = f"Query executed successfully\n\n```csv\n{full_csv}\n```"
        else:
            conn.commit()
            content = "Query executed successfully."

    except ProgrammingError as e:
        content = f"SQL Error: {str(e)}"
        logger.error(content)
    except DatabaseError as e:
        content = f"Database error: {str(e)}"
        logger.error(content)
    except TimeoutError:
        content = f"Execution timed out after {timeout} seconds."
        logger.error(content)
    except Exception as e:
        content = f"Unexpected error: {str(e)}"
        logger.error(content)
    finally:
        if conn:
            conn.close()
        elapsed = time.time() - start_time
        logger.info(f"MySQL execution finished in {elapsed:.2f}s")

    return {"content": f"EXECUTION RESULT of [execute_mysql_sql]:\n{content}"}


def register_tools(registry):
    registry.register_tool("execute_mysql_sql", execute_mysql_sql)
