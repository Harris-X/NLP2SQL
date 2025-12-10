import json
import os
import pymysql
import argparse
from decimal import Decimal
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Union
from run_logger import get_logger
FULL_LOG = os.getenv("ALPHASQL_LOG_FULL", "0") == "1"
SQL_LOG_PATH = os.getenv("ALPHASQL_SQL_LOG_PATH", str(Path(os.getenv("ALPHASQL_LOG_DIR", "logs")) / "sql_exec.jsonl"))
try:
    Path(SQL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

class DecimalEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，用于处理 Decimal、datetime、date 类型。
    
    由于标准JSON编码器无法处理Decimal类型和日期时间类型，
    这个自定义编码器将这些特殊类型转换为JSON兼容的格式。
    """
    def default(self, obj):
        """
        重写default方法，处理特殊数据类型。
        
        Args:
            obj: 需要编码的对象
            
        Returns:
            编码后的值，如果无法处理则调用父类方法
        """
        if isinstance(obj, Decimal):
            # 检查 Decimal 值是否为整数（即小数点后全是零）
            # 如果是整数则转为int，否则转为float
            return int(obj) if obj == obj.to_integral_value() else float(obj)
        elif isinstance(obj, (datetime, date)):
            # 将日期时间对象转为ISO格式字符串
            return obj.isoformat()
        # 其他类型使用父类的默认处理方式
        return super().default(obj)

class execute_sql_with_pymysql:
    """
    SQL执行器类，用于通过pymysql连接MySQL数据库并执行SQL语句。
    
    这个类提供了两个主要功能：
    1. execute_sql_with_pymysql: 执行查询SQL并返回结果
    2. insert_data_with_pymysql: 执行insert语句
    """

    def __init__(self):
        pass

    def execute_sql_with_pymysql(self, input_file_path:str, output_file_path:str, db_config:Dict):
        """
        执行SQL查询语句的主要方法。
        
        从输入JSON文件中读取SQL语句列表，连接到数据库执行这些SQL，
        并将执行结果保存到输出JSON文件中。
        
        Args:
            input_file_path (str): 输入JSON文件路径，包含要执行的SQL语句，文件内部数据格式应为list[dict]
            output_file_path (str): 输出JSON文件路径，用于保存执行结果，文件保存格式为list[dict]
            db_config (dict): 数据库连接配置字典，包含host、user、password等
        """
        logger = get_logger()
        results = []
        conn = None
        logger.info(f"QUERY_START input={input_file_path} output={output_file_path}")
        try:
            conn = pymysql.connect(**db_config)
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            sql_list = json.loads(Path(input_file_path).read_text())
            if not isinstance(sql_list, list):
                logger.error("INPUT_FORMAT_ERROR not list")
                return
            for item in sql_list:
                if 'sql' in item:
                    sql_id = item.get('sql_id')
                    sql_statement = item['sql']
                    if FULL_LOG:
                        logger.info(f"EXECUTE sql_id={sql_id} sql={sql_statement}")
                    else:
                        logger.info(f"EXECUTE sql_id={sql_id} len={len(sql_statement)}")
                    try:
                        cursor.execute(sql_statement)
                        query_result = cursor.fetchall()
                        query_result = self.normalize_numbers_in_result(query_result)
                        results.append({
                            "sql_id": sql_id,
                            "sql": sql_statement,
                            "status": "success",
                            "result": query_result
                        })
                        if FULL_LOG:
                            logger.info(f"EXECUTE_SUCCESS sql_id={sql_id} rows={len(query_result)} payload={json.dumps(query_result, ensure_ascii=False)}")
                        else:
                            logger.info(f"EXECUTE_SUCCESS sql_id={sql_id} rows={len(query_result)}")
                        # Append JSONL per-query entry
                        try:
                            entry = {
                                "ts": datetime.now().isoformat(),
                                "op": "query",
                                "sql_id": sql_id,
                                "status": "success",
                                "statement": sql_statement if FULL_LOG else (sql_statement[:500] + "..." if len(sql_statement) > 500 else sql_statement),
                                "row_count": len(query_result),
                                "rows": query_result if FULL_LOG else None
                            }
                            with open(SQL_LOG_PATH, "a", encoding="utf-8") as f:
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                    except pymysql.Error as e:
                        results.append({
                            "sql_id": sql_id,
                            "sql": sql_statement,
                            "status": "error",
                            "error_message": str(e)
                        })
                        if FULL_LOG:
                            logger.warning(f"EXECUTE_ERROR sql_id={sql_id} err={e} sql={sql_statement}")
                        else:
                            logger.warning(f"EXECUTE_ERROR sql_id={sql_id} err={e}")
                        try:
                            entry = {
                                "ts": datetime.now().isoformat(),
                                "op": "query",
                                "sql_id": sql_id,
                                "status": "error",
                                "statement": sql_statement if FULL_LOG else (sql_statement[:500] + "..." if len(sql_statement) > 500 else sql_statement),
                                "error": str(e)
                            }
                            with open(SQL_LOG_PATH, "a", encoding="utf-8") as f:
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                else:
                    results.append({
                        "sql": None,
                        "status": "error",
                        "error_message": "JSON 元素缺少 'sql' 键"
                    })
                    logger.warning("MISSING_SQL_KEY one item")
        except FileNotFoundError:
            logger.error(f"FILE_NOT_FOUND path={input_file_path}")
            return
        except json.JSONDecodeError:
            logger.error(f"JSON_DECODE_ERROR path={input_file_path}")
            return
        except pymysql.Error as e:
            logger.error(f"DB_ERROR err={e}")
            return
        finally:
            if conn:
                conn.close()
        try:
            output_path = Path(output_file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=4, cls=DecimalEncoder))
            logger.info(f"QUERY_WRITE_DONE file={output_file_path} count={len(results)}")
        except Exception as e:
            logger.error(f"WRITE_ERROR err={e}")

    def insert_data_with_pymysql(self, input_file_path:str, output_file_path:str, db_config:Dict):
        """
        执行SQL插入语句的方法。
        
        从一个包含SQL插入语句的JSON文件中读取数据，连接到数据库执行这些SQL，
        并将执行结果保存到输出JSON文件中。与查询方法的主要区别是不需要返回查询结果。
        
        Args:
            input_file_path (str): 输入JSON文件路径，包含要执行的插入SQL语句,文件内部结果为list[dict]
            output_file_path (str): 输出JSON文件路径，用于保存执行结果,文件保存格式为list[dict]
            db_config (dict): 数据库连接配置字典
        """
        logger = get_logger()
        results = []
        conn = None
        logger.info(f"INSERT_START input={input_file_path} output={output_file_path}")
        try:
            conn = pymysql.connect(**db_config)
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            sql_list = json.loads(Path(input_file_path).read_text())
            if not isinstance(sql_list, list):
                logger.error("INPUT_FORMAT_ERROR not list")
                return
            for item in sql_list:
                if 'insert_sql' in item:
                    sql_id = item.get('sql_id')
                    sql_statement = item['insert_sql']
                    if FULL_LOG:
                        logger.info(f"INSERT_EXEC sql_id={sql_id} sql={sql_statement}")
                    else:
                        logger.info(f"INSERT_EXEC sql_id={sql_id} len={len(sql_statement)}")
                    try:
                        cursor.execute(sql_statement)
                        conn.commit()
                        results.append({
                            "sql_id": sql_id,
                            "insert_sql": sql_statement,
                            "status": "success",
                        })
                        logger.info(f"INSERT_SUCCESS sql_id={sql_id}")
                        try:
                            entry = {
                                "ts": datetime.now().isoformat(),
                                "op": "insert",
                                "sql_id": sql_id,
                                "status": "success",
                                "statement": sql_statement if FULL_LOG else (sql_statement[:500] + "..." if len(sql_statement) > 500 else sql_statement)
                            }
                            with open(SQL_LOG_PATH, "a", encoding="utf-8") as f:
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                    except pymysql.Error as e:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        results.append({
                            "sql_id": sql_id,
                            "insert_sql": sql_statement,
                            "status": "error",
                            "error_message": str(e)
                        })
                        if FULL_LOG:
                            logger.warning(f"INSERT_ERROR sql_id={sql_id} err={e} sql={sql_statement}")
                        else:
                            logger.warning(f"INSERT_ERROR sql_id={sql_id} err={e}")
                        try:
                            entry = {
                                "ts": datetime.now().isoformat(),
                                "op": "insert",
                                "sql_id": sql_id,
                                "status": "error",
                                "statement": sql_statement if FULL_LOG else (sql_statement[:500] + "..." if len(sql_statement) > 500 else sql_statement),
                                "error": str(e)
                            }
                            with open(SQL_LOG_PATH, "a", encoding="utf-8") as f:
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                else:
                    results.append({
                        "sql_id": item.get('sql_id'),
                        "insert_sql": None,
                        "status": "error",
                        "error_message": "JSON 元素缺少 'sql' 键"
                    })
                    logger.warning("MISSING_INSERT_SQL_KEY one item")
        except FileNotFoundError:
            logger.error(f"FILE_NOT_FOUND path={input_file_path}")
            return
        except json.JSONDecodeError:
            logger.error(f"JSON_DECODE_ERROR path={input_file_path}")
            return
        except pymysql.Error as e:
            logger.error(f"DB_ERROR err={e}")
            return
        finally:
            if conn:
                conn.close()
        try:
            output_path = Path(output_file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, ensure_ascii=False, indent=4))
            logger.info(f"INSERT_WRITE_DONE file={output_file_path} count={len(results)}")
        except Exception as e:
            logger.error(f"WRITE_ERROR err={e}")
    
    def normalize_numbers_in_result(self, result_list: List[Dict]) -> List[Dict]:
        """
        对查询结果中的数字进行标准化处理 (使用生成式精简版)。
        
        遍历查询结果，将float类型中实际为整数的值转为int，否则保留两位小数。
        """
        
        # 内部辅助函数，用于处理单个键值对的标准化逻辑
        def _normalize_value(value):
            if isinstance(value, float):
                # 如果是浮点数但无小数部分，则转为整数
                if value.is_integer():
                    return int(value)
                else:
                    # 保留两位小数
                    return round(value, 2)
            if isinstance(value, Decimal): # 针对Decimal类型，同样保留两位小数
                return round(value, 2)
            else:
                # 其他类型保持原样
                return value

        # 使用列表生成式迭代行 (row)，内部使用字典生成式迭代列 (key, value)
        normalized = [
            {
                key: _normalize_value(value)
                for key, value in row.items()
            }
            for row in result_list
        ]
        
        return normalized


# --- 示例用法 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量执行查询或插入 SQL 的辅助脚本 (含日志)')
    parser.add_argument('--mode', choices=['query', 'insert'], default='query', help='执行模式：query 执行查询；insert 执行插入')
    parser.add_argument('--input', required=False, help='输入 JSON 文件路径 (包含 sql 或 insert_sql 列表)')
    parser.add_argument('--output', required=False, help='输出结果 JSON 文件路径')
    parser.add_argument('--host', default=os.getenv('SR_HOST', '127.0.0.1'), help='数据库主机')
    parser.add_argument('--port', type=int, default=int(os.getenv('SR_PORT', '9030')), help='数据库端口')
    parser.add_argument('--user', default=os.getenv('SR_USER', 'root'), help='数据库用户名')
    parser.add_argument('--password', default=os.getenv('SR_PASSWORD', ''), help='数据库密码')
    parser.add_argument('--db', default=os.getenv('SR_DB', 'final_algorithm_competition'), help='数据库名称')

    args = parser.parse_args()

    # 默认路径（允许被命令行覆盖）
    default_input = '/root/autodl-tmp/comp/LLaMA-Factory/Alpha-SQL-master/data/test.json'
    default_output = '/root/autodl-tmp/comp/LLaMA-Factory/Alpha-SQL-master/result/test.json'
    input_path = args.input if args.input else default_input
    output_path = args.output if args.output else default_output

    sql_executor = execute_sql_with_pymysql()
    db_configuration = {
        'host': args.host,
        'user': args.user,
        'password': args.password,
        'db': args.db,
        'port': args.port
    }

    logger = get_logger()
    logger.info(f"CLI_MODE={args.mode} input={input_path} output={output_path}")
    if args.mode == 'insert':
        sql_executor.insert_data_with_pymysql(input_file_path=input_path, output_file_path=output_path, db_config=db_configuration)
    else:
        sql_executor.execute_sql_with_pymysql(input_file_path=input_path, output_file_path=output_path, db_config=db_configuration)
    logger.info("CLI_DONE")