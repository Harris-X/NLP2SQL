import os
import subprocess
import json
import re
import sys
sys.path.append('/home/users/xueqi/text2sql/Spider2/test_fewshot')
from get_fewshot_example import FewshotSelector

class BasePromptBuilder:
    
    def load_system_prompt(self, args):
        """Load system prompt from file"""
        with open(args.system_prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def load_external_knowledge(self, external_knowledge_file, args):
        """Load external knowledge from file"""
        if not external_knowledge_file:
            return None
        
        knowledge_path = os.path.join(args.documents_path, external_knowledge_file)
        if os.path.exists(knowledge_path):
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return None

    def get_combined_knowledge(self, item, args):
        """Prefer inline knowledge if provided, then optionally append file-based external knowledge.

        - If item contains 'knowledge' (string), include it.
        - If item contains 'external_knowledge' (filename under args.documents_path), append its content.
        - If neither exists, return None.
        """
        parts = []
        file_knowledge = self.load_external_knowledge(item.get('external_knowledge'), args)
        if file_knowledge:
            parts.append(file_knowledge)
        inline_knowledge = item.get('knowledge')
        if inline_knowledge:
            parts.append("**IMPORTANT**:"+str(inline_knowledge).strip())
        if not parts:
            return None
        return "\n\n".join(parts)
    
    
    def build_initial_prompt(self, item, args):
        raise NotImplementedError


    def __init__(self):
        # 只保留DDL schema加载
        self.table_schemas = {}
        schema_dir = '/home/users/xueqi/text2sql/Spider2/spider2-snow/resource/mysqldb/final_algorithm_competition/final_algorithm_competition'
        import os
        if os.path.exists(schema_dir):
            for file in os.listdir(schema_dir):
                if file.endswith('.json'):
                    table_name = file.replace('.json', '')
                    try:
                        with open(os.path.join(schema_dir, file), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            ddl = f"CREATE TABLE `{data['table_name']}` (\n"
                            for i, col in enumerate(data['column_names']):
                                col_type = data['column_types'][i]
                                comment = data['description'][i]
                                ddl += f"  `{col}` {col_type} COMMENT '{comment}',\n"
                            ddl = ddl.rstrip(',\n') + "\n);"
                            self.table_schemas[table_name] = ddl
                    except Exception as e:
                        pass
        
        # 初始化FewshotSelector
        self.fewshot_selector = FewshotSelector(
            gold_path='/home/users/xueqi/text2sql/Spider2/methods/spider-agent-tc/data/golden_sql.jsonl',
            model_path='/home/users/xueqi/text2sql/Spider2/test_fewshot/text2vec-large-chinese'
        )

    def get_database_info(self, db_id, args):
        """Get database directory listing"""
        db_path = os.path.join(args.databases_path, db_id, db_id)
        try:
            result = subprocess.run(['ls', db_path], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception as e:
            return f"Error listing database: {str(e)}"




class SpiderAgentPromptBuilder(BasePromptBuilder):

    def build_initial_prompt(self, item, args):
        system_prompt = self.load_system_prompt(args)
        external_knowledge_content = self.get_combined_knowledge(item, args)
        db_info = self.get_database_info(item['db_id'], args)
        table_list = item.get('table_list') or []
        table_list_block = "\n".join(f"- {t}" for t in table_list) if table_list else "None"
        
        # 获取最相似的 golden item
        best_item = self.fewshot_selector.get_best_golden_item(item['instruction'], table_list)
        
        # 构建Example块
        example_block = ""
        if best_item:
            question = best_item['instruction']
            relevant_tables = best_item.get('table_list', [])
            sql = best_item['sql']
            table_list_str = "\n".join(f"- {t}.json" for t in relevant_tables) if relevant_tables else "None"
            schema_calls = "\n".join(f"<tool_call>\n<function=execute_bash>\n<parameter=command>cat {t}.json</parameter>\n</function>\n</tool_call>" for t in relevant_tables) if relevant_tables else ""
            example_block = f"""=== 示例 ===
问题: "{question}"
相关表:
{table_list_str}

# 第 1 步：查看表结构
{schema_calls}

# 第 2 步：编写并执行 SQL
<tool_call>
<function=execute_mysql_sql>
<parameter=sql>
{sql}
</parameter>
</function>
</tool_call>

# 第 3 步：终止并输出最终 SQL
<tool_call>
<function=terminate>
<parameter=answer>
{sql}
</parameter>
</function>
</tool_call>
=== 示例结束 ==="""
#             example_block = f"""=== Example ===
# Question: "{question}"
# Relevant tables:
# {table_list_str}

# # Step 1: View schema
# {schema_calls}

# # Step 2: Write and run SQL
# <tool_call>
# <function=execute_mysql_sql>
# <parameter=sql>
# {sql}
# </parameter>
# </function>
# </tool_call>

# # Step 3: Finalize
# <tool_call>
# <function=terminate>
# <parameter=answer>
# {sql}
# </parameter>
# </function>
# </tool_call>
# === End Example ==="""
        
        best_table_list = best_item.get('table_list') or [] if best_item else []
        best_table_list_block = "\n".join(f"- {t}" for t in best_table_list) if best_table_list else "None"
        # 添加 schema
        schema_blocks = []
        for table in best_table_list:
            if table in self.table_schemas:
                schema_blocks.append(f"Schema for {table}:\n{self.table_schemas[table]}")
        best_schema_block = "\n\n".join(schema_blocks) if schema_blocks else ""
        # best_external_knowledge = self.get_combined_knowledge(best_item, args)
        best_external_knowledge = best_item.get('knowledge')
        
        # Clarify exact schema folder path to reduce location ambiguity for the model.
        exact_schema_dir = os.path.join(args.databases_path, item['db_id'], item['db_id'])
#         user_content= f"""{example_block}
# 你当前所处的 schema 目录为：({exact_schema_dir})，数据库名称是：{item['db_id']}。

# 问题：
# {item['instruction']}

# 外部知识：
# {external_knowledge_content if external_knowledge_content else 'None'}

# 相关表：
# {table_list_block}
# 以上列表即为本题的全部相关表；不得查看、引用或查询未列出的任何其它表。你可以先查看这些相关表的结构（使用 cat），再开始编写 SQL。

# 现在请帮助我编写能够正确回答该问题的 SQL 查询。"""
        user_content = f"""
You are in the schema folder ({exact_schema_dir}), the database name is {item['db_id']}.

Question: {item['instruction']}
External Knowledge: {external_knowledge_content if external_knowledge_content else 'None'}
Relevant tables:
{table_list_block}
The list above contains all tables relevant to this question; you must not view, reference, or query any tables not listed. You can start by querying the schemas of these related tables.

Now help me write the SQL query to answer the question. """

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]


def get_prompt_builder(strategy):
    builders = {
        "spider-agent": SpiderAgentPromptBuilder(),
        # "database": DatabasePromptBuilder(),
        # "multi_step": MultiStepPromptBuilder(),
        # "reasoning": ReasoningPromptBuilder(),
    }
    
    return builders.get(strategy, SpiderAgentPromptBuilder())


# === Example ===
# Question: "How many rows are in the 'orders' table?"
# Relevant tables:
# orders.json

# # Step 1: View schema
# <tool_call>
# <function=execute_bash>
# <command>cat orders.json</command>
# </function>
# </tool_call>

# # Step 2: Write and run SQL
# <tool_call>
# <function=execute_mysql_sql>
# <sql>
# SELECT COUNT(*) AS row_count FROM orders;
# </sql>
# </function>
# </tool_call>

# # Step 3: Finalize
# <tool_call>
# <function=terminate>
# <answer>
# SELECT COUNT(*) AS row_count FROM orders;
# </answer>
# </function>
# </tool_call>
# === End Example ===