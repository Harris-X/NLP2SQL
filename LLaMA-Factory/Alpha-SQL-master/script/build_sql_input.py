#!/usr/bin/env python3
"""从 Alpha-SQL 数据集中提取 SQL 生成执行输入文件。

输入示例文件: mini_dev_mysql_alpaca_filtered.json
输出示例文件结构: [ {"sql_id": "sql_1", "sql": "SELECT ..."}, ... ]

用法:
    python script/build_sql_input.py \
        --source mini_dev_mysql_alpaca_filtered.json \
        --target data/mini_dev_mysql_alpaca_filtered_exec.json

可选参数:
    --limit 50    只取前 50 条，便于快速测试
    --drop-semicolon True 移除末尾分号 (默认保留)
"""
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='提取数据集中的 SQL 生成批量执行输入文件')
    parser.add_argument('--source', required=True, help='源数据集 JSON 路径 (含 output 字段)')
    parser.add_argument('--target', required=True, help='生成的执行输入 JSON 路径')
    parser.add_argument('--limit', type=int, default=0, help='限制输出条数 (0 表示全部)')
    parser.add_argument('--drop-semicolon', action='store_true', help='移除 SQL 末尾分号')

    args = parser.parse_args()

    src_path = Path(args.source)
    if not src_path.exists():
        raise FileNotFoundError(f'源文件不存在: {src_path}')

    data = json.loads(src_path.read_text(encoding='utf-8'))
    results = []
    for i, item in enumerate(data):
        if args.limit and i >= args.limit:
            break
        sql_raw = item.get('output') or ''
        if not sql_raw.strip():
            continue
        sql_clean = sql_raw.strip()
        if args.drop_semicolon:
            sql_clean = sql_clean.rstrip(';')
        results.append({
            'sql_id': f'sql_{i+1}',
            'sql': sql_clean
        })

    tgt_path = Path(args.target)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'生成完成: {tgt_path} 共 {len(results)} 条记录')

if __name__ == '__main__':
    main()
