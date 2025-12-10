#!/usr/bin/env python3
"""从 MCTS 输出目录 (results/mysql_exp) 中抽取最终 SQL。

每个 <question_id>.pkl 保存若干 reasoning paths (List[List[MCTSNode]])，
终止节点的属性 final_sql_query 为生成的完整 SQL。

输出示例结构：
[
  {"sql_id": "12_1", "sql": "SELECT ..."},
  {"sql_id": "12_2", "sql": "SELECT ..."}
]

用法：
    python script/extract_mcts_sqls.py \
        --mcts_dir results/mysql_exp \
        --output data/mysql_mcts_pred_exec.json \
        --deduplicate

选项：
    --deduplicate   对同一个 question_id 下生成的 SQL 去重。
    --limit-per-task N  每个任务最多保留 N 条 SQL (0 表示不限)。
"""
import argparse
import json
import pickle
from pathlib import Path

# 需要导入类定义以便正常反序列化
from alphasql.algorithm.mcts.mcts_node import MCTSNode

def extract_sqls(mcts_dir: str, output_path: str, deduplicate: bool, limit_per_task: int):
    mcts_dir_path = Path(mcts_dir)
    if not mcts_dir_path.exists():
        raise FileNotFoundError(f"MCTS 目录不存在: {mcts_dir}")

    all_items = []
    pkl_files = sorted([p for p in mcts_dir_path.glob('*.pkl') if p.stem.isdigit()])
    for pkl_file in pkl_files:
        question_id = pkl_file.stem
        try:
            paths = pickle.load(open(pkl_file, 'rb'))  # paths: List[List[MCTSNode]]
        except Exception as e:
            print(f"解析 {pkl_file} 失败: {e}")
            continue

        sql_collected = []
        for path in paths:
            if not path:
                continue
            end_node = path[-1]
            final_sql = getattr(end_node, 'final_sql_query', None)
            if final_sql and isinstance(final_sql, str):
                sql_text = final_sql.strip()
                if sql_text:
                    sql_collected.append(sql_text)
        if deduplicate:
            seen = set()
            unique_sqls = []
            for s in sql_collected:
                if s not in seen:
                    seen.add(s)
                    unique_sqls.append(s)
            sql_collected = unique_sqls
        if limit_per_task > 0:
            sql_collected = sql_collected[:limit_per_task]

        for idx, sql in enumerate(sql_collected):
            all_items.append({
                'sql_id': f'{question_id}_{idx+1}',
                'sql': sql
            })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_items, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'共收集 SQL {len(all_items)} 条 -> {out_path}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcts_dir', required=True, help='MCTS 输出目录，例如 results/mysql_exp')
    ap.add_argument('--output', required=True, help='输出 JSON 文件路径')
    ap.add_argument('--deduplicate', action='store_true', help='是否去重')
    ap.add_argument('--limit-per-task', type=int, default=0, help='每个任务限制条数 (0 不限)')
    args = ap.parse_args()
    extract_sqls(args.mcts_dir, args.output, args.deduplicate, args.limit_per_task)
