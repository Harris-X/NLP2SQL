for f in /home/users/xueqi/text2sql/Spider2/spider2-snow/evaluation_suite/sql1-10_submission/*.sql; do
  base=$(basename "$f" .sql)
  python /home/users/xueqi/text2sql/Spider2/methods/spider-agent-tc/sql_exe.py \
    --sql_file "$f" \
    --out "/home/users/xueqi/text2sql/Spider2/spider2-snow/evaluation_suite/sql1-10_submission/${base}_result.json" \
    --credentials /home/users/xueqi/text2sql/Spider2/methods/spider-agent-tc/credentials/mysql_credential.json
done