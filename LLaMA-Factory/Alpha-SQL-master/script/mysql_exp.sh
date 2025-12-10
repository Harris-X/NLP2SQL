#! /bin/bash

start_time=$(date +%s)

echo "Running MCTS for MySQL..."
python -m alphasql.runner.mcts_runner config/mysql_exp.yaml

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"
