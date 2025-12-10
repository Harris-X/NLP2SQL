# export OPENAI_API_BASE="https://api.minimaxi.com/v1"  # Example alternative provider
# export OPENAI_API_KEY="..."  # Move real key to .env instead of committing.
export OPENAI_API_BASE="https://hk-api.gptbest.vip/v1"  # Base only; llm_agent decides chat vs completions.
export OPENAI_API_KEY="sk-76NgizbdFnDhy6ITJ58bLlnrlIeZjW9r3HpZzf0lv1cRPsps"  # TODO: externalize
# INPUT_FILE="../../spider2-snow/spider2-snow.jsonl"
INPUT_FILE=${INPUT_FILE:-"/home/users/xueqi/text2sql/Spider2/methods/spider-agent-tc/data/sql24.jsonl"}
OUTPUT_FOLDER="./results/sql24_claude"
SYSTEM_PROMPT="./prompts/spider_agent.txt"
DATABASES_PATH="/home/users/xueqi/text2sql/Spider2/spider2-snow/resource/mysqldb"  # MUST fill your own absolute path
DOCUMENTS_PATH="/home/users/xueqi/text2sql/Spider2/spider2-snow/resource/common_knowledge"

# MODEL="MiniMax-M2"  # Chat model example
# MODEL="qwen3-coder-480b-a35b-instruct"  # Chat model example
# MODEL="grok-4.1"
# MODEL="gemini-3-pro-preview"
MODEL="claude-opus-4-5-20251101-thinking"
# MODEL="gpt-5.1"  # Completion-only; llm_agent will auto-switch endpoint.
TEMPERATURE=0.1
# For reproducible, non-sampling decoding, keep TEMPERATURE=0 and set TOP_P=1.0
TOP_P=1.0
MAX_NEW_TOKENS=24000
MAX_ROUNDS=30
NUM_THREADS=16
ROLLOUT_NUMBER=2
# EXPERIMENT_SUFFIX="exp3" 

# OUTPUT_FOLDER="./results/${MODEL}_temp${TEMPERATURE}_rounds${MAX_ROUNDS}_rollout${ROLLOUT_NUMBER}_${EXPERIMENT_SUFFIX}"
# mkdir -p "./results"

echo "Output file will be: $OUTPUT_FOLDER"



host=$(hostname -I | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m servers.serve --workers_per_tool 32 --host $host --port $port  &
server_pid=$!

echo "Server (pid=$server_pid) started at $tool_server_url"

sleep 3

python agent/main.py \
    --input_file "$INPUT_FILE" \
    --output_folder "$OUTPUT_FOLDER" \
    --system_prompt_path "$SYSTEM_PROMPT" \
    --databases_path "$DATABASES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --api_host "$host" \
    --api_port "$port" \
    --max_rounds "$MAX_ROUNDS" \
    --num_threads "$NUM_THREADS" \
    --rollout_number "$ROLLOUT_NUMBER"