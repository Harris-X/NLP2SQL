import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def add_key(url):
    data = load_json(url)
    for item in data:
        item['instruction'] = "请你接下来一步步思考，写出正确的SQL查询语句以满足用户的需求。"
    with open('data_updated.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    url = "datasets/minidev/correct_answers.json"
    add_key(url)