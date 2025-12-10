import json

# 手动修复已知的问题
fixes = {
    '1350': {
        'question_cn': '购买"明信片、海报"的事件的状态是什么，购买日期为2019年8月20日？',
        'evidence_cn': "'Post Cards, Posters'是一个expense_description；2019/8/20指的是expense_date = '2019-8-20'；事件的状态指的是event_status"
    },
    '1352': {
        'question_cn': '对于来自"Business"专业的所有俱乐部成员，其中有多少人穿中号T恤？',
        'evidence_cn': "'Business'是一个专业名称；穿中号T恤指的是t_shirt_size = 'Medium'"
    }
}

# 读取非JSON输出并手动解析
with open('datasets/minidev/non_json_outputs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        qid = data['qid']
        if qid in fixes:
            question_cn = fixes[qid]['question_cn']
            evidence_cn = fixes[qid]['evidence_cn']
            print(f'Manually parsed qid={qid}: question_cn={question_cn[:50]}..., evidence_cn={evidence_cn[:50]}...')
            # 更新缓存
            cache_entry = {
                qid: {
                    'question_cn': question_cn,
                    'evidence_cn': evidence_cn
                }
            }
            with open('datasets/minidev/translation_cache.json', 'r+', encoding='utf-8') as cache_f:
                cache = json.load(cache_f)
                cache.update(cache_entry)
                cache_f.seek(0)
                json.dump(cache, cache_f, ensure_ascii=False, indent=2)
                cache_f.truncate()
            print(f'Updated cache for qid={qid}')
        else:
            print(f'No manual fix for qid={qid}')