import sys
sys.path.append('.')
from agent.prompt_builders import SpiderAgentPromptBuilder
import json

builder = SpiderAgentPromptBuilder()

# 测试查询
test_instruction = "统计用户（2025.7月24~7月25日持续活跃，2025.7.24活跃7.25流失到盘外），且帐号体系为qq的性别人数"

# 遮罩测试查询
masked_test = builder.mask_instruction(test_instruction)
print('测试查询遮罩后:', masked_test)

# 找到最相似的golden item
best_item = builder.get_best_golden_item(test_instruction)
print('最相似项目ID:', best_item.get('instance_id'))
print('最相似项目指令前100字符:', best_item.get('instruction', '')[:100])
print('最相似项目遮罩后前100字符:', builder.mask_instruction(best_item.get('instruction', ''))[:100])
