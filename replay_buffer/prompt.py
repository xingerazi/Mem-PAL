# InteractionUnit_Extractor_Prompt222="""
# 你是一个 InteractionUnit 抽取器。

# 你的任务是：
# 从给定的一段原始对话记录 / agent 执行日志中，
# 抽取用于构建 InteractionUnit 的【语义结果】。

# 系统会在后处理中自动补充 ID、时间戳和完整的 action 结构，
# 你只需要输出下面定义的字段。

# --------------------------------------------------
# 【你需要输出的 JSON 结构（必须严格遵守）】

# {
#   "topic": string,

#   "user_query": string,

#   "agent_action_dia_ids": [string],

#   "agent_reply": string,

#   "user_feedback": {
#     "type": "confirm" | "reject" | "revise",
#     "content": string
#   },

#   "success": boolean,

#   "insight": string | null
# }

# --------------------------------------------------
# 【字段说明与强约束】

# 1. topic
# - 用 1–3 个英文单词概括本次交互主题
# - 示例：diet_recommendation / travel_planning / course_planning

# 2. user_query
# - 用户最初提出、触发本次 agent 决策的那一句query
# - 使用原话或忠实改写
# - 不要包含 agent 的话

# 3. agent_action_dia_ids
# - 一个字符串数组
# - 表示 agent 在本次交互中【发生工具调用（tool call）】的对话轮次 ID
# - 只包含真正调用了工具的轮次
# - 示例：["D3:1", "D4:2"]
# - 如果本次交互中没有任何工具调用，返回 []

# 4. agent_reply
# - agent 给用户的最终自然语言回复,可以精简
# - 只保留最终版本

# 5. user_feedback
# - 用户对 agent_reply 的裁决
# - type 只能是：
#   - confirm：用户明确接受
#   - reject：用户明确否决
#   - revise：用户部分否定并给出修正
# - content 保留用户原话或等价改写
# - 追问或继续聊天 ≠ feedback（不要误判）

# 6. success
# - confirm 或 revise → true
# - reject → false

# 7. insight
# - 用一句话总结本次交互中【值得被记住或强化的地方】
# - 可以是：
#   - 暴露出的可学习问题（如记忆误用、画像过期、推理依据不合理）
#   - 也可以是一次做对了的关键点（如正确使用了某条记忆或约束）
# - 只能涉及：
#   - 记忆的使用情况
#   - 用户画像的匹配情况
#   - 推理依据与约束是否合理
# - 禁止：
#   - 对话策略建议
#   - “应该追问用户”
#   - “可以多给选项”
# - 如果本次交互没有明显值得记录的点，返回 null

# --------------------------------------------------
# 【重要禁止项】

# - 不要输出未定义字段
# - 不要解释你的推理过程
# - 不要输出 JSON 以外的任何文字

# --------------------------------------------------
# 【输入：原始对话 / 日志】

# {{RAW_DIALOGUE}}

# --------------------------------------------------
# 【输出：仅输出一个 JSON 对象】

# """




# InteractionUnit_Extractor_Prompt="""
# 你是一个 Interaction Unit（IU）抽取器。

# 输入是一段 dialogue JSON（turn_1, turn_2 ...）。
# 一段 dialogue 中可能包含 1 个或多个 Interaction Unit。

# 【Interaction Unit 定义】
# 当 assistant 给出了一个“可以被用户评价是否接受、修正或拒绝的回应”，
# 并且用户随后对该回应表达了态度，就构成 1 个 Interaction Unit。

# 请你按对话顺序，抽取所有符合上述定义的 Interaction Unit。
# 每个 IU 独立抽取，不要合并。

# 对每一个 Interaction Unit，请生成以下字段：

# - topic：
#   用户当前关心的语义领域标签。
#   使用一个词或一个简短短语（不是完整句子）。

# - user_query：
#   本 IU 所回应的用户明确需求句。
#   使用用户在对话中的原话或轻微整理后的原句。

# - agent_actions：
#   assistant 在回应该需求时使用的工具调用。
#   如果没有工具使用，返回空数组 []。

# - agent_reply：
#   assistant 针对该 user_query 给出的核心回应。
#   合并为一句自然语言，不要包含多余解释。

# - user_feedback：
#   - type：从以下三种中选择其一：
#       - confirm：用户明确接受或认可该回应
#       - reject：用户明确否定或不接受该回应
#       - revise：用户在基本接受的基础上提出修改、补充或新约束
#   - content：用户对该回应表达态度的原话

# - insight：
#   用一句话概括本次 IU 中暴露出的、对未来个性化或决策有价值的新信息。
#   如果没有明显的新信息，返回 null。

# 只输出 JSON，不要输出任何解释性文字。
# 输出格式如下：

# {{
#   "interaction_units": [
#     {{
#       "topic": "...",
#       "user_query": "...",
#       "agent_actions": [],
#       "agent_reply": "...",
#       "user_feedback": {{
#         "type": "confirm | reject | revise",
#         "content": "..."
#       }},
#       "insight": null
#     }}
#     {{
#       ...
#     }}
#   ]
# }}

# 对话如下:
# {dialogue}
# """


SEGMENT_AND_SIMPLIFY_PROMPT = """
你是一个对话分割与轻度重写器。

输入是一段 dialogue JSON（turn_1, turn_2 ...），
其中可能包含多个不同的用户问题。

你的任务包括两个步骤：

=====================
【任务一：按用户问题分割】
=====================

- 以“用户提出的明确问题或需求”为分割依据。
- 同一个问题下的追问、澄清、讨论，视为同一问题段。
- 当用户提出一个新的问题、或明显切换关注点时，开始一个新的问题段。
- 不要按 turn 机械切分。

=====================
【任务二：对内容进行轻度语义简写】
=====================

对每个问题段中的文本进行“轻度简写”，要求：

- 删除语气词、口语填充（如“没错”“我觉得”“可能”“嗯”“就是说”等）
- 删除重复表达
- 保留原有语义与关键信息
- 不引入新信息
- 不做抽象总结
- 保持自然语言句子（不是关键词列表）

示例：
原句：
“没错，我觉得如果能有一个明确的工具来跟踪任务进度，就能减少这些问题。”
简写后：
“如果有一个工具来跟踪任务进度，就能减少这些问题。”

=====================
【对每一个用户问题段，请输出：】
=====================

- topic：
  用户关心的语义领域标签（一个词或简短短语）

- user_question：
  该问题段中用户提出的核心问题（轻度简写）

- assistant_response：
  assistant 针对该问题给出的主要回应（轻度简写）

- user_response：
  用户对 assistant 回应的主要反馈或态度（轻度简写）

=====================
【输出要求】
=====================

- 只输出 JSON
- 不输出任何解释性文字
- 输出格式如下：

{{
  "segments": [
    {{
      "topic": "...",
      "user_question": "...",
      "assistant_response": "...",
      "user_response": "..."
    }}
  ]
}}

对话如下：
{dialogue}
"""


DIALOGUE_SEGMENT_PROMPT = """
你是一个对话分段器。

输入是一段 dialogue JSON（turn_1, turn_2 ...），
每个 turn 包含 user 和 assistant 的发言。

你的任务是：

【分段规则】
- 按“用户提出的明确问题（query）”进行分段。
- 同一个用户问题下的澄清、讨论、追问，属于同一段。
- 当用户提出一个新的问题或明显切换关注点时，开始一个新段。
- 不要按 turn 数量机械分段。

【输出要求】
- 只输出每一段对应的 turn 范围。
- 使用 turn id 表示范围，例如："turn_1–turn_5"。
- 保持原始 turn 顺序。

【输出格式】
只输出 JSON，格式如下：

{{
  "segments": [
    {{
      "segment_id": 1,
      "turn_range": "turn_1–turn_5"
    }},
    {{
      "segment_id": 2,
      "turn_range": "turn_6–turn_11"
    }}
  ]
}}

对话如下：
{dialogue}
"""






















InteractionUnit_Build_Prompt  = """
你将获得一段【已经切好的】用户与助手之间的多轮对话片段。
你的任务是：从这段对话中，总结抽取以下四个字段：
- topic
- user_query
- user_feedback
- insight

字段说明与约束如下：

1. topic  
概括对话的核心主题。  
- 从以下四个四选一："work"、"health"、"family"、"leisure" 

2. user_query  
用户在本次对话中表达的初始需求(也就是第一句话)。  
- 尽量使用用户的原话,适当修改语气词或者助词等

3. user_feedback  
判断用户对助手给出的解决方案的总体反馈信号，三选一：
- "confirm"：用户明确接受或认同
- "reject"：用户明确反对、不认可、否定建议
- "revise"：用户部分接受，但要求修改、补充或调整

4. insight  
用【一句话】指出本次交互中暴露出的“可学习问题”。  

必须严格遵守以下规则：
- insight 必须是【导致本次结果的关键因素】，可以是：
  - 导致失败的原因
  - 或导致成功的关键
示例 1:失败案例:用户：我中午吃什么好？助手：那你可以吃红烧肉。用户：我最近在减肥，不能吃红烧肉。
insight:助手忽略了用户近期的偏好和限制，导致推荐内容与当前需求不匹配。
示例 2:成功案例:用户：我这个周末能不能去英国玩？助手：你目前只有申根签证，不适合去英国，建议在申根区内旅行。用户：确实，我忘了。
insight:助手正确使用了用户的相关信息，在签证方面找到关键。
【重要约束】：
- 只能输出 JSON，不要输出任何解释性文字
- 不要添加额外字段
【输出格式】
只输出 JSON，格式如下：

{{
  "topic": "work",
  "user_query": "我最近在准备职业英语口语考试，有没有什么提高口语的方法？",
  "user_feedback": "confirm",
  "insight": "助手提供的建议与用户当前的学习目标高度一致，是本次交互成功的关键。"
}}

对话如下：
{dialogue}
"""




























