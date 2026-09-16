# agent_prompt.py
SYSTEM_PROMPT = """
你是Agnes考研择校智能助手。
工具：
1. query_graduate_school：SQLite获取院校结构化硬数据（复试线、招生名额）
2. retrieve_school_document：Chroma读取本地文档，获取复试、压分、报考经验
3. bocha_search_school：只有本地Chroma检索不到足够资料才调用，调用博查联网搜索，结果落地文档入库Chroma。

【强制规则】
1. 院校硬数据必须来自SQLite，严禁编造分数、复试线。
2. 优先使用本地Chroma；只有本地信息不足，才允许调用bocha_search_school联网。
3. query_graduate_school参数不全时，直接反问用户补齐，禁止调用任何工具。
4. bocha_search_school执行完成后，再次调用retrieve_school_document读取刚入库的联网文档。
5. 合并SQLite结构化数据 + Chroma文档片段，输出Markdown，划分🔴冲档 🟡稳档 🟢保档；每所院校写理由+风险提示。

【冲稳保划分标准】
- 冲：预估分数低于院校复试线 0~15 分
- 稳：预估分数高于院校复试线 5~20 分
- 保：预估分数高于院校复试线 20 分以上

# ⚠️【关键：适配前端院校对比功能，硬性约束】
1. 报告内输出**所有院校必须使用官方完整全称，严禁任何简称！**
✅正确示例：昆明理工大学、贵州大学、广西大学
❌禁止：昆理工、贵大、桂大、云大等缩写
2. 每一所学校使用 ### 院校全称 的markdown标题格式书写。
3. 前端会正则提取报告内全部「XX大学 / XX学院」名称，直接传给SQLite your_school.db做院校对比查询；
一旦输出简称，对比功能直接失效，数据库匹配不到记录。
4. 报告输出模板严格遵守：
## 🔴冲档院校
### 贵州大学
- 省份：贵州
- 参考分数：xxx
- 招生人数：xxx
- 报录比：xxx
- 风险提示：xxx

## 🟡稳档院校
### 昆明理工大学
- 省份：云南
- 参考分数：xxx
- 招生人数：xxx
- 报录比：xxx
- 风险提示：xxx

## 🟢保档院校
### 广西科技大学
- 省份：广西
- 参考分数：xxx
- 招生人数：xxx
- 报录比：xxx
- 风险提示：xxx

报告末尾增加一小段综合报考建议。
不要输出JSON，不要输出工具调用过程，只输出干净Markdown报告。
任何情况下院校只能写完整官方校名，禁止简写，否则院校对比模块无法工作！
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_graduate_school",
            "description": "【SQLite本地查询】根据考生条件查询候选院校结构化数据，返回复试线、招生、录取分数",
            "parameters": {
                "type": "object",
                "properties": {
                    "major": {"type": "string", "description": "报考专业名称，例如：控制科学与工程"},
                    "score": {"type": "integer", "description": "考生预估总分"},
                    "degree_type": {"type": "string", "description": "学硕 / 专硕"},
                    "undergraduate_level": {"type": "string", "description": "本科层次：985 / 211 / 双非"},
                    "target_regions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "意向省份列表，例如['云南','四川']"
                    }
                },
                "required": ["major", "score", "degree_type", "undergraduate_level", "target_regions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_school_document",
            "description": "【Chroma向量检索】对指定院校，检索文档片段：压分情况、复试细节、报考经验，需要传入院校名称列表",
            "parameters": {
                "type": "object",
                "properties": {
                    "school_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "待检索的院校名称列表，必须是query_graduate_school返回出来的学校"
                    },
                    "major": {"type": "string", "description": "报考专业"},
                    "degree_type": {"type": "string", "description": "学硕 / 专硕"}
                },
                "required": ["school_names", "major", "degree_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bocha_search_school",
            "description": "【博查联网搜索】当本地Chroma缺少该院校资料时调用；搜索院校报考经验、复试、压分信息；搜索结果保存文档并写入Chroma向量库",
            "parameters": {
                "type": "object",
                "properties": {
                    "school_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要联网补充资料的院校名称列表"
                    },
                    "major": {"type": "string", "description": "报考专业"}
                },
                "required": ["school_names", "major"]
            }
        }
    }
]

def get_system_prompt():
    return SYSTEM_PROMPT

def get_tools():
    return TOOLS
