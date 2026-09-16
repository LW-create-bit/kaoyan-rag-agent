import json
import os
import time
import requests
import sqlite3
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI
from agent_prompt import get_system_prompt, get_tools
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)
MODEL = os.getenv("MODEL_ID")
BOCHA_KEY = os.getenv("BOCHA_API_KEY")
# 全局session复用http连接
SESSION = requests.Session()
# 【静态院校层次兜底字典，优先使用，解决院校层次资料不足】
SCHOOL_LEVEL_MAP = {
    "浙江大学": "985",
    "复旦大学": "985",
    "北京大学": "985",
    "清华大学": "985",
    "上海交通大学": "985",
    "南京大学": "985",
    "中国科学技术大学": "985",
    "哈尔滨工业大学": "985",
    "西安交通大学": "985",
    "云南大学": "211",
    "贵州大学": "211",
    "广西大学": "211",
    "海南大学": "211",
    "合肥工业大学": "211",
    "西北大学": "211",
    "昆明理工大学": "双非"
}
# 从agent_prompt加载提示词与工具定义，文件内不再硬编码
SYSTEM_PROMPT = get_system_prompt()
TOOLS_LIST = get_tools()
# chroma初始化，指定embedding函数，禁止自动下载模型
chroma_client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="school_docs", embedding_function=emb_fn)
# ========== SQLite数据库初始化，自动建表/补字段 ==========
def init_db():
    conn = sqlite3.connect("your_school.db")
    cur = conn.cursor()
    # 创建完整表结构
    cur.execute('''
    CREATE TABLE IF NOT EXISTS graduate_school (
        school_name TEXT,
        major TEXT,
        degree_type TEXT,
        re_line INTEGER,
        recruit_num INTEGER,
        province TEXT,
        school_level TEXT,
        year INTEGER,
        PRIMARY KEY (school_name, major, degree_type)
    )
    ''')
    conn.commit()
    conn.close()
    print("[DB] 数据库初始化完成")
# ========== 新增：查询单校结构化硬数据 ==========
def query_school_from_db(school_name: str, major: str, degree_type: str):
    conn = sqlite3.connect("your_school.db")
    cur = conn.cursor()
    sql = """
    SELECT school_level, re_line, recruit_num, year
    FROM graduate_school
    WHERE school_name=? AND major=? AND degree_type=?
    ORDER BY year DESC LIMIT 1
    """
    cur.execute(sql, (school_name, major, degree_type))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "school_name": school_name,
            "school_level": row[0],
            "re_line": row[1],
            "recruit_num": row[2],
            "year": row[3]
        }
    else:
        return None
# ========== 新增：数据库无数据，博查+LLM提取结构化字段并入库 ==========
def fetch_and_save_school_info(school_name: str, major: str, degree_type: str):
    print(f"[AUTO_COLLECT] {school_name} {major} {degree_type} 本地无数据，自动联网提取")
    bocha_url = "https://api.bochaai.com/v1/web-search"
    search_query = f"{school_name} {major} {degree_type} 考研 院校层次 近年复试线 统考招生人数"
    headers = {"Authorization": f"Bearer {BOCHA_KEY}", "Content-Type": "application/json"}
    payload = {"query": search_query, "count": 5, "summary": True}
    try:
        resp = SESSION.post(bocha_url, json=payload, headers=headers, timeout=25)
        web_content = ""
        if resp.status_code == 200:
            res_data = resp.json()
            for page in res_data["data"]["webPages"]["value"]:
                web_content += page["summary"] + "\n"
    except Exception as e:
        print(f"[AUTO_COLLECT WARNING]博查请求失败：{e}")
        web_content = ""
    extract_prompt = """
你是结构化数据提取器，从网页文本提取字段，**只返回纯JSON，禁止任何多余文字、注释**。
字段规则：
school_level：只能是 "985" / "211" / "双非"，找不到填null
re_line：最近一年复试分数线（整数），找不到填null
recruit_num：统考招生人数（整数，不含推免），找不到填null
year：复试线对应的年份，整数，找不到填null
"""
    llm_resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": extract_prompt},
            {"role": "user", "content": web_content}
        ],
        temperature=0
    )
    raw_json = llm_resp.choices[0].message.content.strip()
    info = json.loads(raw_json)
    # 静态字典兜底，如果LLM提取school_level为空，使用本地字典
    if info.get("school_level") is None and school_name in SCHOOL_LEVEL_MAP:
        info["school_level"] = SCHOOL_LEVEL_MAP[school_name]
    # 写入sqlite，主键冲突直接忽略，不会重复入库
    conn = sqlite3.connect("your_school.db")
    cur = conn.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO graduate_school
    (school_name,major,degree_type,re_line,recruit_num,school_level,year)
    VALUES (?,?,?,?,?,?,?)
    """, (
        school_name, major, degree_type,
        info.get("re_line"),
        info.get("recruit_num"),
        info.get("school_level"),
        info.get("year")
    ))
    conn.commit()
    conn.close()
    print(f"[AUTO_COLLECT] {school_name} 结构化数据入库成功")
    return {
        "school_name": school_name,
        "school_level": info.get("school_level"),
        "re_line": info.get("re_line"),
        "recruit_num": info.get("recruit_num"),
        "year": info.get("year")
    }
# ========== 性能计时装饰器 ==========
def timer(func):
    def wrap(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        t_cost = time.time() - t_start
        print(f"\n[TIME] {func.__name__} 执行耗时: {t_cost:.2f} s")
        return res
    return wrap
@timer
def tool_query_graduate_school(major, score, degree_type, undergraduate_level, target_regions):
    conn = sqlite3.connect("your_school.db")
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(target_regions))
    sql = f'''
    SELECT school_name,major,degree_type,re_line,recruit_num,province,school_level,year
    FROM graduate_school
    WHERE major=? AND degree_type=? AND province IN ({placeholders})
    '''
    params = [major, degree_type] + target_regions
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    res = []
    for r in rows:
        res.append({
            "school_name": r[0],
            "major": r[1],
            "degree_type": r[2],
            "re_line": r[3],
            "recruit_num": r[4],
            "province": r[5],
            "school_level": r[6],
            "year": r[7]
        })
    return res

# =====================【优化后的检索函数】=====================
@timer
def tool_retrieve_school_document(school_names, major, degree_type):
    if not school_names:
        return []
    # 一次性批量查询，用元数据过滤多所学校，不再循环多次调用collection.query
    query_text = f"{major} {degree_type} 考研复试经验 压分 报录比 保护一志愿"
    results = collection.query(
        query_texts=[query_text],
        n_results=9,
        where={"school": {"$in": school_names}}
    )
    output = []
    doc_list = results.get("documents", [[]])[0]
    meta_list = results.get("metadatas", [[]])[0]
    for doc, meta in zip(doc_list, meta_list):
        output.append({"fragment": doc, "metadata": meta})
    return output
# ============================================================

# 子线程执行单个学校博查请求，只返回数据，不操作chroma
def _bocha_single_school_task(school, major):
    os.makedirs("documents", exist_ok=True)
    bocha_url = "https://api.bochaai.com/v1/web-search"
    query = f"{school} {major} 考研 复试经验 压分 报录比 保护一志愿"
    payload = {"query": query, "count": 8, "summary": True}
    headers = {"Authorization": f"Bearer {BOCHA_KEY}", "Content-Type": "application/json"}
    try:
        resp = SESSION.post(bocha_url, json=payload, headers=headers, timeout=20)
        if resp.status_code != 200:
            print(f"[WARNING] {school} 博查接口请求失败 status={resp.status_code}")
            return school, None
        data = resp.json()
        # 保存文件
        save_path = os.path.join("documents", f"{school}_{major}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return school, data
    except Exception as e:
        print(f"[WARNING] {school} 请求异常: {str(e)}")
        return school, None
@timer
def tool_bocha_search_school(school_names, major):
    success_schools = []
    all_summaries = []
    all_metas = []
    all_ids = []
    max_workers = 4  # 可调整，推荐3~5，不要过大
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_bocha_single_school_task, school, major)
            for school in school_names
        ]
        for future in as_completed(futures):
            school, data = future.result()
            if not data:
                continue
            # 解析摘要
            summaries = []
            if data.get("data") and data["data"].get("webPages") and data["data"]["webPages"]["value"]:
                for item in data["data"]["webPages"]["value"]:
                    s = item.get("summary", "")
                    if s:
                        summaries.append(s)
            if summaries:
                for idx, text in enumerate(summaries):
                    all_summaries.append(text)
                    all_metas.append({"school": school, "major": major})
                    all_ids.append(f"{school}_{major}_{idx}")
            success_schools.append(school)
    # 主线程统一写入Chroma，避免多线程并发add
    if all_summaries:
        collection.add(
            documents=all_summaries,
            metadatas=all_metas,
            ids=all_ids
        )
    return {"success_schools": success_schools}
@timer
def bocha_search_by_province_major(province_list, major, degree_type, score):
    os.makedirs("documents", exist_ok=True)
    bocha_url = "https://api.bochaai.com/v1/web-search"
    province_str = " ".join(province_list)
    query = f"{province_str} {major} {degree_type} 考研院校 复试线 招生人数 报录比 压分 保护一志愿"
    payload = {"query": query, "count": 12, "summary": True}
    headers = {"Authorization": f"Bearer {BOCHA_KEY}", "Content-Type": "application/json"}
    try:
        resp = SESSION.post(bocha_url, json=payload, headers=headers, timeout=20)
        if resp.status_code != 200:
            print(f"[WARNING]省份维度博查请求失败")
            return []
        data = resp.json()
        save_path = os.path.join("documents", f"prov_{province_str}_{major}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        summaries = []
        if data.get("data") and data["data"].get("webPages") and data["data"]["webPages"]["value"]:
            for item in data["data"]["webPages"]["value"]:
                s = item.get("summary", "")
                if s:
                    summaries.append(s)
        return summaries
    except Exception as e:
        print(f"[WARNING]省份搜索异常：{e}")
        return []
def handle_tool_call(tool_call):
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    if func_name == "query_graduate_school":
        return tool_query_graduate_school(**args)
    elif func_name == "retrieve_school_document":
        return tool_retrieve_school_document(**args)
    elif func_name == "bocha_search_school":
        return tool_bocha_search_school(**args)
    else:
        return {"error": f"未知工具:{func_name}"}
def chat_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("==== KaoYan‑Agent 考研择校智能体 ====")
    print("输入你的考研信息，输入 exit 退出\n")
    while True:
        user_input = input("用户：")
        user_input = user_input.strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            assistant_msg_dict = {
                "role": "assistant",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                "content": msg.content if msg.content else None
            }
            messages.append(assistant_msg_dict)
            sql_tool_result = None
            for tc in msg.tool_calls:
                print(f"\n[DEBUG]执行工具：{tc.function.name}")
                tool_result = handle_tool_call(tc)
                print(f"[DEBUG]工具返回结果：{json.dumps(tool_result, ensure_ascii=False)[:400]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })
                if tc.function.name == "query_graduate_school":
                    sql_tool_result = tool_result
            # =====分支1：SQL查询拿到院校数据=====
            if sql_tool_result and len(sql_tool_result) > 0:
                school_name_list = [item["school_name"] for item in sql_tool_result]
                major_name = sql_tool_result[0]["major"]
                degree_type_name = sql_tool_result[0]["degree_type"]
                # =========【核心新增：遍历院校，补齐数据库硬字段，自动采集】========
                school_struct_data = []
                for sch in sql_tool_result:
                    db_data = query_school_from_db(sch["school_name"], major_name, degree_type_name)
                    if db_data is None:
                        db_data = fetch_and_save_school_info(sch["school_name"], major_name, degree_type_name)
                    school_struct_data.append(db_data)
                # =====================================================
                print(f"\n[DEBUG]【代码强制】调用博查搜索，院校列表：{school_name_list}")
                bocha_ret = tool_bocha_search_school(school_name_list, major_name)
                print(f"[DEBUG]博查完成，成功搜索院校：{bocha_ret['success_schools']}")
                print(f"[DEBUG]【代码强制】调用向量库检索")
                retrieve_ret = tool_retrieve_school_document(school_name_list, major_name, degree_type_name)
                print(f"[DEBUG]向量检索拿到文档片段数量：{len(retrieve_ret)}")
                messages.append({
                    "role": "user",
                    "content": f"""
【院校结构化硬数据（数据库，优先填充表格）】
{json.dumps(school_struct_data, ensure_ascii=False)}
【外部网页检索信息】
{json.dumps(retrieve_ret, ensure_ascii=False)}
任务：结合上面数据库院校硬数据 + 网页检索得到的压分、报录比、复试口碑等软信息，输出完整🔴冲档 🟡稳档 🟢保档 Markdown择校报告。
> 重点：对比表格优先使用【院校结构化硬数据】填充：院校层次、近年复试线、招生人数。
> 数据库有值就直接填入表格，不要写【资料不足】；只有硬数据为空才写【资料不足】。
报告每个学校补充网页获取到的风险点、复试情况。不要再调用任何工具，直接输出完整回答。
"""
                })
            # =====分支2：SQL返回空列表，没有本地硬数据，直接全网搜索省份+专业=====
            else:
                print("\n[DEBUG]SQL数据库无本地院校数据，触发省份维度全网搜索")
                # 从用户原始问题提取信息交给博查
                # 复用模型解析出来的参数，从第一条tool_call拿参数
                first_tc = msg.tool_calls[0]
                args = json.loads(first_tc.function.arguments)
                province_list = args.get("target_regions", [])
                major = args.get("major", "")
                degree_type = args.get("degree_type", "")
                score = args.get("score", "")
                web_docs = bocha_search_by_province_major(province_list, major, degree_type, score)
                print(f"[DEBUG]省份全网搜索获取网页摘要：{len(web_docs)}条")
                messages.append({
                    "role": "user",
                    "content": f"""
⚠️本地SQL数据库没有该省份院校硬数据。
【全网网页搜索信息】
{json.dumps(web_docs, ensure_ascii=False)}
任务：仅基于上面网页搜索到的公开考研信息，输出🔴冲档 🟡稳档 🟢保档Markdown择校报告。
注意：没有本地数据库复试线、招生名额硬数据，所有数据全部来自网页，需要标注信息来源为网络公开资料，提醒数据仅供参考。不要再调用任何工具，直接输出完整回答。
"""
                })
            print("[DEBUG]发起最终LLM生成报告请求...")
            final_resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3
            )
            final_content = final_resp.choices[0].message.content
            print(f"\nAgent:\n{final_content}\n")
            messages.append({"role": "assistant", "content": final_content})
        else:
            print(f"\nAgent:\n{msg.content}\n")
            messages.append({"role": "assistant", "content": msg.content})
if __name__ == "__main__":
    init_db()
    chat_loop()
