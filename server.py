"""考研择校智能体 · FastAPI 薄接口层
改动：移除独立院校库分页接口；对比仅用于智能体输出的候选院校对比
1. **不修改 kaoYan_agent.py 的任何一行**
2. 院校对比：接收学校名称列表，查询 your_school.db 返回对比表格
3. AI学习计划接口保留
"""
import os
import sys
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# ========== 放到 server.py 的全局区域，import下面 ==========
import time
CACHE_DB = "bocha_cache.db"
def init_bocha_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS bocha_cache (
        school_name TEXT,
        major TEXT,
        result_json TEXT,
        create_ts INTEGER,
        PRIMARY KEY (school_name, major)
    )
    ''')
    conn.commit()
    conn.close()

def get_bocha_cache(school_name:str, major:str, expire_days=7):
    expire_ts = int(time.time()) - expire_days * 24 * 3600
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("SELECT result_json FROM bocha_cache WHERE school_name=? AND major=? AND create_ts>?",
                (school_name, major, expire_ts))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def save_bocha_cache(school_name:str, major:str, data):
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute('''
    INSERT OR REPLACE INTO bocha_cache(school_name, major, result_json, create_ts)
    VALUES (?,?,?,?)
    ''', (school_name, major, json.dumps(data, ensure_ascii=False), int(time.time())))
    conn.commit()
    conn.close()

# 在app启动的时候初始化缓存表
init_bocha_cache_db()


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import kaoYan_agent  # noqa: E402

app = FastAPI(title="考研择校智能体接口", version="1.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- 原有模型定义 ----------------------
class QueryRequest(BaseModel):
    query: str
    score: Optional[int] = None
    undergraduate_level: Optional[str] = None
    major: Optional[str] = None
    degree_type: Optional[str] = None
    target_regions: Optional[List[str]] = None

class LogItem(BaseModel):
    time: str
    step: str
    detail: str = ""

class QueryResponse(BaseModel):
    log_list: List[LogItem]
    report_markdown: str

class LogCollector:
    def __init__(self):
        self.items: List[LogItem] = []
    def __call__(self, step: str, detail: str = ""):
        self.items.append(
            LogItem(
                time=datetime.now().strftime("%H:%M:%S"),
                step=step,
                detail=str(detail)[:300],
            )
        )
        print(f"[{self.items[-1].time}] {step} | {self.items[-1].detail}")

# 一次请求最多对几所院校做联网补信息（本地库命中很多时，逐所联网会把耗时线性放大）
MAX_BOCHA_SCHOOLS = 3
# 博查并发线程数
BOCHA_MAX_WORKERS = 4


def _pick_key_schools(rows, score, limit=MAX_BOCHA_SCHOOLS):
    """按「与考生分数的差距」挑出最值得联网补信息的院校。

    差距越小越关键（冲/稳档的判断全靠它），差距过大的保档院校没必要
    再花一次网络往返去查。
    """
    def gap(r):
        try:
            return abs(int(score) - int(r.get("re_line", 0)))
        except (TypeError, ValueError):
            return 10 ** 6
    return sorted(rows, key=gap)[:limit]


def _bocha_search_concurrent(school_names, major, max_workers=None):
    if not school_names:
        return {"success_schools": [], "failed_schools": []}
    # 先筛选：命中缓存的直接拿，剩下的才请求博查
    cached_map = {}
    need_search = []
    for name in school_names:
        cache_data = get_bocha_cache(name, major)
        if cache_data is not None:
            cached_map[name] = cache_data
        else:
            need_search.append(name)
    ok, failed = list(cached_map.keys()), []
    # 只有不在缓存的学校才去联网搜索
    if len(need_search) > 0:
        workers = min(max_workers or BOCHA_MAX_WORKERS, len(need_search))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(kaoYan_agent.tool_bocha_search_school, [name], major): name
                for name in need_search
            }
            for fut in as_completed(future_map):
                name = future_map[fut]
                try:
                    ret = fut.result() or {}
                    if name in (ret.get("success_schools") or []):
                        ok.append(name)
                        save_bocha_cache(name, major, ret)
                    else:
                        failed.append(name)
                except Exception as e:
                    failed.append(name)
                    print(f"[WARNING] 博查并发任务失败 {name}: {type(e).__name__}: {e}")
    return {"success_schools": ok, "failed_schools": failed}


def _build_search_context(school_names, major, degree_type="", max_items=8, per_len=200):
    """组装「联网信息」上下文，并严格控制长度。

    注意：tool_bocha_search_school 的返回值只有「成功院校名单」，
    真正的网页摘要是被它写进 chroma 的 —— 所以必须从向量库取回来，
    否则 prompt 里的「网页检索信息」其实是个空壳，模型只能靠自己的知识编。

    同时做截断：最多 max_items 条、每条 per_len 字，避免长上下文拖慢推理。
    """
    try:
        docs = kaoYan_agent.tool_retrieve_school_document(school_names, major, degree_type)
    except Exception as e:
        print(f"[WARNING] 向量检索失败: {type(e).__name__}: {e}")
        return "（未检索到网页片段）"

    fragments = []
    for d in docs or []:
        text = (d.get("fragment") or "").strip().replace("\n", " ")
        if text:
            fragments.append(text[:per_len])
    if not fragments:
        return "（未检索到网页片段）"
    return "\n".join(f"- {t}" for t in fragments[:max_items])


def run_agent_once(user_input: str, log: LogCollector) -> str:
    a = kaoYan_agent
    messages = [{"role": "system", "content": a.SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_input})
    log("LLM 意图解析", "判断是否需要调用工具")
    response = a.client.chat.completions.create(
        model=a.MODEL,
        messages=messages,
        tools=a.TOOLS_LIST,
        temperature=0.3,
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        log("完成", "模型未调用工具（可能正在追问缺失信息）")
        return msg.content or "（模型本次没有返回内容）"
    messages.append({
        "role": "assistant",
        "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        "content": msg.content if msg.content else None,
    })
    sql_tool_result = None
    sql_args = {}
    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        log("工具调用", f"{name} 参数={json.dumps(args, ensure_ascii=False)[:160]}")
        tool_result = a.handle_tool_call(tc)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(tool_result, ensure_ascii=False),
        })
        if name == "query_graduate_school":
            sql_tool_result = tool_result
            sql_args = args
            hit = len(tool_result) if tool_result else 0
            log("数据库查询", f"SQLite 本地库命中 {hit} 所院校")
    if sql_tool_result and len(sql_tool_result) > 0:
        school_names = [item["school_name"] for item in sql_tool_result]
        major_name = sql_tool_result[0].get("major", "")
        degree_type = sql_tool_result[0].get("degree_type", "")
        key_schools = _pick_key_schools(sql_tool_result, sql_args.get("score") or 0)
        key_names = [s["school_name"] for s in key_schools]

        log(
            "博查联网搜索",
            f"命中 {len(school_names)} 所，为控制耗时仅对最匹配的 {len(key_names)} 所联网"
            f"（并发 {min(BOCHA_MAX_WORKERS, max(len(key_names), 1))} 线程）：{'、'.join(key_names)}",
        )
        bocha_ret = _bocha_search_concurrent(key_names, major_name)
        ok_schools = bocha_ret.get("success_schools", []) if isinstance(bocha_ret, dict) else []
        failed_schools = bocha_ret.get("failed_schools", []) if isinstance(bocha_ret, dict) else []
        log(
            "博查联网搜索",
            f"成功 {len(ok_schools)} 所：{'、'.join(ok_schools) or '无'}"
            + (f"；失败 {len(failed_schools)} 所：{'、'.join(failed_schools)}" if failed_schools else ""),
        )

        log("向量检索", "从 Chroma 检索复试/压分/报录比相关片段")
        retrieve_ret = a.tool_retrieve_school_document(key_names, major_name, degree_type)
        log("向量检索", f"拿到 {len(retrieve_ret)} 条文档片段")
        messages.append({
            "role": "user",
            "content": f"""
【外部网页检索信息】
（注：以下网页片段仅覆盖院校：{'、'.join(key_names)}；本地库中的其余院校没有联网信息，
请只依据其本地库硬数据分析，不要为它们编造任何网络来源的信息）
{json.dumps(retrieve_ret, ensure_ascii=False)}
任务：结合上面数据库院校硬数据 + 网页检索得到的压分、报录比、复试口碑等软信息，输出完整🔴冲档 🟡稳档 🟢保档 Markdown择校报告。
报告每个学校补充网页获取到的风险点、复试情况。不要再调用任何工具，直接输出完整回答。
"""
        })
    else:
        try:
            first_args = json.loads(msg.tool_calls[0].function.arguments)
        except json.JSONDecodeError:
            first_args = {}
        province_list = first_args.get("target_regions", [])
        major = first_args.get("major", "")
        degree_type = first_args.get("degree_type", "")
        score = first_args.get("score", "")
        log("数据库查询", "本地库无命中，改用省份维度全网搜索")
        web_docs = a.bocha_search_by_province_major(province_list, major, degree_type, score)
        log("博查联网搜索", f"省份维度搜索拿到 {len(web_docs)} 条网页摘要")
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
    log("LLM 生成报告", "综合硬数据 + 软信息，输出冲稳保报告")
    final_resp = a.client.chat.completions.create(
        model=a.MODEL,
        messages=messages,
        temperature=0.3,
    )
    content = final_resp.choices[0].message.content or "（模型没有返回内容）"
    log("完成", f"报告生成完毕，共 {len(content)} 字")
    return content

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    log = LogCollector()
    log("接收请求", req.query[:80])
    try:
        report = run_agent_once(req.query, log)
    except Exception as e:
        log("执行出错", f"{type(e).__name__}: {e}")
        report = (
            f"## ⚠️ 执行失败\n\n"
            f"`{type(e).__name__}: {e}`\n\n"
            f"请查看左侧日志定位问题，常见原因：\n"
            f"- `.env` 里的 `API_KEY` / `BOCHA_API_KEY` 未配置或已失效\n"
        )
    return QueryResponse(log_list=log.items, report_markdown=report)

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("MODEL_ID", "")}

@app.get("/")
def index():
    html_path = BASE_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "index.html 不存在，请放在项目目录"}

# ---------------------- 候选院校对比接口：接收学校名称列表（LLM生成对比，不读取your_school.db） ----------------------
class CompareByNameRequest(BaseModel):
    school_names: List[str]
    major: Optional[str] = None

@app.post("/school/compare_by_name")
def school_compare_by_name(req: CompareByNameRequest):
    """不读取本地your_school.db，调用博查+LLM生成院校对比Markdown"""
    school_list = req.school_names
    major = req.major or ""
    if len(school_list) < 2:
        raise HTTPException(status_code=400, detail="至少选择2所院校进行对比")

    # 博查只回传「成功院校名单」，真正的网页摘要被写进了 chroma，
    # 所以这里再从向量库取回片段，作为联网上下文（并截断长度控 token）
    try:
        bocha_ret = _bocha_search_concurrent(school_list, major)
        ok_schools = bocha_ret.get("success_schools", []) if isinstance(bocha_ret, dict) else []
        search_context = _build_search_context(ok_schools or school_list, major)
    except Exception:
        search_context = "博查搜索获取信息失败"

    prompt = f"""
【网页公开检索信息】
{search_context}

对以下考研院校输出精简Markdown对比表格：
院校列表：{','.join(school_list)}
目标专业：{major if major else "未指定"}

对比维度：院校层次、省份、近年复试线、招生人数、报录风险、复试友好度。
要求：
1. 信息不足写【资料不足】；
2. 只输出表格，简短，不要多余解释、不要大段文字；
3. 表格末尾小字：数据来源于公开网络仅供参考。
"""
    try:
        resp = kaoYan_agent.client.chat.completions.create(
            model=kaoYan_agent.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,   # 精简表格用不了太多 token，卡住上限直接减少推理耗时
        )
        compare_markdown = resp.choices[0].message.content or "生成院校对比表格失败"
        return {
            "compare_markdown": compare_markdown,
            "school_names": school_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"院校对比生成异常：{str(e)}")


# ---------------------- AI学习计划Agent接口 ----------------------
class StudyPlanRequest(BaseModel):
    exam_date: str
    daily_hours: float
    weak_subjects: List[str]
    strong_subjects: List[str]
    target_score: int

@app.post("/studyplan/generate")
def generate_study_plan(req: StudyPlanRequest):
    """仅生成学习计划，用户打卡数据全部保存在浏览器 localStorage"""
    prompt = f"""
你是考研学习规划助手。
考试日期：{req.exam_date}
每日可学习总时长：{req.daily_hours} 小时
薄弱科目：{','.join(req.weak_subjects)}，需要倾斜更多复习时间
优势科目：{','.join(req.strong_subjects)}，适当分配时间维持手感
目标总分：{req.target_score}

输出一份完整Markdown考研学习计划：
1. 分阶段：基础阶段、强化阶段、冲刺阶段；
2. 每个阶段给出时间范围、各科每日时间分配，薄弱科目增加时间占比；
3. 给出每科复习重点、任务清单；
4. 给出简单打卡建议，方便用户记录进度。
只输出markdown计划文本。
"""
    try:
        resp = kaoYan_agent.client.chat.completions.create(
            model=kaoYan_agent.MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        )
        md = resp.choices[0].message.content or "生成计划失败"
        return {"plan_markdown": md}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成学习计划异常：{str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
