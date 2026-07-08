import streamlit as st
import pandas as pd
import json
import io
import re
from collections import defaultdict
from datetime import datetime, timedelta
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────
API_KEY = "sk-1989ebf3a6914ffca95cf2fd6cfc0181"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

COL_SUBMIT = "开始答题时间"
COL_NAME   = "1.信息录入"
COL_DATE   = "2.记录日期"
COL_TEXT   = "3.写一写你今天做过的事情"
# ─────────────────────────────────────────────────────


def parse_submit_date(raw: str):
    try:
        return datetime.strptime(raw.strip(), "%d-%b-%Y %H:%M:%S")
    except Exception:
        return None


def resolve_date(date_str: str, submit_dt: datetime) -> str:
    s = date_str.strip()
    base = submit_dt.date()

    relative = {
        "今天": 0, "今日": 0,
        "昨天": 1, "昨日": 1,
        "前天": 2, "前日": 2,
        "前两天": 2, "前几天": 2,
        "大前天": 3,
    }
    for kw, delta in relative.items():
        if kw in s:
            return str(base - timedelta(days=delta))

    m = re.match(r"^(\d{1,2})[.\-/](\d{1,2})$", s)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            return str(datetime(base.year, month, day).date())
        except Exception:
            pass

    m = re.match(r"^(\d{1,2})月(\d{1,2})日?$", s)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            return str(datetime(base.year, month, day).date())
        except Exception:
            pass

    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    return f"{s}（基准{base}）"


def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    df = df[[COL_SUBMIT, COL_NAME, COL_DATE, COL_TEXT]].copy()
    df.columns = ["submit_time", "name", "date_raw", "text"]
    df["name"]        = df["name"].astype(str).str.strip()
    df["date_raw"]    = df["date_raw"].astype(str).str.strip()
    df["text"]        = df["text"].astype(str).str.strip()
    df["submit_time"] = df["submit_time"].astype(str).str.strip()
    df = df[df["text"].notna() & (df["text"] != "") & (df["text"] != "nan")]

    def resolve(row):
        dt = parse_submit_date(row["submit_time"])
        if dt is None:
            return row["date_raw"]
        return resolve_date(row["date_raw"], dt)

    df["date"] = df.apply(resolve, axis=1)
    return df


def find_overlaps(client: OpenAI, date: str, users: list) -> list:
    """调用 LLM 返回两两重叠对"""
    if len(users) < 2:
        return []

    user_block = "\n".join(
        f"- 用户【{u['name']}】：{u['text']}" for u in users
    )

    prompt = f"""你是一个生活轨迹分析助手。

以下是 {date} 这一天，多名用户描述的自己做过的事情：

{user_block}

请找出其中存在生活轨迹重叠的用户对。
"轨迹重叠"指：两人在同一天做了相似的事情、处于相似的场景、或有共同的行为模式。
不要根据性格匹配，只根据实际行为/场景/时间是否重叠来判断。

对于每一对有重叠的用户，输出 JSON 数组：
[
  {{
    "user_a": "用户名",
    "user_b": "用户名",
    "overlap": "一句话描述重叠的具体轨迹"
  }}
]

如果没有任何重叠，输出空数组 []。只输出 JSON，不要输出其他内容。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        results = json.loads(raw)
        for r in results:
            r["date"] = date
        return results
    except json.JSONDecodeError:
        return []


def group_overlaps(pairs: list) -> list:
    """
    将两两重叠对通过连通分量算法聚合成群组。
    同一组内只要有任意两人互相重叠，就归为一组。
    """
    # 按日期分别处理
    date_pairs = defaultdict(list)
    for p in pairs:
        date_pairs[p["date"]].append(p)

    groups = []

    for date, pair_list in date_pairs.items():
        # 构建邻接表 + 记录每对的重叠描述
        adj = defaultdict(set)
        overlap_map = {}

        for p in pair_list:
            a, b = p["user_a"], p["user_b"]
            adj[a].add(b)
            adj[b].add(a)
            key = tuple(sorted([a, b]))
            overlap_map[key] = p["overlap"]

        # BFS 找连通分量
        visited = set()
        all_nodes = set(adj.keys())

        for start in all_nodes:
            if start in visited:
                continue

            # BFS
            component = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            members = sorted(component)

            # 收集组内所有边的重叠描述（去重）
            seen_overlaps = set()
            overlap_texts = []
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    key = tuple(sorted([a, b]))
                    if key in overlap_map:
                        text = overlap_map[key]
                        if text not in seen_overlaps:
                            seen_overlaps.add(text)
                            overlap_texts.append(text)

            groups.append({
                "日期": date,
                "同频人数": len(members),
                "群组成员": "、".join(members),
                "共同轨迹": "；".join(overlap_texts),
            })

    # 按日期、人数降序排列
    groups.sort(key=lambda g: (g["日期"], -g["同频人数"]))
    return groups


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="同频群组")
        ws = writer.sheets["同频群组"]
        ws.column_dimensions["A"].width = 14
        ws.column_dimensions["B"].width = 10
        ws.column_dimensions["C"].width = 40
        ws.column_dimensions["D"].width = 60
    return buf.getvalue()


# ── 页面 ──────────────────────────────────────────────
st.set_page_config(page_title="同频时记", page_icon="🔁", layout="centered")
st.title("同频时记 · 生活轨迹重叠分析")
st.caption("上传问卷 CSV，自动发现同一天做了相似事情的人群")

uploaded = st.file_uploader("上传问卷文件（CSV 或 Excel）", type=["csv", "xlsx", "xls"])

if uploaded:
    df = load_data(uploaded)

    st.subheader("日期解析预览")
    preview = df[["name", "date_raw", "date", "text"]].copy()
    preview.columns = ["微信名", "填写日期", "解析日期", "描述"]
    st.dataframe(preview, use_container_width=True)

    if st.button("开始分析轨迹重叠", type="primary"):
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        all_pairs = []

        dates = df["date"].unique().tolist()
        progress = st.progress(0, text="分析中...")

        for i, date in enumerate(dates):
            group = df[df["date"] == date]
            users = group[["name", "text"]].to_dict("records")
            progress.progress((i + 1) / len(dates), text=f"正在分析 {date}（{len(users)} 人）")

            pairs = find_overlaps(client, date, users)
            all_pairs.extend(pairs)

        progress.empty()

        if not all_pairs:
            st.info("未发现任何轨迹重叠。")
        else:
            # 聚合成群组
            groups = group_overlaps(all_pairs)
            out_df = pd.DataFrame(groups, columns=["日期", "同频人数", "群组成员", "共同轨迹"])

            st.subheader(f"发现 {len(out_df)} 个同频群组")
            st.dataframe(out_df, use_container_width=True)

            excel_bytes = to_excel_bytes(out_df)
            st.download_button(
                label="下载 Excel 报告",
                data=excel_bytes,
                file_name="同频群组报告.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
