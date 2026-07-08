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


def find_groups(client: OpenAI, date: str, users: list) -> list:
    """
    调用 LLM，直接按共同活动主题输出群组。
    同一个人可以同时出现在多个主题群组中。
    """
    if len(users) < 2:
        return []

    user_block = "\n".join(
        f"- 用户【{u['name']}】：{u['text']}" for u in users
    )

    prompt = f"""你是一个生活轨迹分析助手。

以下是 {date} 这一天，多名用户描述的自己做过的事情：

{user_block}

请按照"共同活动/场景"将用户聚合成群组。规则：
1. 每个群组代表一种具体的共同轨迹（如：都去健身、都在咖啡馆、都看了电影等）
2. 每个群组至少包含 2 名用户
3. 同一个用户可以同时出现在多个群组（例如他既去了健身房又去了图书馆）
4. 只根据实际行为/场景重叠判断，不要根据性格或情绪匹配

输出 JSON 数组，每个元素是一个群组：
[
  {{
    "theme": "共同活动的简短标题（5字以内）",
    "members": ["用户名1", "用户名2", ...],
    "overlap": "一句话描述这些人共同做了什么"
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


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="同频群组")
        ws = writer.sheets["同频群组"]
        ws.column_dimensions["A"].width = 14
        ws.column_dimensions["B"].width = 14
        ws.column_dimensions["C"].width = 10
        ws.column_dimensions["D"].width = 40
        ws.column_dimensions["E"].width = 60
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
        all_groups = []

        dates = df["date"].unique().tolist()
        progress = st.progress(0, text="分析中...")

        for i, date in enumerate(dates):
            group = df[df["date"] == date]
            users = group[["name", "text"]].to_dict("records")
            progress.progress((i + 1) / len(dates), text=f"正在分析 {date}（{len(users)} 人）")

            groups = find_groups(client, date, users)
            for g in groups:
                all_groups.append({
                    "日期": g.get("date", date),
                    "活动主题": g.get("theme", ""),
                    "同频人数": len(g.get("members", [])),
                    "群组成员": "、".join(g.get("members", [])),
                    "共同轨迹": g.get("overlap", ""),
                })

        progress.empty()

        if not all_groups:
            st.info("未发现任何轨迹重叠。")
        else:
            out_df = pd.DataFrame(all_groups, columns=["日期", "活动主题", "同频人数", "群组成员", "共同轨迹"])
            out_df = out_df.sort_values(["日期", "同频人数"], ascending=[True, False]).reset_index(drop=True)

            st.subheader(f"发现 {len(out_df)} 个同频群组")
            st.dataframe(out_df, use_container_width=True)

            excel_bytes = to_excel_bytes(out_df)
            unique_dates = sorted(out_df["日期"].unique().tolist())
            if len(unique_dates) == 1:
                date_label = unique_dates[0]
            else:
                date_label = f"{unique_dates[0]}至{unique_dates[-1]}"
            st.download_button(
                label="下载 Excel 报告",
                data=excel_bytes,
                file_name=f"同频群组报告_{date_label}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
