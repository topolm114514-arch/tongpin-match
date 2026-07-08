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


def find_groups(client: OpenAI, users: list) -> list:
    """
    调用 LLM，对所有日期的用户一起按活动主题分组。
    每个用户携带自己的日期，LLM 忽略日期差异只看活动相似度。
    返回的每个 member 带 name 和 date 字段。
    """
    if len(users) < 2:
        return []

    user_block = "\n".join(
        f"- 用户【{u['name']}】（{u['date']}）：{u['text']}" for u in users
    )

    prompt = f"""你是一个生活轨迹分析助手。

以下是多名用户在不同日期描述的自己做过的事情：

{user_block}

请按照"共同活动/场景"将用户聚合成群组，忽略日期差异，只看活动本身是否相似。规则：
1. 每个群组代表一种具体的共同轨迹（如：都去健身、都在咖啡馆、都看了电影等）
2. 每个群组至少包含 2 名用户
3. 同一个用户可以同时出现在多个群组（例如他既去了健身房又去了图书馆）
4. 只根据实际行为/场景重叠判断，不要根据性格或情绪匹配

输出 JSON 数组，每个元素是一个群组，members 中每项需包含用户名和其对应日期：
[
  {{
    "theme": "共同活动的简短标题（5字以内）",
    "members": [{{"name": "用户名", "date": "日期"}}],
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
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def format_members(members: list) -> str:
    """
    将 members 列表格式化为显示字符串。
    若组内所有人日期相同，不加括号；否则日期不同于多数的人加（日期）。
    """
    from collections import Counter
    dates = [m["date"] for m in members]
    most_common_date = Counter(dates).most_common(1)[0][0]
    all_same = len(set(dates)) == 1

    parts = []
    for m in members:
        if all_same or m["date"] == most_common_date:
            parts.append(m["name"])
        else:
            parts.append(f"{m['name']}（{m['date']}）")
    return "、".join(parts)


def build_personal_messages(raw_groups: list, users: list) -> pd.DataFrame:
    """为每个人生成私发文案：列出他/她和哪些人有哪些共同轨迹"""
    name_to_date = {u["name"]: u["date"] for u in users}
    person_items = defaultdict(list)

    for g in raw_groups:
        members = g.get("members", [])
        # 兼容纯字符串 members
        if members and isinstance(members[0], str):
            members = [{"name": m, "date": name_to_date.get(m, "")} for m in members]

        overlap = g.get("overlap", "")

        for m in members:
            others = [x for x in members if x["name"] != m["name"]]
            if others:
                person_items[m["name"]].append({
                    "others": others,
                    "overlap": overlap,
                })

    rows = []
    for person in sorted(person_items.keys()):
        lines = []
        for item in person_items[person]:
            count = len(item["others"])
            lines.append(f"还有{count}个人和你一样{item['overlap']}")
        rows.append({
            "成员": person,
            "私发文案": "同学你好，昨天在我们的群组里，\n" + "\n".join(lines),
        })

    return pd.DataFrame(rows, columns=["成员", "私发文案"])


def to_excel_bytes(groups_df: pd.DataFrame, personal_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1：群组总览
        groups_df.to_excel(writer, index=False, sheet_name="同频群组")
        ws1 = writer.sheets["同频群组"]
        ws1.column_dimensions["A"].width = 14
        ws1.column_dimensions["B"].width = 10
        ws1.column_dimensions["C"].width = 45
        ws1.column_dimensions["D"].width = 60

        # Sheet 2：私发文案
        personal_df.to_excel(writer, index=False, sheet_name="私发文案")
        ws2 = writer.sheets["私发文案"]
        ws2.column_dimensions["A"].width = 18
        ws2.column_dimensions["B"].width = 70
        # 自动换行
        from openpyxl.styles import Alignment
        for row in ws2.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        # 行高自适应（估算）
        for i, row_data in enumerate(personal_df["私发文案"], start=2):
            line_count = str(row_data).count("\n") + 1
            ws2.row_dimensions[i].height = max(18, line_count * 18)

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

        users = df[["name", "date", "text"]].to_dict("records")
        progress = st.progress(0, text="分析中，请稍候...")

        raw_groups = find_groups(client, users)
        progress.progress(1.0, text="分析完成")
        progress.empty()

        all_groups = []
        for g in raw_groups:
            members = g.get("members", [])
            # 兼容 LLM 返回纯字符串 members 的情况
            if members and isinstance(members[0], str):
                name_to_date = {u["name"]: u["date"] for u in users}
                members = [{"name": m, "date": name_to_date.get(m, "")} for m in members]

            all_groups.append({
                "活动主题": g.get("theme", ""),
                "同频人数": len(members),
                "群组成员": format_members(members),
                "共同轨迹": g.get("overlap", ""),
            })

        if not all_groups:
            st.info("未发现任何轨迹重叠。")
        else:
            out_df = pd.DataFrame(all_groups, columns=["活动主题", "同频人数", "群组成员", "共同轨迹"])
            out_df = out_df.sort_values("同频人数", ascending=False).reset_index(drop=True)

            st.subheader(f"发现 {len(out_df)} 个同频群组")
            st.dataframe(out_df, use_container_width=True)

            personal_df = build_personal_messages(raw_groups, users)

            st.subheader("私发文案预览")
            st.dataframe(personal_df, use_container_width=True)

            excel_bytes = to_excel_bytes(out_df, personal_df)
            all_dates = sorted(df["date"].unique().tolist())
            date_label = all_dates[0] if len(all_dates) == 1 else f"{all_dates[0]}至{all_dates[-1]}"
            st.download_button(
                label="下载 Excel 报告",
                data=excel_bytes,
                file_name=f"同频群组报告_{date_label}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
