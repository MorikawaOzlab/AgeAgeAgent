#!/usr/bin/env python3
"""
make_scml_log_viewer.py

同じフォルダ内の stats.xlsx と data フォルダ内の actions.txt / agents.txt /
negs.txt / simsteps.txt を読み込み、SCMLログ確認用の単体HTMLを生成します。

使い方:
    python make_scml_log_viewer.py

別のPythonファイルから呼び出す場合:
    from pathlib import Path
    from make_scml_log_viewer import generate_scml_log_viewer

    generate_scml_log_viewer(
        base_dir=Path(__file__).resolve().parent,
        stats_path="stats.xlsx",
        data_dir="data",
        output_path="scml_log_viewer.html",
    )

配置例:
    folder/
    ├─ stats.xlsx
    ├─ make_scml_log_viewer.py
    └─ data/
       ├─ actions.txt
       ├─ agents.txt
       ├─ negs.txt
       └─ simsteps.txt

引数で変更する場合:
    python make_scml_log_viewer.py stats.xlsx scml_log_viewer.html --data-dir data

依存ライブラリ:
    なし。Python標準ライブラリだけで動きます。
"""

from __future__ import annotations

import argparse
import json
import math
import posixpath
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET

XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

KNOWN_AGENT_METRICS = [
    "shortfall_quantity",
    "shortfall_penalty",
    "inventory_penalized",
    "inventory_output",
    "inventory_input",
    "disposal_cost",
    "storage_cost",
    "productivity",
    "bankrupt",
    "balance",
    "score",
]

DATA_FILES = ["actions.txt", "agents.txt", "negs.txt", "simsteps.txt"]


@dataclass(frozen=True)
class GenerationResult:
    """HTML生成結果。別Pythonファイルから呼び出したときに扱いやすい戻り値。"""

    output_path: Path
    stats_path: Path
    data_dir: Path
    stats_agent_count: int
    action_count: int
    negotiation_count: int
    simstep_count: int
    missing_files: list[str]


__all__ = [
    "GenerationResult",
    "build_app_data",
    "build_html",
    "generate_scml_log_viewer",
    "generate_html",
    "read_xlsx_first_sheet",
    "build_stats",
    "build_negotiation_data",
]


def column_name_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters.upper():
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return index - 1


def maybe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if re.fullmatch(r"0\d+", text):
        return text
    try:
        num = float(text)
    except ValueError:
        return text
    if math.isfinite(num) and num.is_integer():
        return int(num)
    return num


def safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return str(value)


def read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall("x:si", XLSX_NS):
        texts = [t.text or "" for t in si.findall(".//x:t", XLSX_NS)]
        strings.append("".join(texts))
    return strings


def find_first_sheet_path(zf: ZipFile) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    first_sheet = workbook.find("x:sheets/x:sheet", XLSX_NS)
    if first_sheet is None:
        raise ValueError("Workbookにシートが見つかりません。")

    rel_id = first_sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id")
    if not rel_id:
        raise ValueError("先頭シートのRelationship IDが見つかりません。")

    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall("r:Relationship", REL_NS):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target", "")
            if target.startswith("/"):
                return target.lstrip("/")
            return posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(f"シートのXMLパスを解決できませんでした: {rel_id}")


def read_xlsx_first_sheet(xlsx_path: Path) -> list[list[Any]]:
    with ZipFile(xlsx_path) as zf:
        shared_strings = read_shared_strings(zf)
        sheet_path = find_first_sheet_path(zf)
        root = ET.fromstring(zf.read(sheet_path))

        rows: list[list[Any]] = []
        max_cols = 0
        for row_el in root.findall(".//x:sheetData/x:row", XLSX_NS):
            row: list[Any] = []
            for cell in row_el.findall("x:c", XLSX_NS):
                ref = cell.attrib.get("r", "")
                col_index = column_name_to_index(ref) if ref else len(row)
                while len(row) < col_index:
                    row.append(None)

                cell_type = cell.attrib.get("t")
                value_el = cell.find("x:v", XLSX_NS)
                inline_el = cell.find("x:is", XLSX_NS)
                value: Any = None

                if cell_type == "s" and value_el is not None:
                    idx = int(value_el.text or 0)
                    value = shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
                elif cell_type == "inlineStr" and inline_el is not None:
                    value = "".join(t.text or "" for t in inline_el.findall(".//x:t", XLSX_NS))
                elif cell_type == "b" and value_el is not None:
                    value = value_el.text == "1"
                elif value_el is not None:
                    value = maybe_number(value_el.text)

                row.append(value)
            max_cols = max(max_cols, len(row))
            rows.append(row)

    for row in rows:
        if len(row) < max_cols:
            row.extend([None] * (max_cols - len(row)))
    return rows


def split_agent_metric(header: str) -> tuple[str, str] | None:
    name = str(header).strip()
    for metric in KNOWN_AGENT_METRICS:
        prefix = metric + "_"
        if name.startswith(prefix):
            agent = name[len(prefix) :]
            if "@" in agent:
                return metric, agent

    if "_" in name:
        metric, agent = name.rsplit("_", 1)
        if metric and "@" in agent:
            return metric, agent
    return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if value is None:
        return None
    try:
        num = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    return num if math.isfinite(num) else None


def last_number(values: list[Any]) -> float | None:
    for value in reversed(values):
        num = as_number(value)
        if num is not None:
            return num
    return None


def first_number(values: list[Any]) -> float | None:
    for value in values:
        num = as_number(value)
        if num is not None:
            return num
    return None


def numeric_sum(values: list[Any]) -> float | None:
    nums = [as_number(v) for v in values]
    nums = [v for v in nums if v is not None]
    return sum(nums) if nums else None


def numeric_max(values: list[Any]) -> float | None:
    nums = [as_number(v) for v in values]
    nums = [v for v in nums if v is not None]
    return max(nums) if nums else None


def build_stats(rows: list[list[Any]], source_file: str) -> dict[str, Any]:
    if not rows:
        return {"sourceFile": source_file, "columns": [], "rows": [], "steps": [], "agents": {}, "metrics": [], "summary": []}

    headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    data_rows = rows[1:]

    records: list[dict[str, Any]] = []
    for row in data_rows:
        record = {headers[i]: safe_json_value(row[i]) if i < len(row) else None for i in range(len(headers))}
        records.append(record)

    steps: list[Any] = []
    for i, record in enumerate(records):
        step_value = record.get("step", i)
        steps.append(step_value if step_value is not None else i)

    agents: dict[str, dict[str, Any]] = {}
    global_metrics: dict[str, list[Any]] = {}

    for header in headers:
        parsed = split_agent_metric(header)
        values = [record.get(header) for record in records]
        if parsed is None:
            nums = [as_number(v) for v in values]
            if any(v is not None for v in nums):
                global_metrics[header] = values
            continue

        metric, agent = parsed
        agents.setdefault(agent, {"metrics": {}})
        agents[agent]["metrics"][metric] = values

    metric_names = sorted({metric for agent_info in agents.values() for metric in agent_info["metrics"].keys()})

    summary: list[dict[str, Any]] = []
    for agent, info in agents.items():
        metrics = info["metrics"]
        row = {"agent": agent}
        if "score" in metrics:
            row["finalScore"] = last_number(metrics["score"])
            row["initialScore"] = first_number(metrics["score"])
        if "balance" in metrics:
            row["finalBalance"] = last_number(metrics["balance"])
        if "shortfall_quantity" in metrics:
            row["shortfallQuantityTotal"] = numeric_sum(metrics["shortfall_quantity"])
            row["shortfallQuantityMax"] = numeric_max(metrics["shortfall_quantity"])
        if "productivity" in metrics:
            row["finalProductivity"] = last_number(metrics["productivity"])
        summary.append(row)

    summary.sort(key=lambda r: (r.get("finalScore") is None, -(r.get("finalScore") or 0)))

    return {
        "sourceFile": source_file,
        "columns": headers,
        "rows": records,
        "steps": steps,
        "agents": agents,
        "metrics": metric_names,
        "globalMetrics": global_metrics,
        "summary": summary,
    }


def convert_text_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    if text == "True":
        return True
    if text == "False":
        return False
    if text.lower() == "nan":
        return None
    if text.lower() == "inf":
        return "inf"
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", text):
        try:
            num = float(text)
            return num if math.isfinite(num) else text
        except ValueError:
            return text
    return text


def read_dataframe_text(path: Path) -> list[dict[str, Any]]:
    """pandas DataFrame.to_string() 風の空白区切りテキストを読む。"""
    if not path.exists():
        return []

    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return []

    headers = lines[0].split()
    rows: list[dict[str, Any]] = []
    error_index = headers.index("error") if "error" in headers else None

    for line in lines[1:]:
        values = line.split()
        if not values:
            continue

        # pandas の表示では左端に行番号が出るため、列数が合うようなら捨てる。
        if len(values) == len(headers) + 1:
            values = values[1:]
        elif len(values) == len(headers) and values[0].lstrip("-+").isdigit():
            # negs.txt は error 列が空欄で、行番号を捨てると1列足りなくなることがある。
            values = values[1:]

        if len(values) == len(headers) - 1 and error_index is not None:
            values.insert(error_index, "")

        if len(values) < len(headers):
            values.extend([""] * (len(headers) - len(values)))
        elif len(values) > len(headers):
            # ほぼ起こらないが、余った分は最後の列に連結して壊れにくくする。
            values = values[: len(headers) - 1] + [" ".join(values[len(headers) - 1 :])]

        rows.append({header: convert_text_value(value) for header, value in zip(headers, values)})
    return rows


def find_data_file(base_dir: Path, data_dir: Path, filename: str) -> Path | None:
    preferred = data_dir / filename
    if preferred.exists():
        return preferred
    fallback = base_dir / filename
    if fallback.exists():
        return fallback
    return None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def int_or_none(value: Any) -> int | None:
    num = as_number(value)
    return int(num) if num is not None else None


def float_or_none(value: Any) -> float | None:
    return as_number(value)


def build_negotiation_data(base_dir: Path, data_dir: Path) -> dict[str, Any]:
    paths = {name: find_data_file(base_dir, data_dir, name) for name in DATA_FILES}
    missing = [name for name, path in paths.items() if path is None]

    agents_raw = read_dataframe_text(paths["agents.txt"]) if paths["agents.txt"] else []
    negs_raw = read_dataframe_text(paths["negs.txt"]) if paths["negs.txt"] else []
    actions_raw = read_dataframe_text(paths["actions.txt"]) if paths["actions.txt"] else []
    simsteps_raw = read_dataframe_text(paths["simsteps.txt"]) if paths["simsteps.txt"] else []

    agent_by_id: dict[str, dict[str, Any]] = {}
    agent_by_name: dict[str, dict[str, Any]] = {}
    for row in agents_raw:
        agent_id = str(row.get("id", ""))
        name = str(row.get("name", ""))
        agent_type = str(row.get("type", ""))
        if not name:
            continue
        item = {"id": agent_id, "name": name, "type": agent_type}
        agent_by_id[agent_id] = item
        agent_by_name[name] = item

    negotiations: list[dict[str, Any]] = []
    neg_by_id: dict[str, dict[str, Any]] = {}
    for row in negs_raw:
        agent0 = str(row.get("agent_time0") or agent_by_id.get(str(row.get("agent0_id", "")), {}).get("name", ""))
        agent1 = str(row.get("agent_time1") or agent_by_id.get(str(row.get("agent1_id", "")), {}).get("name", ""))
        item = {
            "id": int_or_none(row.get("id")),
            "sim_step": int_or_none(row.get("sim_step")),
            "simstep_id": int_or_none(row.get("simstep_id")),
            "round_step": int_or_none(row.get("step")),
            "relative_time": float_or_none(row.get("relative_time")),
            "has_agreement": bool_value(row.get("has_agreement")),
            "timedout": bool_value(row.get("timedout")),
            "ended": bool_value(row.get("ended")),
            "erred": bool_value(row.get("erred")),
            "error": row.get("error", ""),
            "agent0_id": str(row.get("agent0_id", "")),
            "agent1_id": str(row.get("agent1_id", "")),
            "agent0": agent0,
            "agent1": agent1,
            "quantity": int_or_none(row.get("quantity")),
            "delivery_step": int_or_none(row.get("delivery_step")),
            "unit_price": float_or_none(row.get("unit_price")),
            "product": int_or_none(row.get("product")),
            "needed_sales0": int_or_none(row.get("needed_sales0")),
            "needed_sales1": int_or_none(row.get("needed_sales1")),
            "needed_supplies0": int_or_none(row.get("needed_supplies0")),
            "needed_supplies1": int_or_none(row.get("needed_supplies1")),
            "trading_price": float_or_none(row.get("trading_price")),
        }
        negotiations.append(item)
        if item["id"] is not None:
            neg_by_id[str(item["id"])] = item

        for name in [agent0, agent1]:
            if name and name not in agent_by_name:
                agent_by_name[name] = {"id": "", "name": name, "type": ""}

    actions: list[dict[str, Any]] = []
    for row in actions_raw:
        neg_id = int_or_none(row.get("neg_id"))
        neg = neg_by_id.get(str(neg_id)) if neg_id is not None else None
        sender = str(row.get("sender", ""))
        receiver = str(row.get("receiver", ""))
        item = {
            "id": int_or_none(row.get("id")),
            "neg_id": neg_id,
            "sim_step": neg.get("sim_step") if neg else None,
            "neg_round": int_or_none(row.get("step")),
            "relative_time": float_or_none(row.get("relative_time")),
            "time": float_or_none(row.get("time")),
            "sender": sender,
            "receiver": receiver,
            "sender_agent_id": str(row.get("sender_agent_id", "")),
            "receiver_agent_id": str(row.get("receiver_agent_id", "")),
            "state": str(row.get("state", "")),
            "quantity": int_or_none(row.get("quantity")),
            "delivery_step": int_or_none(row.get("delivery_step")),
            "unit_price": float_or_none(row.get("unit_price")),
            "agent0": neg.get("agent0") if neg else "",
            "agent1": neg.get("agent1") if neg else "",
            "has_agreement": neg.get("has_agreement") if neg else None,
        }
        actions.append(item)
        for name in [sender, receiver]:
            if name and name not in agent_by_name:
                agent_by_name[name] = {"id": "", "name": name, "type": ""}

    simsteps: list[dict[str, Any]] = []
    for row in simsteps_raw:
        simsteps.append(
            {
                "id": int_or_none(row.get("id")),
                "step": int_or_none(row.get("step")),
                "started": float_or_none(row.get("started")),
                "ended": float_or_none(row.get("ended")),
                "relative_time_start": float_or_none(row.get("relative_time_start")),
                "relative_time_end": float_or_none(row.get("relative_time_end")),
                "duration": float_or_none(row.get("duration")),
                "world": str(row.get("world", "")),
            }
        )

    return {
        "sourceFiles": {name: str(path.name) if path else None for name, path in paths.items()},
        "missingFiles": missing,
        "agents": sorted(agent_by_name.values(), key=lambda x: x.get("name", "")),
        "negotiations": negotiations,
        "actions": actions,
        "simsteps": simsteps,
    }


def build_html(app_data: dict[str, Any]) -> str:
    data_json = json.dumps(app_data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    template = r'''<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SCML Log Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7fb;
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --danger: #dc2626;
      --good: #059669;
      --warn: #d97706;
      --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
      --radius: 18px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { max-width: 100%; overflow-x: hidden; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header { padding: 26px 28px 12px; }
    h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: -0.03em; }
    .subtitle { color: var(--muted); font-size: 14px; }
    main { padding: 14px 28px 36px; max-width: 1500px; width: 100%; margin: 0 auto; overflow-x: hidden; }
    .grid { display: grid; gap: 16px; min-width: 0; }
    .grid > * { min-width: 0; }
    .grid-2 { grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr); }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); padding: 18px; min-width: 0; }
    .card h2 { margin: 0 0 12px; font-size: 18px; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 14px; }
    .chart-zoom-controls { align-items: center; margin: 0 0 12px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 14px; background: #f8fafc; }
    .chart-zoom-controls .field { min-width: 260px; }
    .chart-zoom-controls input[type="range"] { width: 220px; padding: 0; }
    .zoom-number { width: 88px; }
    .zoom-label { min-width: 58px; color: var(--muted); font-size: 12px; font-weight: 800; }
    .field { display: grid; gap: 5px; min-width: 160px; }
    label { font-size: 12px; color: var(--muted); font-weight: 700; }
    select, input { height: 38px; border: 1px solid var(--line); border-radius: 12px; padding: 0 12px; background: white; color: var(--text); font-size: 14px; }
    .hidden-state { display: none !important; }
    input[type="checkbox"] { height: auto; }
    .checkbox { display: flex; gap: 8px; align-items: center; height: 38px; }
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 16px; }
    .tab { border: 1px solid var(--line); background: white; color: var(--text); border-radius: 999px; padding: 9px 14px; cursor: pointer; font-weight: 700; }
    .tab.active { background: var(--accent); color: white; border-color: var(--accent); }
    .button-picker { margin: 12px 0 0; padding: 12px; border: 1px solid var(--line); border-radius: 16px; background: #f8fafc; }
    .button-picker:first-of-type { margin-top: 2px; }
    .button-picker-title { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 10px; color: var(--muted); font-size: 12px; font-weight: 800; }
    .button-grid { display: flex; flex-wrap: wrap; gap: 8px; }
    .choice-button { border: 1px solid var(--line); background: white; color: var(--text); border-radius: 999px; padding: 8px 12px; cursor: pointer; font-size: 13px; font-weight: 800; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }
    .choice-button:hover { border-color: var(--accent); background: var(--accent-soft); }
    .choice-button.active { background: var(--accent); color: white; border-color: var(--accent); box-shadow: 0 6px 18px rgba(37, 99, 235, 0.22); }
    .choice-button.all-button { border-style: dashed; }
    .metric-buttons .choice-button { font-size: 12px; padding: 7px 10px; }
    .view { display: none; }
    .view.active { display: block; }
    canvas { width: 100%; height: 380px; display: block; }
    .small-chart { height: 300px; }
    .chart-scroll { width: 100%; max-width: 100%; min-width: 0; overflow-x: auto; overflow-y: hidden; overscroll-behavior-x: contain; padding-bottom: 6px; border: 1px solid var(--line); border-radius: 14px; background: white; }
    .chart-scroll canvas { display: block; }
    .stats-chart-canvas { height: 380px; min-width: 100%; max-width: none; flex: 0 0 auto; }
    .contract-chart-canvas { height: 300px; min-width: 100%; max-width: none; flex: 0 0 auto; }
    .chart-scroll canvas.cached-chart { display: none; }
    .kpis { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 14px; }
    .kpi { border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #fbfdff; }
    .kpi .name { color: var(--muted); font-size: 12px; font-weight: 700; }
    .kpi .value { font-size: 22px; font-weight: 800; margin-top: 4px; }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 14px; max-height: 520px; }
    .compact-table { max-height: 240px; margin-top: 12px; box-shadow: none; }
    .compact-table table { min-width: 720px; }
    table { border-collapse: collapse; width: 100%; min-width: 760px; background: white; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; font-size: 13px; white-space: nowrap; }
    th { position: sticky; top: 0; background: #f8fafc; color: #374151; z-index: 1; }
    th:first-child, td:first-child, .left { text-align: left; }
    tr:hover td { background: #f9fbff; }
    .clickable tbody tr { cursor: pointer; }
    .pill { display: inline-flex; align-items: center; justify-content: center; min-width: 74px; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
    .pill.agreement { color: var(--good); background: #d1fae5; }
    .pill.ended { color: var(--danger); background: #fee2e2; }
    .pill.continuing { color: var(--warn); background: #fef3c7; }
    .note { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .warning { border: 1px solid #fde68a; background: #fffbeb; color: #92400e; border-radius: 14px; padding: 12px 14px; margin-bottom: 14px; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; font-size: 12px; color: var(--muted); }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
    .right { text-align: right; }
    .muted { color: var(--muted); }
    @media (max-width: 900px) {
      main, header { padding-left: 14px; padding-right: 14px; }
      .grid-2 { grid-template-columns: 1fr; }
      .kpis { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      canvas { height: 300px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SCML Log Viewer</h1>
    <div class="subtitle" id="subtitle"></div>
  </header>
  <main>
    <div id="warningBox"></div>

    <section class="card">
      <div class="controls">
        <div class="field">
          <label for="compareMode">stats表示</label>
          <select id="compareMode">
            <option value="single">選択エージェントのみ</option>
            <option value="all">全エージェント表示</option>
          </select>
        </div>
        <label class="checkbox"><input type="checkbox" id="normalizeCheck" /> statsを正規化</label>
      </div>

      <select id="agentSelect" class="hidden-state" aria-hidden="true" tabindex="-1"></select>
      <select id="partnerSelect" class="hidden-state" aria-hidden="true" tabindex="-1"></select>
      <select id="metricSelect" class="hidden-state" aria-hidden="true" tabindex="-1"></select>

      <div class="button-picker">
        <div class="button-picker-title">
          <span id="agentPickerTitle">表示エージェント（複数選択可）</span>
          <span id="selectedAgentLabel"></span>
        </div>
        <div id="agentButtonGrid" class="button-grid"></div>
      </div>

      <div id="partnerPicker" class="button-picker">
        <div class="button-picker-title">
          <span>交渉相手</span>
          <span id="selectedPartnerLabel"></span>
        </div>
        <div id="partnerButtonGrid" class="button-grid"></div>
      </div>

      <div id="metricPicker" class="button-picker metric-buttons">
        <div class="button-picker-title">
          <span>stats指標</span>
          <span id="selectedMetricLabel"></span>
        </div>
        <div id="metricButtonGrid" class="button-grid"></div>
      </div>

      <div class="tabs">
        <button class="tab active" data-view="statsView">stats推移</button>
        <button class="tab" data-view="negotiationView">交渉アクション</button>
        <button class="tab" data-view="contractView">契約分析</button>
      </div>
    </section>

    <section id="statsView" class="view active">
      <div class="grid grid-2">
        <div class="card">
          <h2>stats グラフ</h2>
          <div class="controls chart-zoom-controls">
            <div class="field">
              <label for="statsZoomRange">横軸倍率</label>
              <input id="statsZoomRange" type="range" min="0.1" max="4" step="0.05" value="0.25" />
            </div>
            <input id="statsZoomInput" class="zoom-number" type="number" min="0.1" max="4" step="0.05" value="0.25" />
            <span id="statsZoomLabel" class="zoom-label">25%</span>
          </div>
          <div id="statsChartScroller" class="chart-scroll stats-chart-scroll">
            <canvas id="statsChart" class="stats-chart-canvas"></canvas>
          </div>
          <div id="statsLegend" class="legend"></div>
        </div>
        <div class="card">
          <h2>Final score ランキング</h2>
          <div id="rankingTable" class="table-wrap"></div>
        </div>
      </div>
    </section>

    <section id="negotiationView" class="view">
      <div class="grid">
        <div class="card">
          <h2>相手別サマリー</h2>
          <div id="partnerSummary" class="table-wrap clickable"></div>
          <p class="note">行をクリックすると、その相手だけに絞り込めます。</p>
        </div>
        <div class="card">
          <div class="controls">
            <div class="field">
              <label for="actionStateSelect">アクション状態</label>
              <select id="actionStateSelect">
                <option value="__ALL__">すべて</option>
                <option value="agreement">agreement</option>
                <option value="continuing">continuing</option>
                <option value="ended">ended</option>
              </select>
            </div>
            <div class="field" style="min-width: 260px;">
              <label for="actionSearch">表内検索</label>
              <input id="actionSearch" placeholder="agent名 / state / neg_id など" />
            </div>
          </div>
          <h2>交渉アクション一覧</h2>
          <div id="actionCount" class="note"></div>
          <div id="actionTable" class="table-wrap"></div>
        </div>
      </div>
    </section>

    <section id="contractView" class="view">
      <div class="grid">
        <div class="card">
          <h2>stepごとの契約数</h2>
          <div class="controls chart-zoom-controls">
            <div class="field">
              <label for="contractZoomRange">横軸倍率</label>
              <input id="contractZoomRange" type="range" min="0.1" max="4" step="0.05" value="1" />
            </div>
            <input id="contractZoomInput" class="zoom-number" type="number" min="0.1" max="4" step="0.05" value="1" />
            <span id="contractZoomLabel" class="zoom-label">100%</span>
          </div>
          <div id="contractChartScroller" class="chart-scroll"></div>
          <div id="contractChartLegend" class="legend"></div>
          <div id="contractChartNote" class="note"></div>
          <div id="contractStepTable" class="table-wrap compact-table"></div>
        </div>
        <div class="card">
          <h2>契約KPI</h2>
          <div id="contractKpis" class="kpis"></div>
        </div>
        <div class="card">
          <h2>実際に成立した契約一覧</h2>
          <div id="contractCount" class="note"></div>
          <div id="contractTable" class="table-wrap"></div>
        </div>
      </div>
    </section>
  </main>

  <script id="app-data" type="application/json">__APP_DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('app-data').textContent);
    const $ = (id) => document.getElementById(id);
    const nf = new Intl.NumberFormat('ja-JP', { maximumFractionDigits: 3 });
    const colors = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed', '#0891b2', '#be123c', '#4d7c0f', '#9333ea', '#0f766e', '#b45309', '#1d4ed8', '#db2777', '#475569'];
    const CONTRACT_CHART_STEP_WIDTH = 42;
    const CONTRACT_CHART_BAR_WIDTH = 30;
    const CONTRACT_CHART_HEIGHT = 300;
    const STATS_CHART_BASE_STEP_WIDTH = 36;
    const STATS_CHART_HEIGHT = 380;
    let statsXAxisScale = 0.25;
    let contractXAxisScale = 1;
    let selectedAgents = [];
    let contractChartCache = new Map();
    let contractChartPreRendered = false;
    let visibleContractChartCanvas = null;
    let resizeTimer = null;

    // 表示切替を軽くするためのキャッシュ群。
    // グラフや巨大テーブルは、同じ条件ならDOM/Canvasを作り直さない。
    let agentNamesCache = null;
    let baseContractStepLabelsCache = null;
    const contractRowsCache = new Map();
    const contractAggregateCache = new Map();
    const contractPayloadCache = new Map();
    const contractYMaxCache = new Map();
    const contractInfoCache = new Map();
    const partnerSummaryCache = new Map();
    const actionTableCache = new Map();
    const statsChartCache = new Map();
    let lastStatsRenderKey = '';
    let lastPartnerSummaryKey = '';
    let lastActionTableKey = '';
    let lastContractInfoKey = '';
    let partnerOptionsDirty = false;

    function escapeHTML(value) {
      return String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
    }
    function fmt(value) {
      if (value === null || value === undefined || value === '') return '';
      if (typeof value === 'number' && Number.isFinite(value)) return nf.format(value);
      return escapeHTML(value);
    }
    function num(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }
    function statePill(state) {
      const cls = ['agreement', 'ended', 'continuing'].includes(state) ? state : '';
      return `<span class="pill ${cls}">${escapeHTML(state)}</span>`;
    }
    function otherParty(row, agent) {
      if (row.agent0 === agent) return row.agent1;
      if (row.agent1 === agent) return row.agent0;
      if (row.sender === agent) return row.receiver;
      if (row.receiver === agent) return row.sender;
      return '';
    }
    function uniqueSorted(values) {
      return [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b), 'ja'));
    }

    function clampZoom(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return 1;
      return Math.min(4, Math.max(0.1, n));
    }

    function updateZoomLabel(kind) {
      const scale = kind === 'stats' ? statsXAxisScale : contractXAxisScale;
      const label = $(kind === 'stats' ? 'statsZoomLabel' : 'contractZoomLabel');
      if (label) label.textContent = `${Math.round(scale * 100)}%`;
    }

    function syncZoomInputs(kind, scale) {
      const prefix = kind === 'stats' ? 'stats' : 'contract';
      const range = $(`${prefix}ZoomRange`);
      const input = $(`${prefix}ZoomInput`);
      const text = String(Math.round(scale * 100) / 100);
      if (range) range.value = text;
      if (input) input.value = text;
      updateZoomLabel(kind);
    }

    function setStatsXAxisScale(value) {
      statsXAxisScale = clampZoom(value);
      syncZoomInputs('stats', statsXAxisScale);
      lastStatsRenderKey = '';
      renderStats(true);
    }

    function applyContractChartZoomToCanvas(canvas) {
      if (!canvas) return;
      const baseWidth = Number(canvas.dataset.baseCssWidth || canvas.style.width.replace('px', '') || canvas.width || 320);
      canvas.style.width = `${Math.max(320, baseWidth * contractXAxisScale)}px`;
    }

    function applyContractXAxisScale(value) {
      contractXAxisScale = clampZoom(value);
      syncZoomInputs('contract', contractXAxisScale);
      contractChartCache.forEach((entry) => applyContractChartZoomToCanvas(entry.canvas));
    }
    function agentNames() {
      if (agentNamesCache) return agentNamesCache;
      const names = new Set();
      Object.keys(DATA.stats?.agents || {}).forEach((x) => names.add(x));
      (DATA.negotiation?.agents || []).forEach((a) => names.add(a.name));
      (DATA.negotiation?.negotiations || []).forEach((n) => { if (n.agent0) names.add(n.agent0); if (n.agent1) names.add(n.agent1); });
      (DATA.negotiation?.actions || []).forEach((a) => { if (a.sender) names.add(a.sender); if (a.receiver) names.add(a.receiver); });
      agentNamesCache = uniqueSorted([...names]).filter((x) => !['NoAgent', 'SELLER', 'BUYER'].includes(x));
      return agentNamesCache;
    }

    function selectedAgentNames() {
      const valid = new Set(agentNames());
      selectedAgents = selectedAgents.filter((agent) => valid.has(agent));
      if (!selectedAgents.length && $('agentSelect')?.value) selectedAgents = [$('agentSelect').value];
      if (!selectedAgents.length) selectedAgents = agentNames().slice(0, 1);
      return [...selectedAgents];
    }

    function selectedAgentSet() {
      return new Set(selectedAgentNames());
    }

    function syncPrimaryAgentSelect() {
      const first = selectedAgentNames()[0];
      if (first && $('agentSelect')) $('agentSelect').value = first;
    }

    function selectedAgentLabel() {
      const agents = selectedAgentNames();
      if (!agents.length) return '未選択';
      if (agents.length <= 2) return agents.join(', ');
      return `${agents[0]}, ${agents[1]} ほか${agents.length - 2}体`;
    }

    function rowMatchesSelectedAgents(row) {
      const set = selectedAgentSet();
      return set.has(row.agent0) || set.has(row.agent1) || set.has(row.sender) || set.has(row.receiver);
    }

    function otherPartiesForSelected(row) {
      const partners = [];
      selectedAgentNames().forEach((agent) => {
        const other = otherParty(row, agent);
        if (other) partners.push(other);
      });
      return uniqueSorted(partners);
    }

    function rowMatchesPartner(row, partner) {
      if (partner === '__ALL__') return true;
      return otherPartiesForSelected(row).includes(partner);
    }

    function activeViewId() {
      const activeTab = document.querySelector('.tab.active');
      return activeTab?.dataset?.view || 'statsView';
    }

    function multiAgentSelectionAllowed() {
      return activeViewId() === 'statsView';
    }

    function enforceAgentSelectionMode() {
      if (multiAgentSelectionAllowed() || selectedAgents.length <= 1) return false;
      selectedAgents = selectedAgents.slice(0, 1);
      syncPrimaryAgentSelect();
      return true;
    }

    function updateScopedPickers() {
      const view = activeViewId();
      const partnerVisible = view === 'negotiationView' || view === 'contractView';
      const metricVisible = view === 'statsView';
      if ($('partnerPicker')) $('partnerPicker').style.display = partnerVisible ? 'block' : 'none';
      if ($('metricPicker')) $('metricPicker').style.display = metricVisible ? 'block' : 'none';
      if ($('agentPickerTitle')) {
        $('agentPickerTitle').textContent = multiAgentSelectionAllowed() ? '表示エージェント（複数選択可）' : '表示エージェント（単一選択）';
      }
    }

    function renderAgentButtons() {
      const agents = agentNames();
      $('agentButtonGrid').innerHTML = agents.map((agent) => {
        const info = (DATA.negotiation?.agents || []).find((a) => a.name === agent);
        const title = info?.type ? `${agent} / ${info.type}` : agent;
        return `<button type="button" class="choice-button" data-agent="${escapeHTML(agent)}" title="${escapeHTML(title)}">${escapeHTML(agent)}</button>`;
      }).join('');
      $('agentButtonGrid').querySelectorAll('.choice-button').forEach((btn) => {
        btn.addEventListener('click', () => {
          const agent = btn.dataset.agent;
          if (activeViewId() === 'statsView' && $('compareMode') && $('compareMode').value !== 'single') {
            $('compareMode').value = 'single';
          }
          if (!multiAgentSelectionAllowed()) {
            selectedAgents = [agent];
          } else {
            const exists = selectedAgents.includes(agent);
            if (exists && selectedAgents.length > 1) {
              selectedAgents = selectedAgents.filter((a) => a !== agent);
            } else if (!exists) {
              selectedAgents.push(agent);
            }
          }
          afterAgentSelectionChanged();
        });
      });
      updateAgentButtonStates();
    }

    function updateAgentButtonStates() {
      const current = new Set(selectedAgentNames());
      $('selectedAgentLabel').textContent = `選択中: ${selectedAgentLabel()}`;
      $('agentButtonGrid').querySelectorAll('.choice-button').forEach((btn) => {
        btn.classList.toggle('active', current.has(btn.dataset.agent));
      });
    }

    function afterAgentSelectionChanged() {
      enforceAgentSelectionMode();
      syncPrimaryAgentSelect();
      updateAgentButtonStates();
      partnerOptionsDirty = true;
      if (activeViewId() !== 'statsView') {
        updatePartnerSelect();
        partnerOptionsDirty = false;
      }
      renderCurrentView();
    }

    function renderPartnerButtons() {
      const select = $('partnerSelect');
      const buttons = Array.from(select.options).map((option) => {
        const value = option.value;
        const label = option.textContent || value;
        const allClass = value === '__ALL__' ? ' all-button' : '';
        return `<button type="button" class="choice-button${allClass}" data-partner="${escapeHTML(value)}">${escapeHTML(label)}</button>`;
      }).join('');
      $('partnerButtonGrid').innerHTML = buttons;
      $('partnerButtonGrid').querySelectorAll('.choice-button').forEach((btn) => {
        btn.addEventListener('click', () => {
          $('partnerSelect').value = btn.dataset.partner;
          updatePartnerButtonStates();
          renderCurrentView();
        });
      });
      updatePartnerButtonStates();
    }

    function updatePartnerButtonStates() {
      const current = $('partnerSelect').value || '__ALL__';
      $('selectedPartnerLabel').textContent = current === '__ALL__' ? '選択中: 全員' : `選択中: ${current}`;
      $('partnerButtonGrid').querySelectorAll('.choice-button').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.partner === current);
      });
    }

    function renderMetricButtons() {
      const metrics = DATA.stats?.metrics || [];
      $('metricButtonGrid').innerHTML = metrics.map((metric) => {
        return `<button type="button" class="choice-button" data-metric="${escapeHTML(metric)}">${escapeHTML(metric)}</button>`;
      }).join('');
      $('metricButtonGrid').querySelectorAll('.choice-button').forEach((btn) => {
        btn.addEventListener('click', () => {
          $('metricSelect').value = btn.dataset.metric;
          updateMetricButtonStates();
          renderStats();
        });
      });
      updateMetricButtonStates();
    }

    function updateMetricButtonStates() {
      const current = $('metricSelect').value;
      $('selectedMetricLabel').textContent = current ? `選択中: ${current}` : '';
      $('metricButtonGrid').querySelectorAll('.choice-button').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.metric === current);
      });
    }

    function populateSelects() {
      const agents = agentNames();
      const agentSelect = $('agentSelect');
      agentSelect.innerHTML = agents.map((a) => `<option value="${escapeHTML(a)}">${escapeHTML(a)}</option>`).join('');
      const preferred = agents.find((a) => a.includes('My')) || agents.find((a) => a.includes('ASS0')) || agents[0];
      selectedAgents = preferred ? [preferred] : agents.slice(0, 1);
      syncPrimaryAgentSelect();
      populateCompareModeOptions();

      const metrics = DATA.stats?.metrics || [];
      $('metricSelect').innerHTML = metrics.map((m) => `<option value="${escapeHTML(m)}">${escapeHTML(m)}</option>`).join('');
      if (metrics.includes('score')) $('metricSelect').value = 'score';
      updatePartnerSelect();
      renderAgentButtons();
      renderMetricButtons();
    }

    function updatePartnerSelect() {
      const partners = uniqueSorted([
        ...(DATA.negotiation?.negotiations || []).filter(rowMatchesSelectedAgents).flatMap(otherPartiesForSelected),
        ...(DATA.negotiation?.actions || []).filter(rowMatchesSelectedAgents).flatMap(otherPartiesForSelected),
      ]);
      const current = $('partnerSelect').value;
      $('partnerSelect').innerHTML = '<option value="__ALL__">全員</option>' + partners.map((p) => `<option value="${escapeHTML(p)}">${escapeHTML(p)}</option>`).join('');
      if (partners.includes(current)) $('partnerSelect').value = current;
      else $('partnerSelect').value = '__ALL__';
      renderPartnerButtons();
    }

    function drawLineChart(canvas, labels, series, options = {}) {
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(220, Math.floor(rect.height * dpr));
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const pad = { left: 58, right: 18, top: 18, bottom: 42 };
      const innerW = Math.max(1, w - pad.left - pad.right);
      const innerH = Math.max(1, h - pad.top - pad.bottom);
      const allValues = [];
      series.forEach((s) => (s.values || []).forEach((v) => { const n = num(v); if (n !== null) allValues.push(n); }));
      if (!allValues.length || !series.length) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '14px sans-serif';
        ctx.fillText('表示できる数値データがありません', pad.left, pad.top + 24);
        return;
      }
      let minY = Math.min(...allValues);
      let maxY = Math.max(...allValues);
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const yPad = (maxY - minY) * 0.08;
      minY -= yPad; maxY += yPad;
      const xAt = (i) => pad.left + (labels.length <= 1 ? innerW / 2 : (i / (labels.length - 1)) * innerW);
      const yAt = (v) => pad.top + (1 - (v - minY) / (maxY - minY)) * innerH;

      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= 5; i++) {
        const y = pad.top + (i / 5) * innerH;
        const value = maxY - (i / 5) * (maxY - minY);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillText(nf.format(value), pad.left - 8, y);
      }
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const tickCount = Math.min(8, labels.length);
      for (let i = 0; i < tickCount; i++) {
        const idx = Math.round(i * (labels.length - 1) / Math.max(1, tickCount - 1));
        const x = xAt(idx);
        ctx.fillText(String(labels[idx]), x, h - pad.bottom + 14);
      }
      ctx.strokeStyle = '#cbd5e1';
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.stroke();

      series.forEach((s, si) => {
        ctx.strokeStyle = s.color || colors[si % colors.length];
        ctx.lineWidth = options.thin ? 1.2 : 2.2;
        ctx.globalAlpha = options.thin ? 0.72 : 1;
        ctx.beginPath();
        let started = false;
        (s.values || []).forEach((v, i) => {
          const n = num(v);
          if (n === null) { started = false; return; }
          const x = xAt(i), y = yAt(n);
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }

    function drawBarChart(canvas, labels, values) {
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(220, Math.floor(rect.height * dpr));
      ctx.scale(dpr, dpr);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const pad = { left: 48, right: 14, top: 18, bottom: 42 };
      const innerW = Math.max(1, w - pad.left - pad.right);
      const innerH = Math.max(1, h - pad.top - pad.bottom);
      const maxY = Math.max(1, ...values);
      ctx.strokeStyle = '#e5e7eb';
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= 5; i++) {
        const y = pad.top + (i / 5) * innerH;
        const value = maxY - (i / 5) * maxY;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillText(nf.format(value), pad.left - 8, y);
      }
      const gap = 2;
      const barW = Math.max(2, innerW / Math.max(1, labels.length) - gap);
      ctx.fillStyle = '#2563eb';
      values.forEach((v, i) => {
        const x = pad.left + i * (innerW / Math.max(1, labels.length)) + gap / 2;
        const bh = (v / maxY) * innerH;
        const y = pad.top + innerH - bh;
        ctx.fillRect(x, y, barW, bh);
      });
      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const tickCount = Math.min(10, labels.length);
      for (let i = 0; i < tickCount; i++) {
        const idx = Math.round(i * (labels.length - 1) / Math.max(1, tickCount - 1));
        const x = pad.left + (idx + 0.5) * (innerW / Math.max(1, labels.length));
        ctx.fillText(String(labels[idx]), x, h - pad.bottom + 14);
      }
    }

    function drawStackedContractChart(canvas, labels, rows, yMax) {
      const ctx = canvas.getContext('2d');
      // たくさんの横長canvasを事前生成するため、契約グラフだけはdpr=1に固定して軽くする。
      const dpr = 1;
      const pad = { left: 52, right: 18, top: 30, bottom: 42 };
      const innerW = Math.max(1, labels.length * CONTRACT_CHART_STEP_WIDTH);
      const fallbackW = Math.max(
        320,
        Math.min(1460, (document.querySelector('main')?.clientWidth || window.innerWidth || 980) - 72)
      );
      const parentW = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
      const cssWidth = Math.max(320, fallbackW, parentW, pad.left + innerW + pad.right);
      const cssHeight = CONTRACT_CHART_HEIGHT;

      // step数が増えても棒を細くせず、canvas自体を横に長くして親要素で横スクロールする。
      canvas.dataset.baseCssWidth = String(cssWidth);
      canvas.style.width = `${cssWidth}px`;
      canvas.style.height = `${cssHeight}px`;
      canvas.width = Math.floor(cssWidth * dpr);
      canvas.height = Math.floor(cssHeight * dpr);
      applyContractChartZoomToCanvas(canvas);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const w = cssWidth, h = cssHeight;
      ctx.clearRect(0, 0, w, h);
      const innerH = Math.max(1, h - pad.top - pad.bottom);
      const maxY = Math.max(1, Math.ceil(yMax || 1));
      const sellColor = colors[3];
      const buyColor = colors[5];
      const yAt = (v) => pad.top + (1 - v / maxY) * innerH;

      ctx.strokeStyle = '#e5e7eb';
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= 5; i++) {
        const y = pad.top + (i / 5) * innerH;
        const value = maxY - (i / 5) * maxY;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillText(nf.format(value), pad.left - 8, y);
      }

      rows.forEach((row, i) => {
        const x = pad.left + i * CONTRACT_CHART_STEP_WIDTH + (CONTRACT_CHART_STEP_WIDTH - CONTRACT_CHART_BAR_WIDTH) / 2;
        let bottom = pad.top + innerH;
        const buyH = (row.buyCount / maxY) * innerH;
        if (row.buyCount > 0) {
          ctx.fillStyle = buyColor;
          ctx.fillRect(x, bottom - buyH, CONTRACT_CHART_BAR_WIDTH, buyH);
          if (buyH > 16) {
            ctx.fillStyle = 'white';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(row.buyCount), x + CONTRACT_CHART_BAR_WIDTH / 2, bottom - buyH / 2);
          }
          bottom -= buyH;
        }
        const sellH = (row.sellCount / maxY) * innerH;
        if (row.sellCount > 0) {
          ctx.fillStyle = sellColor;
          ctx.fillRect(x, bottom - sellH, CONTRACT_CHART_BAR_WIDTH, sellH);
          if (sellH > 16) {
            ctx.fillStyle = 'white';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(row.sellCount), x + CONTRACT_CHART_BAR_WIDTH / 2, bottom - sellH / 2);
          }
        }
        if ((row.sellCount + row.buyCount) > 0) {
          const totalTop = yAt(row.sellCount + row.buyCount);
          ctx.fillStyle = '#374151';
          ctx.font = '10px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(`S${row.sellQty}/B${row.buyQty}`, x + CONTRACT_CHART_BAR_WIDTH / 2, Math.max(10, totalTop - 3));
        }
      });

      ctx.strokeStyle = '#cbd5e1';
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.stroke();

      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const labelInterval = labels.length > 80 ? 10 : labels.length > 40 ? 5 : labels.length > 20 ? 2 : 1;
      labels.forEach((label, i) => {
        if (i % labelInterval !== 0 && i !== labels.length - 1) return;
        const x = pad.left + i * CONTRACT_CHART_STEP_WIDTH + CONTRACT_CHART_STEP_WIDTH / 2;
        ctx.fillText(String(label), x, h - pad.bottom + 14);
      });
    }

    function contractRoleFallback(c, agent) {
      if (c.agent0 === agent) return 'sell';
      if (c.agent1 === agent) return 'buy';
      return '';
    }

    function agentLayerNumber(agentName) {
      // SCMLログでは、エージェント名末尾の @0, @1, @2 ... が階層を表す。
      // 数字が小さいエージェントから大きいエージェントへ販売されるため、
      // 自分の番号 < 相手の番号 なら売り、自分の番号 > 相手の番号 なら買い。
      const match = String(agentName ?? '').match(/@(\d+)\s*$/);
      return match ? Number(match[1]) : null;
    }

    function contractRoleForAgent(c, agent) {
      if (c.agent0 !== agent && c.agent1 !== agent) return '';

      const opponent = otherParty(c, agent);
      const myLayer = agentLayerNumber(agent);
      const opponentLayer = agentLayerNumber(opponent);

      if (myLayer !== null && opponentLayer !== null) {
        if (myLayer < opponentLayer) return 'sell';
        if (myLayer > opponentLayer) return 'buy';
        return '';
      }

      // @番号を読めない古いログなどでは、従来どおり agent0=売り、agent1=買いで代替する。
      return contractRoleFallback(c, agent);
    }

    function contractRoleLabel(role) {
      if (role === 'sell') return '売り';
      if (role === 'buy') return '買い';
      return '';
    }

    function baseContractStepLabels() {
      if (baseContractStepLabelsCache) return baseContractStepLabelsCache;
      const simSteps = (DATA.negotiation?.simsteps || []).map((s) => s.step).filter((s) => s !== null && s !== undefined);
      if (simSteps.length) {
        baseContractStepLabelsCache = uniqueSorted(simSteps.map(Number)).sort((a, b) => a - b);
        return baseContractStepLabelsCache;
      }
      const maxStep = Math.max(0, ...(DATA.negotiation?.negotiations || []).map((n) => Number(n.sim_step)).filter(Number.isFinite));
      baseContractStepLabelsCache = Array.from({ length: maxStep + 1 }, (_, i) => i);
      return baseContractStepLabelsCache;
    }

    function contractRowsCacheKey(agent, partner = '__ALL__') {
      return `${agent}|||${partner}`;
    }

    function contractRowsForAgent(agent, partner = '__ALL__') {
      const key = contractRowsCacheKey(agent, partner);
      if (contractRowsCache.has(key)) return contractRowsCache.get(key);
      const rows = (DATA.negotiation?.negotiations || []).filter((n) => {
        if (!n.has_agreement) return false;
        if (n.agent0 !== agent && n.agent1 !== agent) return false;
        if (partner !== '__ALL__' && otherParty(n, agent) !== partner) return false;
        return true;
      }).sort((a, b) => (a.sim_step ?? -1) - (b.sim_step ?? -1) || (a.id ?? 0) - (b.id ?? 0));
      contractRowsCache.set(key, rows);
      return rows;
    }

    function aggregateContractsByStep(contracts, agent, baseLabels = baseContractStepLabels(), cacheKey = null) {
      if (cacheKey && contractAggregateCache.has(cacheKey)) return contractAggregateCache.get(cacheKey);
      const byStep = new Map(baseLabels.map((step) => [Number(step), {
        step: Number(step), sellCount: 0, buyCount: 0, sellQty: 0, buyQty: 0,
      }]));
      contracts.forEach((c) => {
        const step = Number(c.sim_step);
        if (!Number.isFinite(step)) return;
        if (!byStep.has(step)) byStep.set(step, { step, sellCount: 0, buyCount: 0, sellQty: 0, buyQty: 0 });
        const row = byStep.get(step);
        const role = contractRoleForAgent(c, agent);
        const qty = num(c.quantity) || 0;
        if (role === 'sell') {
          row.sellCount += 1;
          row.sellQty += qty;
        } else if (role === 'buy') {
          row.buyCount += 1;
          row.buyQty += qty;
        }
      });
      const rows = [...byStep.values()].sort((a, b) => a.step - b.step);
      if (cacheKey) contractAggregateCache.set(cacheKey, rows);
      return rows;
    }

    function contractYAxisMaxForAllAgents(partner = '__ALL__') {
      if (contractYMaxCache.has(partner)) return contractYMaxCache.get(partner);
      const labels = baseContractStepLabels();
      let maxCount = 1;
      agentNames().forEach((agent) => {
        const key = contractRowsCacheKey(agent, partner);
        const rows = aggregateContractsByStep(contractRowsForAgent(agent, partner), agent, labels, key);
        rows.forEach((row) => { maxCount = Math.max(maxCount, row.sellCount + row.buyCount); });
      });
      contractYMaxCache.set(partner, maxCount);
      return maxCount;
    }

    function agentLevelNumber(agentName) {
      return agentLayerNumber(agentName);
    }

    function availableLevelNumbers() {
      return [...new Set(Object.keys(DATA.stats?.agents || {})
        .map(agentLevelNumber)
        .filter((level) => level !== null && level !== undefined))]
        .sort((a, b) => Number(a) - Number(b));
    }

    function agentsForLevel(level) {
      return Object.keys(DATA.stats?.agents || {})
        .filter((agent) => String(agentLevelNumber(agent)) === String(level));
    }

    function compareModeLevel() {
      const value = $('compareMode')?.value || '';
      const match = value.match(/^level:(\d+)$/);
      return match ? match[1] : null;
    }

    function populateCompareModeOptions() {
      const select = $('compareMode');
      if (!select) return;
      const current = select.value || 'single';
      const levelOptions = availableLevelNumbers()
        .map((level) => `<option value="level:${escapeHTML(level)}">レベル${escapeHTML(level)}を表示</option>`)
        .join('');
      select.innerHTML =
        '<option value="single">選択エージェントのみ</option>' +
        levelOptions +
        '<option value="all">全エージェント表示</option>';
      if (Array.from(select.options).some((option) => option.value === current)) {
        select.value = current;
      } else {
        select.value = 'single';
      }
    }

    function applyCompareModeSelection() {
      const level = compareModeLevel();
      if (level !== null) {
        const levelAgents = agentsForLevel(level);
        if (levelAgents.length) {
          selectedAgents = levelAgents;
          syncPrimaryAgentSelect();
          updateAgentButtonStates();
          partnerOptionsDirty = true;
        }
      }
      renderStats(true);
    }

    function statsAgentsForCurrentMode() {
      const compare = $('compareMode').value;
      if (compare === 'all') return Object.keys(DATA.stats?.agents || {});
      const level = compareModeLevel();
      if (level !== null) return agentsForLevel(level);
      return selectedAgentNames();
    }

    function statsChartSize() {
      const steps = DATA.stats?.steps || [];
      const scroller = $('statsChartScroller');
      const containerW = scroller ? scroller.clientWidth : 0;
      const width = Math.max(320, containerW, 64 + Math.max(1, steps.length) * STATS_CHART_BASE_STEP_WIDTH * statsXAxisScale);
      return { width, height: STATS_CHART_HEIGHT };
    }

    function prepareStatsChartCanvas() {
      const size = statsChartSize();
      const canvas = $('statsChart');
      canvas.style.width = `${size.width}px`;
      canvas.style.height = `${size.height}px`;
      return size;
    }

    function statsSelectionKey() {
      const size = prepareStatsChartCanvas();
      const metric = $('metricSelect').value;
      const compare = $('compareMode').value;
      const normalize = $('normalizeCheck').checked ? '1' : '0';
      const agents = compare === 'all' ? ['__ALL__'] : statsAgentsForCurrentMode();
      return JSON.stringify({ metric, compare, normalize, agents, width: Math.round(size.width), height: Math.round(size.height) });
    }

    function copyCanvasToStatsChart(cached) {
      const target = $('statsChart');
      target.style.width = `${cached.cssWidth}px`;
      target.style.height = `${cached.cssHeight}px`;
      target.width = cached.canvas.width;
      target.height = cached.canvas.height;
      const ctx = target.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, target.width, target.height);
      ctx.drawImage(cached.canvas, 0, 0);
    }

    function cacheCurrentStatsChart(key, legendHTML) {
      const source = $('statsChart');
      const clone = document.createElement('canvas');
      clone.width = source.width;
      clone.height = source.height;
      clone.getContext('2d').drawImage(source, 0, 0);
      statsChartCache.set(key, {
        canvas: clone,
        legendHTML,
        cssWidth: parseFloat(source.style.width) || source.getBoundingClientRect().width,
        cssHeight: parseFloat(source.style.height) || source.getBoundingClientRect().height,
      });
    }

    function renderStats(force = false) {
      const cacheKey = statsSelectionKey();
      if (!force && lastStatsRenderKey === cacheKey) return;
      const cached = statsChartCache.get(cacheKey);
      if (cached) {
        copyCanvasToStatsChart(cached);
        $('statsLegend').innerHTML = cached.legendHTML;
        lastStatsRenderKey = cacheKey;
        if (!$('rankingTable').dataset.ready) renderRanking();
        return;
      }

      const metric = $('metricSelect').value;
      const compare = $('compareMode').value;
      const normalize = $('normalizeCheck').checked;
      const steps = DATA.stats?.steps || [];
      const targetAgents = statsAgentsForCurrentMode();
      const series = targetAgents.map((agent, i) => {
        let values = (((DATA.stats?.agents || {})[agent] || {}).metrics || {})[metric] || [];
        if (normalize) values = normalizeValues(values);
        return { name: agent, values, color: colors[i % colors.length] };
      }).filter((s) => s.values.length);

      drawLineChart($('statsChart'), steps, series, { thin: compare !== 'single' || series.length > 1 });
      let legendPrefix = '';
      const level = compareModeLevel();
      if (level !== null) {
        legendPrefix = `<span>レベル${escapeHTML(level)}を表示</span>`;
      }
      const legendHTML = legendPrefix + series.slice(0, 16).map((s, i) => `<span><i class="swatch" style="background:${s.color}"></i>${escapeHTML(s.name)}</span>`).join('') + (series.length > 16 ? `<span>...他 ${series.length - 16}</span>` : '');
      $('statsLegend').innerHTML = legendHTML;
      cacheCurrentStatsChart(cacheKey, legendHTML);
      lastStatsRenderKey = cacheKey;
      if (!$('rankingTable').dataset.ready) renderRanking();
    }

    function normalizeValues(values) {
      const nums = values.map(num).filter((v) => v !== null);
      if (!nums.length) return values;
      const min = Math.min(...nums), max = Math.max(...nums);
      if (min === max) return values.map((v) => num(v) === null ? null : 0);
      return values.map((v) => { const n = num(v); return n === null ? null : (n - min) / (max - min); });
    }

    function renderRanking() {
      const rows = DATA.stats?.summary || [];
      const html = `<table><thead><tr><th>agent</th><th>final score</th><th>balance</th><th>shortfall total</th><th>productivity</th></tr></thead><tbody>` +
        rows.map((r) => `<tr data-agent="${escapeHTML(r.agent)}"><td class="left">${escapeHTML(r.agent)}</td><td>${fmt(r.finalScore)}</td><td>${fmt(r.finalBalance)}</td><td>${fmt(r.shortfallQuantityTotal)}</td><td>${fmt(r.finalProductivity)}</td></tr>`).join('') +
        `</tbody></table>`;
      $('rankingTable').innerHTML = html;
      $('rankingTable').dataset.ready = '1';
      $('rankingTable').querySelectorAll('tbody tr').forEach((tr) => tr.addEventListener('click', () => {
        selectedAgents = [tr.dataset.agent];
        afterAgentSelectionChanged();
      }));
    }

    function filteredNegotiations(agreementsOnly = false) {
      const partner = $('partnerSelect').value;
      return (DATA.negotiation?.negotiations || []).filter((n) => {
        if (!rowMatchesSelectedAgents(n)) return false;
        if (!rowMatchesPartner(n, partner)) return false;
        if (agreementsOnly && !n.has_agreement) return false;
        return true;
      });
    }

    function filteredActions() {
      const partner = $('partnerSelect').value;
      const state = $('actionStateSelect').value;
      const q = $('actionSearch').value.trim().toLowerCase();
      return (DATA.negotiation?.actions || []).filter((a) => {
        if (!rowMatchesSelectedAgents(a)) return false;
        if (!rowMatchesPartner(a, partner)) return false;
        if (state !== '__ALL__' && a.state !== state) return false;
        if (q) {
          const selectedLabel = selectedAgentLabel();
          const opponents = otherPartiesForSelected(a).join(' ');
          const hay = `${a.id} ${a.neg_id} ${a.sim_step} ${a.sender} ${a.receiver} ${a.state} ${a.quantity} ${a.delivery_step} ${a.unit_price} ${selectedLabel} ${opponents}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      }).sort((a, b) => (a.sim_step ?? -1) - (b.sim_step ?? -1) || (a.neg_id ?? 0) - (b.neg_id ?? 0) || (a.neg_round ?? 0) - (b.neg_round ?? 0));
    }

    function renderPartnerSummary(force = false) {
      const cacheKey = selectedAgentNames().join('|||');
      if (!force && lastPartnerSummaryKey === cacheKey) return;
      const cached = partnerSummaryCache.get(cacheKey);
      if (cached) {
        $('partnerSummary').innerHTML = cached;
        attachPartnerSummaryEvents();
        lastPartnerSummaryKey = cacheKey;
        return;
      }
      const partners = uniqueSorted((DATA.negotiation?.negotiations || []).filter(rowMatchesSelectedAgents).flatMap(otherPartiesForSelected));
      const rows = partners.map((p) => {
        const negs = (DATA.negotiation?.negotiations || []).filter((n) => rowMatchesSelectedAgents(n) && rowMatchesPartner(n, p));
        const agreements = negs.filter((n) => n.has_agreement);
        const totalQty = agreements.reduce((s, n) => s + (num(n.quantity) || 0), 0);
        const totalValue = agreements.reduce((s, n) => s + (num(n.quantity) || 0) * (num(n.unit_price) || 0), 0);
        const avgPrice = totalQty ? totalValue / totalQty : null;
        return {
          partner: p,
          negotiations: negs.length,
          agreements: agreements.length,
          rate: negs.length ? agreements.length / negs.length : 0,
          totalQty,
          avgPrice,
          totalValue,
          timeouts: negs.filter((n) => n.timedout).length,
          noAgreement: negs.filter((n) => !n.has_agreement).length,
        };
      }).sort((a, b) => b.agreements - a.agreements || b.negotiations - a.negotiations);
      const html = `<table><thead><tr><th>opponent</th><th>negotiations</th><th>agreements</th><th>agreement rate</th><th>total qty</th><th>avg price</th><th>total value</th><th>timeout</th><th>no agreement</th></tr></thead><tbody>` +
        rows.map((r) => `<tr data-partner="${escapeHTML(r.partner)}"><td class="left">${escapeHTML(r.partner)}</td><td>${fmt(r.negotiations)}</td><td>${fmt(r.agreements)}</td><td>${fmt(r.rate * 100)}%</td><td>${fmt(r.totalQty)}</td><td>${fmt(r.avgPrice)}</td><td>${fmt(r.totalValue)}</td><td>${fmt(r.timeouts)}</td><td>${fmt(r.noAgreement)}</td></tr>`).join('') +
        `</tbody></table>`;
      partnerSummaryCache.set(cacheKey, html);
      $('partnerSummary').innerHTML = html;
      attachPartnerSummaryEvents();
      lastPartnerSummaryKey = cacheKey;
    }

    function attachPartnerSummaryEvents() {
      $('partnerSummary').querySelectorAll('tbody tr').forEach((tr) => tr.addEventListener('click', () => {
        $('partnerSelect').value = tr.dataset.partner;
        updatePartnerButtonStates();
        renderCurrentView();
      }));
    }

    function actionDirection(a) {
      const set = selectedAgentSet();
      const senderSelected = set.has(a.sender);
      const receiverSelected = set.has(a.receiver);
      if (senderSelected && receiverSelected) return '選択内';
      if (senderSelected) return '送信';
      if (receiverSelected) return '受信';
      return '';
    }

    function renderActionTable(force = false) {
      const cacheKey = JSON.stringify({ agents: selectedAgentNames(), partner: $('partnerSelect').value, state: $('actionStateSelect').value, q: $('actionSearch').value.trim().toLowerCase() });
      if (!force && lastActionTableKey === cacheKey) return;
      const cached = actionTableCache.get(cacheKey);
      if (cached) {
        $('actionCount').textContent = cached.countText;
        $('actionTable').innerHTML = cached.html;
        lastActionTableKey = cacheKey;
        return;
      }
      const rows = filteredActions();
      const limit = 500;
      $('actionCount').textContent = `${selectedAgentLabel()} の ${rows.length}件のアクションを表示対象にしています。表は最大${limit}件まで表示します。`;
      const html = `<table><thead><tr><th>sim step</th><th>neg id</th><th>round</th><th>direction</th><th>sender → receiver</th><th>state</th><th>quantity</th><th>delivery</th><th>price</th><th>time</th></tr></thead><tbody>` +
        rows.slice(0, limit).map((a) => {
          const direction = actionDirection(a);
          return `<tr><td>${fmt(a.sim_step)}</td><td>${fmt(a.neg_id)}</td><td>${fmt(a.neg_round)}</td><td>${direction}</td><td class="left">${escapeHTML(a.sender)} → ${escapeHTML(a.receiver)}</td><td>${statePill(a.state)}</td><td>${fmt(a.quantity)}</td><td>${fmt(a.delivery_step)}</td><td>${fmt(a.unit_price)}</td><td>${fmt(a.time)}</td></tr>`;
        }).join('') + `</tbody></table>`;
      $('actionTable').innerHTML = html;
      actionTableCache.set(cacheKey, { countText: $('actionCount').textContent, html });
      lastActionTableKey = cacheKey;
    }

    function contractRows() {
      const agent = selectedAgentNames()[0];
      const partner = $('partnerSelect').value;
      return contractRowsForAgent(agent, partner);
    }


    function contractChartKey(agent, partner = '__ALL__') {
      return `${agent}|||${partner}`;
    }

    function contractPartnersForAgent(agent) {
      // partner選択ボタンに出る可能性がある相手を、成立契約の有無に関係なく事前描画対象にする。
      // これで契約が0件の相手を選んでも、切り替え時に新規描画しない。
      const partners = uniqueSorted([
        ...(DATA.negotiation?.negotiations || [])
          .filter((n) => n.agent0 === agent || n.agent1 === agent)
          .map((n) => otherParty(n, agent)),
        ...(DATA.negotiation?.actions || [])
          .filter((a) => a.sender === agent || a.receiver === agent)
          .map((a) => otherParty(a, agent)),
      ]);
      return ['__ALL__', ...partners];
    }

    function buildContractChartPayload(agent, partner = '__ALL__') {
      const payloadKey = contractChartKey(agent, partner);
      if (contractPayloadCache.has(payloadKey)) return contractPayloadCache.get(payloadKey);
      const contracts = contractRowsForAgent(agent, partner);
      const labels = baseContractStepLabels();
      const stepRows = aggregateContractsByStep(contracts, agent, labels, payloadKey);
      const chartLabels = stepRows.map((r) => r.step);
      const yMax = contractYAxisMaxForAllAgents(partner);
      const payload = { agent, partner, contracts, labels, stepRows, chartLabels, yMax };
      contractPayloadCache.set(payloadKey, payload);
      return payload;
    }

    function ensureContractChart(agent, partner = '__ALL__') {
      const key = contractChartKey(agent, partner);
      if (contractChartCache.has(key)) return contractChartCache.get(key);

      const payload = buildContractChartPayload(agent, partner);
      const canvas = document.createElement('canvas');
      canvas.className = 'small-chart contract-chart-canvas cached-chart';
      canvas.style.display = 'none';
      canvas.dataset.contractChartKey = key;
      canvas.setAttribute('aria-label', `${agent} ${partner} contract chart`);
      $('contractChartScroller').appendChild(canvas);
      drawStackedContractChart(canvas, payload.chartLabels, payload.stepRows, payload.yMax);

      const entry = { ...payload, canvas };
      contractChartCache.set(key, entry);
      return entry;
    }

    function showContractChart(agent, partner = '__ALL__') {
      // preRenderContractCharts() で作っておいたcanvasを表示/非表示するだけ。
      // 毎回全canvasを走査すると重いので、直前に表示していたcanvasだけ非表示にする。
      const entry = contractChartCache.get(contractChartKey(agent, partner)) || ensureContractChart(agent, partner);
      if (visibleContractChartCanvas !== entry.canvas) {
        if (visibleContractChartCanvas) visibleContractChartCanvas.style.display = 'none';
        entry.canvas.style.display = 'block';
        visibleContractChartCanvas = entry.canvas;
      }
      return entry;
    }

    function preRenderContractCharts() {
      if (contractChartPreRendered) return;
      contractChartPreRendered = true;
      agentNames().forEach((agent) => {
        contractPartnersForAgent(agent).forEach((partner) => ensureContractChart(agent, partner));
      });
    }

    function clearContractChartCache() {
      contractChartCache.forEach((entry) => entry.canvas.remove());
      contractChartCache.clear();
      contractChartPreRendered = false;
      visibleContractChartCanvas = null;
    }

    function buildContractInfo(agent, partner, chartEntry) {
      const infoKey = contractChartKey(agent, partner);
      if (contractInfoCache.has(infoKey)) return contractInfoCache.get(infoKey);

      const contracts = chartEntry.contracts;
      const stepRows = chartEntry.stepRows;
      const legendHTML = [
        `<span><i class="swatch" style="background:${colors[3]}"></i>売り契約数</span>`,
        `<span><i class="swatch" style="background:${colors[5]}"></i>買い契約数</span>`,
        `<span>棒の中の数字=契約数 / S・Bラベル=売り数量・買い数量</span>`,
        `<span>step数が多い場合はグラフ部分を横スクロールできます</span>`,
      ].join('');
      const noteText = `${agent} / ${partner === '__ALL__' ? '全交渉相手' : partner} の成立契約を sim_step ごとに集計しています。縦軸の最大値は全エージェント共通です。契約グラフはページ読み込み時に全エージェント・全交渉相手分を事前描画し、切り替え時は表示/非表示だけを変更します。`;

      const sellCount = stepRows.reduce((s, r) => s + r.sellCount, 0);
      const buyCount = stepRows.reduce((s, r) => s + r.buyCount, 0);
      const sellQty = stepRows.reduce((s, r) => s + r.sellQty, 0);
      const buyQty = stepRows.reduce((s, r) => s + r.buyQty, 0);
      const totalQty = sellQty + buyQty;
      const sellValue = contracts.reduce((s, c) => {
        if (contractRoleForAgent(c, agent) !== 'sell') return s;
        return s + (num(c.quantity) || 0) * (num(c.unit_price) || 0);
      }, 0);
      const buyValue = contracts.reduce((s, c) => {
        if (contractRoleForAgent(c, agent) !== 'buy') return s;
        return s + (num(c.quantity) || 0) * (num(c.unit_price) || 0);
      }, 0);
      const sellAvgPrice = sellQty ? sellValue / sellQty : null;
      const buyAvgPrice = buyQty ? buyValue / buyQty : null;
      const activeSteps = stepRows.filter((r) => (r.sellCount + r.buyCount) > 0).length;
      const kpiHTML = [
        ['契約数', contracts.length],
        ['売り契約数', sellCount],
        ['買い契約数', buyCount],
        ['合計数量', totalQty],
        ['売り数量', sellQty],
        ['買い数量', buyQty],
        ['売り平均単価', sellAvgPrice],
        ['買い平均単価', buyAvgPrice],
        ['契約があったstep数', activeSteps],
      ].map(([name, value]) => `<div class="kpi"><div class="name">${escapeHTML(name)}</div><div class="value">${fmt(value)}</div></div>`).join('');

      const nonZeroSteps = stepRows.filter((r) => (r.sellCount + r.buyCount) > 0);
      const stepTableHTML = `<table><thead><tr><th>sim step</th><th>売り契約数</th><th>売り数量</th><th>買い契約数</th><th>買い数量</th><th>合計契約数</th><th>合計数量</th></tr></thead><tbody>` +
        nonZeroSteps.map((r) => `<tr><td>${fmt(r.step)}</td><td>${fmt(r.sellCount)}</td><td>${fmt(r.sellQty)}</td><td>${fmt(r.buyCount)}</td><td>${fmt(r.buyQty)}</td><td>${fmt(r.sellCount + r.buyCount)}</td><td>${fmt(r.sellQty + r.buyQty)}</td></tr>`).join('') +
        `</tbody></table>`;

      const countText = `${agent} に関係する成立契約が ${contracts.length}件あります。`;
      const contractTableHTML = `<table><thead><tr><th>sim step</th><th>neg id</th><th>opponent</th><th>売買区分</th><th>quantity</th><th>delivery</th><th>unit price</th><th>value</th><th>product</th><th>trading price</th><th>round</th></tr></thead><tbody>` +
        contracts.map((c) => {
          const opponent = otherParty(c, agent);
          const role = contractRoleForAgent(c, agent);
          const side = `${contractRoleLabel(role)} (${c.agent0 === agent ? 'agent0' : 'agent1'})`;
          const value = (num(c.quantity) || 0) * (num(c.unit_price) || 0);
          return `<tr><td>${fmt(c.sim_step)}</td><td>${fmt(c.id)}</td><td class="left">${escapeHTML(opponent)}</td><td class="left">${escapeHTML(side)}</td><td>${fmt(c.quantity)}</td><td>${fmt(c.delivery_step)}</td><td>${fmt(c.unit_price)}</td><td>${fmt(value)}</td><td>${fmt(c.product)}</td><td>${fmt(c.trading_price)}</td><td>${fmt(c.round_step)}</td></tr>`;
        }).join('') + `</tbody></table>`;

      const info = { legendHTML, noteText, kpiHTML, stepTableHTML, countText, contractTableHTML };
      contractInfoCache.set(infoKey, info);
      return info;
    }

    function renderContractChartAndTable(force = false) {
      const agent = selectedAgentNames()[0];
      const partner = $('partnerSelect').value;
      const chartEntry = showContractChart(agent, partner);
      const infoKey = contractChartKey(agent, partner);
      if (!force && lastContractInfoKey === infoKey) return;
      const info = buildContractInfo(agent, partner, chartEntry);
      $('contractChartLegend').innerHTML = info.legendHTML;
      $('contractChartNote').textContent = info.noteText;
      $('contractKpis').innerHTML = info.kpiHTML;
      $('contractStepTable').innerHTML = info.stepTableHTML;
      $('contractCount').textContent = info.countText;
      $('contractTable').innerHTML = info.contractTableHTML;
      lastContractInfoKey = infoKey;
    }


    function renderWarnings() {
      const missing = DATA.negotiation?.missingFiles || [];
      if (!missing.length) {
        $('warningBox').innerHTML = '';
        return;
      }
      $('warningBox').innerHTML = `<div class="warning">dataフォルダ内で見つからなかったファイル: ${missing.map(escapeHTML).join(', ')}。見つかったデータだけでHTMLを生成しています。</div>`;
    }

    function renderCurrentView(force = false) {
      const selectionChanged = enforceAgentSelectionMode();
      const view = activeViewId();
      updateScopedPickers();
      updateAgentButtonStates();
      if ((selectionChanged || partnerOptionsDirty) && (view === 'negotiationView' || view === 'contractView')) {
        updatePartnerSelect();
        partnerOptionsDirty = false;
      }
      if (view === 'statsView') {
        renderStats(force);
      } else if (view === 'negotiationView') {
        renderPartnerSummary(force);
        renderActionTable(force);
      } else if (view === 'contractView') {
        renderContractChartAndTable(force);
      }
    }

    function renderAll() {
      renderWarnings();
      updateAgentButtonStates();
      updatePartnerButtonStates();
      updateMetricButtonStates();
      renderCurrentView();
    }

    function setupEvents() {
      $('agentSelect').addEventListener('change', () => { selectedAgents = [$('agentSelect').value]; afterAgentSelectionChanged(); });
      $('partnerSelect').addEventListener('change', () => { updatePartnerButtonStates(); renderCurrentView(); });
      $('metricSelect').addEventListener('change', () => { updateMetricButtonStates(); renderStats(); });
      $('compareMode').addEventListener('change', applyCompareModeSelection);
      $('normalizeCheck').addEventListener('change', () => renderStats());
      $('statsZoomRange').addEventListener('input', () => setStatsXAxisScale($('statsZoomRange').value));
      $('statsZoomInput').addEventListener('change', () => setStatsXAxisScale($('statsZoomInput').value));
      $('contractZoomRange').addEventListener('input', () => applyContractXAxisScale($('contractZoomRange').value));
      $('contractZoomInput').addEventListener('change', () => applyContractXAxisScale($('contractZoomInput').value));
      $('actionStateSelect').addEventListener('change', () => renderActionTable());
      $('actionSearch').addEventListener('input', () => renderActionTable());
      window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          // 契約分析グラフは初期描画済みcanvasの表示切替だけにするため、
          // リサイズ時もキャッシュ破棄・再描画はしない。
          if (activeViewId() === 'statsView') renderStats();
        }, 160);
      });
      document.querySelectorAll('.tab').forEach((btn) => btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
        btn.classList.add('active');
        $(btn.dataset.view).classList.add('active');
        renderCurrentView();
      }));
    }

    function init() {
      const statsAgents = Object.keys(DATA.stats?.agents || {}).length;
      const actions = DATA.negotiation?.actions?.length || 0;
      const negs = DATA.negotiation?.negotiations?.length || 0;
      $('subtitle').textContent = `generated: ${DATA.generatedAt} / stats agents: ${statsAgents} / actions: ${actions} / negotiations: ${negs}`;
      populateSelects();
      setupEvents();
      syncZoomInputs('stats', statsXAxisScale);
      syncZoomInputs('contract', contractXAxisScale);
      preRenderContractCharts();
      renderAll();
    }

    init();
  </script>
</body>
</html>
'''
    return template.replace("__APP_DATA__", data_json)


def resolve_path(base_dir: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def find_stats_path(base_dir: Path, stats_path: str | Path = "stats.xlsx") -> Path:
    """stats.xlsx を探す。既定名が無い場合は stats*.xlsx の先頭を使う。"""

    resolved = resolve_path(base_dir, stats_path)
    if resolved.exists():
        return resolved

    if str(stats_path) == "stats.xlsx":
        candidates = sorted(base_dir.glob("stats*.xlsx"))
        if candidates:
            return candidates[0]

    return resolved


def build_app_data(
    stats_path: str | Path = "stats.xlsx",
    data_dir: str | Path = "data",
    *,
    base_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """stats.xlsx と data/*.txt を読み込み、HTML埋め込み用データを作る。

    別ファイルからデータだけ欲しい場合にも使える。

    Args:
        stats_path: statsのxlsxパス。相対パスならbase_dir基準。
        data_dir: actions.txt等が入ったフォルダ。相対パスならbase_dir基準。
        base_dir: 相対パスの基準。省略時は現在の作業ディレクトリ。
        generated_at: HTMLに表示する生成日時。省略時は現在時刻。

    Returns:
        build_html() に渡せる辞書。

    Raises:
        FileNotFoundError: stats xlsx が見つからない場合。
    """

    root = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    resolved_stats_path = find_stats_path(root, stats_path)
    resolved_data_dir = resolve_path(root, data_dir)

    if not resolved_stats_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {resolved_stats_path}")

    rows = read_xlsx_first_sheet(resolved_stats_path)
    stats = build_stats(rows, resolved_stats_path.name)
    negotiation = build_negotiation_data(root, resolved_data_dir)
    return {
        "generatedAt": generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
        "negotiation": negotiation,
    }


def generate_html(
    stats_path: str | Path = "stats.xlsx",
    data_dir: str | Path = "data",
    *,
    base_dir: str | Path | None = None,
    generated_at: str | None = None,
) -> str:
    """SCMLログビューアHTMLを文字列として返す。ファイルには書き込まない。"""

    app_data = build_app_data(
        stats_path=stats_path,
        data_dir=data_dir,
        base_dir=base_dir,
        generated_at=generated_at,
    )
    return build_html(app_data)


def generate_scml_log_viewer(
    stats_path: str | Path = "stats.xlsx",
    output_path: str | Path = "scml_log_viewer.html",
    data_dir: str | Path = "data",
    *,
    base_dir: str | Path | None = None,
    generated_at: str | None = None,
    verbose: bool = False,
) -> GenerationResult:
    """SCMLログビューアHTMLを生成する公開API。

    ほかのPythonファイルからは、この関数を呼び出すのが一番簡単。

    Example:
        from pathlib import Path
        from make_scml_log_viewer import generate_scml_log_viewer

        result = generate_scml_log_viewer(base_dir=Path(__file__).parent)
        print(result.output_path)

    Args:
        stats_path: statsのxlsxパス。相対パスならbase_dir基準。
        output_path: 出力HTMLパス。相対パスならbase_dir基準。
        data_dir: actions.txt等が入ったフォルダ。相対パスならbase_dir基準。
        base_dir: 相対パスの基準。省略時は現在の作業ディレクトリ。
        generated_at: HTMLに表示する生成日時。省略時は現在時刻。
        verbose: Trueなら生成結果をprintする。

    Returns:
        GenerationResult。生成先や件数を含む。
    """

    root = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    resolved_stats_path = find_stats_path(root, stats_path)
    resolved_output_path = resolve_path(root, output_path)
    resolved_data_dir = resolve_path(root, data_dir)

    app_data = build_app_data(
        stats_path=resolved_stats_path,
        data_dir=resolved_data_dir,
        base_dir=root,
        generated_at=generated_at,
    )
    html = build_html(app_data)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(html, encoding="utf-8")

    stats = app_data["stats"]
    negotiation = app_data["negotiation"]
    result = GenerationResult(
        output_path=resolved_output_path,
        stats_path=resolved_stats_path,
        data_dir=resolved_data_dir,
        stats_agent_count=len(stats.get("agents", {})),
        action_count=len(negotiation.get("actions", [])),
        negotiation_count=len(negotiation.get("negotiations", [])),
        simstep_count=len(negotiation.get("simsteps", [])),
        missing_files=list(negotiation.get("missingFiles", [])),
    )

    if verbose:
        print(f"生成しました: {result.output_path}")
        print(f"stats agents: {result.stats_agent_count}")
        print(f"actions: {result.action_count}")
        print(f"negotiations: {result.negotiation_count}")
        print(f"simsteps: {result.simstep_count}")
        if result.missing_files:
            print("見つからなかったdataファイル: " + ", ".join(result.missing_files))

    return result


def generate_html_log(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stats.xlsx と data/*.txt から SCMLログビューアHTMLを生成します。")
    parser.add_argument("stats", nargs="?", default="stats.xlsx", help="入力stats xlsx。既定: stats.xlsx")
    parser.add_argument("output", nargs="?", default="scml_log_viewer.html", help="出力HTML。既定: scml_log_viewer.html")
    parser.add_argument("--data-dir", default="data", help="actions.txt等が入ったフォルダ。既定: data")
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent
    detected_stats_path = find_stats_path(base_dir, args.stats)
    if str(args.stats) == "stats.xlsx" and detected_stats_path.name != "stats.xlsx":
        print(f"stats.xlsx が見つからなかったため、代わりに {detected_stats_path.name} を読み込みます。")

    try:
        generate_scml_log_viewer(
            stats_path=detected_stats_path,
            output_path=args.output,
            data_dir=args.data_dir,
            base_dir=base_dir,
            verbose=True,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(generate_html_log())
