"""Markdown解析与生成：纯函数，零I/O"""

from typing import Any, Dict, List, Tuple

from hcm.imports import findvaluebykeyinsection, log
from hcm.models import ConfigSnapshot, UpdateRecord
from hcm.utils import format_timestamp


def parse_table(markdown_content: str) -> Tuple[Dict[str, ConfigSnapshot], Dict[str, List[UpdateRecord]]]:
    """从Markdown对比表格反向解析配置和更新记录"""
    configs: Dict[str, ConfigSnapshot] = {}
    update_records: Dict[str, List[UpdateRecord]] = {}

    try:
        lines = markdown_content.split("\n")
        current_section = None
        device_names: List[str] = []
        device_id_map: Dict[str, str] = {}

        # 第一步：解析设备名称
        for line in lines:
            line = line.strip()
            if line.startswith("| 配置项 |"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) > 1:
                    device_names = parts[1:]
                    log.info(f"从表格中解析到设备名称: {device_names}")
                    break

        if not device_names:
            log.warning("无法从表格中解析设备名称")
            return {}, {}

        # 第二步：建立设备映射
        for device_name in device_names:
            device_id = findvaluebykeyinsection("happyjpinifromcloud", "device", device_name)
            if not device_id:
                device_id = f"device_{device_name.replace(' ', '_')}"
            device_id_map[device_name] = device_id
            configs[device_id] = ConfigSnapshot(
                system={
                    "device_name": device_name,
                    "device_id": device_id,
                    "host_user": "N/A",
                    "system": {
                        "platform": "N/A", "system": "N/A", "release": "N/A",
                        "version": "N/A", "machine": "N/A", "processor": "N/A",
                        "architecture": "N/A", "distro": "N/A", "kernel": "N/A",
                    },
                },
                python={"python_version": "N/A", "conda_version": "N/A",
                         "pip_version": "N/A", "virtual_env": "N/A", "conda_env": "N/A"},
            )
            update_records[device_id] = []

        # 第三步：逐行解析
        for line in lines:
            line = line.strip()
            if line.startswith("## "):
                if "系统信息" in line: current_section = "system"
                elif "Python环境" in line: current_section = "python"
                elif "主要库版本" in line or "核心库版本" in line: current_section = "libraries"
                elif "AI/ML" in line: current_section = "ai_libs"
                elif "项目信息" in line: current_section = "project"
                elif "信息收集时间" in line: current_section = "collection_time"
                elif "更新历史" in line: current_section = "update_history"
                continue
            if not line.startswith("|") or line.startswith("|:---") or line.startswith("|---"):
                continue

            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            config_item = cells[0] if cells[0] else ""
            if not config_item or config_item in ("配置项", "主机", "时间", "库名"):
                continue

            _parse_section(current_section, config_item, cells, device_names, device_id_map,
                          configs, update_records)

        # 第四步：过滤有效配置
        valid: Dict[str, ConfigSnapshot] = {}
        for did, cfg in configs.items():
            if _has_valid_data(cfg):
                valid[did] = cfg
                log.info(f"解析到有效配置: {cfg.system['device_name']}")
        log.info(f"从Markdown表格解析了 {len(valid)} 个有效主机的配置")
        return valid, update_records

    except Exception as e:
        log.error(f"解析Markdown表格失败: {e}")
        import traceback
        log.error(traceback.format_exc())
        return {}, {}


def _parse_section(section, item, cells, device_names, device_id_map, configs, update_records):
    if section == "system":
        for j, name in enumerate(device_names):
            if j + 1 >= len(cells):
                continue
            val = cells[j + 1]
            if val in ("N/A", "Not found", "Unknown", "Not installed", ""):
                continue
            did = device_id_map.get(name)
            if not did or did not in configs:
                continue
            key = item.strip("*")
            target = configs[did].system.setdefault("system", {})
            mapping = {
                "系统": "system", "发行版": "distro", "内核版本": "kernel",
                "架构": "machine", "主机用户": "host_user", "平台": "platform",
                "系统版本": "release", "系统详细版本": "version",
                "机器类型": "machine", "处理器": "processor",
            }
            if key in mapping:
                if mapping[key] in ("host_user",):
                    configs[did].system[mapping[key]] = val
                else:
                    target[mapping[key]] = val

    elif section == "python":
        for j, name in enumerate(device_names):
            if j + 1 >= len(cells):
                continue
            val = cells[j + 1]
            if val in ("N/A", "Not found", "Unknown", "Not installed", ""):
                continue
            did = device_id_map.get(name)
            if not did or did not in configs:
                continue
            key = item.strip("*")
            mapping = {
                "Python版本": "python_version", "Conda版本": "conda_version",
                "Pip版本": "pip_version", "虚拟环境": "virtual_env", "Conda环境": "conda_env",
            }
            if key in mapping:
                configs[did].python[mapping[key]] = val

    elif section in ("libraries", "ai_libs"):
        for j, name in enumerate(device_names):
            if j + 1 >= len(cells):
                continue
            val = cells[j + 1]
            if val in ("N/A", "Not found", "Unknown", "Not installed", ""):
                continue
            did = device_id_map.get(name)
            if not did or did not in configs:
                continue
            configs[did].libraries[item.strip("*")] = val

    elif section == "project":
        for j, name in enumerate(device_names):
            if j + 1 >= len(cells):
                continue
            val = cells[j + 1]
            if val in ("N/A", "Not found", ""):
                continue
            did = device_id_map.get(name)
            if not did or did not in configs:
                continue
            if item.strip("*") == "项目路径":
                configs[did].project["project_path"] = val

    elif section == "collection_time":
        if item in device_names:
            did = device_id_map.get(item)
            if did and did in configs and len(cells) >= 2:
                t = cells[1] if len(cells) > 1 else "N/A"
                if t not in ("N/A", ""):
                    configs[did].collection_time = t

    elif section == "update_history":
        if len(cells) >= 3:
            time_str, host_name, summary = cells[0], cells[1], cells[2].strip("*")
            did = device_id_map.get(host_name)
            if did:
                if did not in update_records:
                    update_records[did] = []
                update_records[did].append(UpdateRecord(
                    timestamp=time_str, device_id=did, device_name=host_name,
                    has_changes=summary != "无变化", summary=summary,
                ))


def _has_valid_data(cfg: ConfigSnapshot) -> bool:
    if cfg.system.get("host_user", "N/A") != "N/A":
        return True
    if cfg.python.get("python_version", "N/A") != "N/A":
        return True
    if cfg.libraries:
        if any(v not in ("N/A", "Not installed", "Unknown") for v in cfg.libraries.values()):
            return True
    if cfg.collection_time != "N/A":
        return True
    return False


def generate_table(configs: Dict[str, ConfigSnapshot]) -> str:
    """生成多主机Markdown对比表格"""
    if not configs:
        return "# 主机配置对比表\n\n暂无配置信息\n"

    device_ids = sorted(configs.keys(), key=lambda x: configs[x].system.get("device_name", x))
    names = [configs[did].system.get("device_name", did) for did in device_ids]

    lines = ["# 主机配置对比表\n"]

    # 1. 系统信息
    lines.append("\n## 1. 系统信息\n")
    lines.append("| 配置项 | " + " | ".join(names) + " |")
    lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
    for item in ["系统", "发行版", "内核版本", "架构", "主机用户"]:
        row = [f"**{item}**"]
        for did in device_ids:
            sys_info = configs[did].system.get("system", {})
            if item == "系统": val = sys_info.get("system", "N/A")
            elif item == "发行版": val = sys_info.get("distro", "N/A")
            elif item == "内核版本": val = sys_info.get("kernel", "N/A")
            elif item == "架构": val = sys_info.get("architecture", "N/A")
            elif item == "主机用户": val = configs[did].system.get("host_user", "N/A")
            else: val = "N/A"
            row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    # 2. Python环境
    lines.append("\n## 2. Python环境\n")
    lines.append("| 配置项 | " + " | ".join(names) + " |")
    lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
    for item in ["Python版本", "Conda版本", "Pip版本", "虚拟环境", "Conda环境"]:
        row = [f"**{item}**"]
        for did in device_ids:
            py = configs[did].python
            if item == "Python版本": val = py.get("python_version", "N/A")
            elif item == "Conda版本": val = py.get("conda_version", "N/A")
            elif item == "Pip版本": v = py.get("pip_version", "N/A"); val = v[1] if isinstance(v, list) and len(v) > 1 else str(v)
            elif item == "虚拟环境": val = py.get("virtual_env", "N/A")
            elif item == "Conda环境": val = py.get("conda_env", "N/A")
            else: val = "N/A"
            row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    # 3. 库版本
    lines.append("\n## 3. 主要库版本\n")
    lib_categories = {
        "基础库": ["pandas", "numpy", "matplotlib", "seaborn", "scipy"],
        "Jupyter": ["IPython", "jupyter_core", "jupyterlab", "notebook"],
        "AI/ML": ["scikit-learn", "torch", "tensorflow", "keras", "pytorch"],
        "NLP": ["transformers", "langchain", "nltk", "spacy"],
        "HappyJoplin": ["joppy", "itchat", "py2ifttt", "pygsheets"],
        "其他": ["geopandas", "plotly", "dash", "pathmagic", "arrow"],
    }
    for category, libs in lib_categories.items():
        lines.append(f"\n### {category}\n")
        lines.append("| 库名 | " + " | ".join(names) + " |")
        lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
        for lib in libs:
            row = [f"**{lib}**"]
            for did in device_ids:
                row.append(str(configs[did].libraries.get(lib, "Not installed")))
            lines.append("| " + " | ".join(row) + " |")

    # 4. 项目信息
    lines.append("\n## 4. 项目信息\n")
    lines.append("| 配置项 | " + " | ".join(names) + " |")
    lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
    for item in ["项目路径", "配置文件数量"]:
        row = [f"**{item}**"]
        for did in device_ids:
            proj = configs[did].project
            if item == "项目路径": val = proj.get("project_path", "N/A")
            else:
                cfs = proj.get("config_files", {})
                val = len([k for k, v in cfs.items() if v != "Not found"])
            row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    # 5. 收集时间
    lines.append("\n## 5. 信息收集时间\n")
    lines.append("| 主机 | 收集时间 |")
    lines.append("|:---|:---|")
    for did in device_ids:
        lines.append(f"| {configs[did].system.get('device_name', did)} | {format_timestamp(configs[did].collection_time)} |")

    return "\n".join(lines) + "\n\n"


def generate_update_history(all_records: Dict[str, List[UpdateRecord]]) -> str:
    """生成更新历史Markdown"""
    if not all_records:
        return "\n## 更新历史\n\n暂无更新记录"

    all_updates = []
    for device_id, records in all_records.items():
        for r in records:
            all_updates.append(r)

    if not all_updates:
        return "\n## 更新历史\n\n暂无更新记录"

    all_updates.sort(key=lambda x: x.timestamp, reverse=True)
    recent = all_updates[:30]

    lines = ["\n## 更新历史\n", "*按时间倒序排列，最近更新在前*\n"]
    lines.append("| 时间 | 主机 | 变化摘要 |")
    lines.append("|:---|:---|:---|")
    for u in recent:
        ts = format_timestamp(u.timestamp)
        name = u.device_name or u.device_id
        summary = f"**{u.summary}**" if u.has_changes else u.summary
        lines.append(f"| {ts} | {name} | {summary} |")
    lines.append(f"\n*总计 {len(all_updates)} 条更新记录，显示最近 {len(recent)} 条*")
    return "\n".join(lines)
