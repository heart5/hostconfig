"""辅助函数"""

from datetime import datetime
from typing import List

from func.jpfuncs import getinivaluefromcloud as _getinivaluefromcloud
from hcm.imports import log
from hcm.models import LibsConfig


def _parse_lib_list(libs_str: str) -> List[str]:
    """将库列表字符串解析为列表"""
    if not libs_str:
        return []
    for sep in [",", ";", "\n"]:
        if sep in libs_str:
            return [lib.strip() for lib in libs_str.split(sep) if lib.strip()]
    return [lib.strip() for lib in libs_str.split() if lib.strip()]


def get_libs_config_from_cloud() -> LibsConfig:
    """从云端获取库配置，失败则返回默认 LibsConfig。

    readinifromcloud() 首次成功后通过时间戳缓存，后续调用为本地文件读取。
    3个 getinivaluefromcloud 包在同一个 try 内——首个网络失败直接降级，避免重试。
    """
    try:
        return LibsConfig(
            required_libs=_parse_lib_list(
                _getinivaluefromcloud("hostconfig", "required_libs")),
            optional_libs=_parse_lib_list(
                _getinivaluefromcloud("hostconfig", "optional_libs")),
            ai_libs=_parse_lib_list(
                _getinivaluefromcloud("hostconfig", "ai_libs")),
        )
    except Exception as e:
        log.warning(f"获取云端库配置失败: {e}")

    return LibsConfig(
        required_libs=[
            "pandas", "numpy", "matplotlib", "jupyter", "jupyterlab",
            "notebook", "seaborn", "scipy", "scikit-learn", "geopandas",
            "plotly", "dash", "joppy", "pathmagic",
        ],
        optional_libs=[
            "torch", "tensorflow", "keras", "pytorch", "transformers",
            "langchain", "openai", "anthropic", "cohere", "arrow",
            "itchat", "py2ifttt",
        ],
        ai_libs=[
            "torch", "tensorflow", "keras", "pytorch", "transformers",
            "langchain", "openai", "anthropic", "cohere", "llama_index",
        ],
    )


def format_timestamp(timestamp: str) -> str:
    """格式化ISO时间戳为 YYYY-MM-DD HH:MM:SS"""
    if not timestamp or timestamp == "N/A":
        return "N/A"
    try:
        if "T" in timestamp:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
            return timestamp
    except Exception:
        return timestamp
