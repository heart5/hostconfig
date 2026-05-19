"""数据模型：ConfigSnapshot、UpdateRecord"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConfigSnapshot:
    """主机配置快照"""
    system: Dict[str, Any] = field(default_factory=dict)
    python: Dict[str, Any] = field(default_factory=dict)
    libraries: Dict[str, str] = field(default_factory=dict)
    project: Dict[str, Any] = field(default_factory=dict)
    collection_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "python": self.python,
            "libraries": self.libraries,
            "project": self.project,
            "collection_time": self.collection_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigSnapshot":
        return cls(
            system=data.get("system", {}),
            python=data.get("python", {}),
            libraries=data.get("libraries", {}),
            project=data.get("project", {}),
            collection_time=data.get("collection_time", ""),
        )


@dataclass
class UpdateRecord:
    """更新记录"""
    timestamp: str = ""
    device_id: str = ""
    device_name: str = ""
    has_changes: bool = False
    summary: str = ""


@dataclass
class LibsConfig:
    """待监测的库列表配置"""
    required_libs: List[str] = field(default_factory=list)
    optional_libs: List[str] = field(default_factory=list)
    ai_libs: List[str] = field(default_factory=list)
