"""本地JSON持久化：读写配置快照、更新记录、合并、智能保存"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from hcm.imports import getdirmain, log
from hcm.models import ConfigSnapshot, UpdateRecord


class LocalStorage:
    """本地JSON文件存储管理器"""

    def __init__(self, config_dir: Path = None):
        if config_dir is None:
            config_dir = getdirmain() / "data" / "hostconfig"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    # ---- 配置快照 ----

    def load(self, device_id: str) -> ConfigSnapshot:
        """加载单个设备配置"""
        fp = self.config_dir / f"{device_id}.json"
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                return ConfigSnapshot.from_dict(json.load(f))
        return ConfigSnapshot()

    def save(self, snapshot: ConfigSnapshot, file_path: Path = None) -> bool:
        """保存配置快照到JSON文件"""
        if file_path is None:
            file_path = self.config_dir / f"{snapshot.system.get('device_id', 'unknown')}.json"
        file_path = Path(file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)
            log.info(f"配置已保存到: {file_path}")
            return True
        except Exception as e:
            log.error(f"保存配置失败: {e}")
            return False

    def load_all(self) -> Dict[str, ConfigSnapshot]:
        """加载所有设备的配置"""
        configs = {}
        for fp in self.config_dir.glob("*.json"):
            if "_updates.json" in str(fp):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    device_id = data["system"]["device_id"]
                    configs[device_id] = ConfigSnapshot.from_dict(data)
            except Exception as e:
                log.error(f"加载配置文件 {fp} 失败: {e}")
        return configs

    # ---- 更新记录 ----

    def load_update_records(self, device_id: str) -> List[UpdateRecord]:
        fp = self.config_dir / f"{device_id}_updates.json"
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                return [UpdateRecord(**r) for r in json.load(f)]
        return []

    def save_update_records(self, device_id: str, records: List[UpdateRecord]):
        fp = self.config_dir / f"{device_id}_updates.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in records], f, indent=2, ensure_ascii=False)

    def load_all_update_records(self) -> Dict[str, List[UpdateRecord]]:
        all_records = {}
        for fp in self.config_dir.glob("*_updates.json"):
            try:
                device_id = fp.stem.replace("_updates", "")
                with open(fp, "r", encoding="utf-8") as f:
                    all_records[device_id] = [
                        UpdateRecord(**r) for r in json.load(f)
                    ]
            except Exception as e:
                log.error(f"加载更新记录 {fp} 失败: {e}")
        return all_records

    def save_all_update_records(self, all_records: Dict[str, List[UpdateRecord]]):
        """保存所有设备的更新记录（智能去重）"""
        for device_id, records in all_records.items():
            fp = self.config_dir / f"{device_id}_updates.json"
            should_save = False
            if not fp.exists():
                should_save = True
            else:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if len(existing) != len(records):
                        should_save = True
                    elif records and existing:
                        if records[0].timestamp != existing[0].get("timestamp"):
                            should_save = True
                except Exception:
                    should_save = True
            if should_save:
                self.save_update_records(device_id, records)

    # ---- 合并 ----

    def merge(
        self, parsed: Dict[str, ConfigSnapshot], local: Dict[str, ConfigSnapshot]
    ) -> Dict[str, ConfigSnapshot]:
        """合并解析配置和本地配置（本地优先）"""
        merged = {}
        all_ids = set(parsed.keys()) | set(local.keys())
        for device_id in all_ids:
            parsed_snap = parsed.get(device_id)
            local_snap = local.get(device_id)
            if parsed_snap is None:
                merged[device_id] = local_snap
            elif local_snap is None:
                merged[device_id] = parsed_snap
            else:
                merged[device_id] = self._merge_single(parsed_snap, local_snap)
        return merged

    def _merge_single(self, parsed: ConfigSnapshot, local: ConfigSnapshot) -> ConfigSnapshot:
        """单设备配置合并（本地优先）"""
        result = ConfigSnapshot.from_dict(parsed.to_dict())

        # system
        for key in ["device_id", "device_name", "host_user"]:
            if (key not in result.system or result.system.get(key) in (None, "N/A", "")):
                result.system[key] = local.system.get(key, "N/A")
        for key in ["platform", "system", "release", "version", "machine",
                     "processor", "architecture", "distro", "kernel"]:
            if (key not in result.system.get("system", {})
                    or result.system.get("system", {}).get(key) in (None, "N/A", "")):
                result.system.setdefault("system", {})
                result.system["system"][key] = local.system.get("system", {}).get(key, "N/A")

        # python
        for key in ["python_version", "python_implementation", "python_compiler",
                     "python_build", "conda_version", "pip_version", "virtual_env", "conda_env"]:
            if key not in result.python or result.python.get(key) in (None, "N/A", ""):
                result.python[key] = local.python.get(key, "N/A")

        # libraries
        if local.libraries and (
            not result.libraries
            or all(v == "Not installed" for v in result.libraries.values())
        ):
            result.libraries = dict(local.libraries)
        else:
            for lib, ver in local.libraries.items():
                if lib not in result.libraries or result.libraries.get(lib) == "Not installed":
                    result.libraries[lib] = ver

        # project
        if not result.project.get("project_path") or result.project["project_path"] == "N/A":
            result.project["project_path"] = local.project.get("project_path", "N/A")
        if not result.project.get("config_files"):
            result.project["config_files"] = local.project.get("config_files", {})

        # collection_time
        if not result.collection_time or result.collection_time == "N/A":
            result.collection_time = local.collection_time or "N/A"

        return result

    # ---- 智能保存 ----

    def save_smart(self, configs: Dict[str, ConfigSnapshot]):
        """智能保存：比较时间戳和内容，仅写有变化的"""
        saved = skipped = 0
        for device_id, snapshot in configs.items():
            if not snapshot.system.get("device_id") or not snapshot.system.get("device_name"):
                log.warning(f"配置缺少设备信息，跳过: {device_id}")
                skipped += 1
                continue
            fp = self.config_dir / f"{device_id}.json"
            should_save = False
            reason = ""
            if not fp.exists():
                should_save = True
                reason = "文件不存在"
            else:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    existing_time = existing.get("collection_time", "")
                    if snapshot.collection_time > existing_time:
                        should_save = True
                        reason = f"时间更新 ({existing_time} -> {snapshot.collection_time})"
                    elif not snapshot.collection_time or snapshot.collection_time == "N/A":
                        if any(
                            v not in ("N/A", "Not installed", "Unknown", "")
                            for section in [snapshot.system, snapshot.python, snapshot.libraries]
                            for v in (section.values() if isinstance(section, dict) else [])
                        ):
                            should_save = True
                            reason = "有实际配置数据"
                except Exception as e:
                    should_save = True
                    reason = f"读取失败: {e}"
            if should_save:
                self.save(snapshot, fp)
                saved += 1
                log.info(f"保存配置: {device_id} ({reason}) -> {fp}")
            else:
                skipped += 1
                log.debug(f"跳过保存（无变化）: {device_id}")
        log.info(f"配置保存统计: 保存{saved}个, 跳过{skipped}个")
        self._cleanup_old(configs)

    def _cleanup_old(self, current_configs: Dict[str, Any], days: int = 30):
        """清理超过N天的过期配置文件"""
        try:
            current_ids = set(current_configs.keys())
            for fp in self.config_dir.glob("*.json"):
                if "_updates.json" in str(fp):
                    continue
                device_id = fp.stem
                if device_id not in current_ids:
                    file_age = (datetime.now() - datetime.fromtimestamp(fp.stat().st_mtime)).days
                    if file_age > days:
                        try:
                            fp.unlink()
                            log.info(f"清理过期配置: {device_id} ({file_age}天)")
                        except Exception as e:
                            log.error(f"清理失败: {device_id}, {e}")
        except Exception as e:
            log.error(f"清理过期配置失败: {e}")

    # ---- 比较 ----

    @staticmethod
    def configs_are_equal(snap1: ConfigSnapshot, snap2: ConfigSnapshot) -> bool:
        """深度比较两个配置是否实质相同"""
        # system
        for key in ["device_name", "host_user"]:
            if snap1.system.get(key) != snap2.system.get(key):
                return False
        for key in ["system", "distro", "kernel"]:
            if (snap1.system.get("system", {}).get(key)
                    != snap2.system.get("system", {}).get(key)):
                return False
        # python
        for key in ["python_version", "conda_version", "conda_env"]:
            if snap1.python.get(key) != snap2.python.get(key):
                return False
        # libraries
        libs1 = {k: v for k, v in snap1.libraries.items()
                 if v not in ("Not installed", "Unknown", "N/A")}
        libs2 = {k: v for k, v in snap2.libraries.items()
                 if v not in ("Not installed", "Unknown", "N/A")}
        if libs1 != libs2:
            return False
        return True
