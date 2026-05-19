"""编排器：串联收集→对比→保存→同步全流程"""

import json
from datetime import datetime
from typing import Tuple

from hcm.collector import HostConfigCollector
from hcm.imports import log
from hcm.joplin_sync import update_joplin_note
from hcm.models import ConfigSnapshot, UpdateRecord
from hcm.storage import LocalStorage
from hcm.utils import format_timestamp


class SyncService:
    """主机配置同步服务：编排收集、对比、保存、Joplin同步全流程"""

    def __init__(
        self,
        collector: HostConfigCollector,
        storage: LocalStorage,
    ):
        self.collector = collector
        self.storage = storage

    def run(self) -> Tuple[bool, UpdateRecord]:
        """执行完整同步流程"""
        log.info("开始执行主机配置同步...")

        # 1. 收集当前主机配置
        snapshot = self.collector.collect_all()
        device_name = snapshot.system.get("device_name", "unknown")
        log.info(f"配置收集完成: {device_name}")

        # 2. 与本地已保存配置对比
        has_changes, summary = self._compare_with_previous(snapshot)
        device_id = snapshot.system["device_id"]

        # 3. 构建更新记录
        record = UpdateRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            device_id=device_id,
            device_name=device_name,
            has_changes=has_changes,
            summary=summary,
        )

        # 4. 有变化时保存本地配置
        if has_changes:
            self.storage.save(snapshot)
        else:
            log.debug(f"跳过保存（无变化）: {device_id}")

        # 5. 同步到Joplin笔记
        success, message = update_joplin_note(self.storage, snapshot, record)

        if success:
            if record.has_changes:
                log.info(f"主机配置已更新到Joplin笔记，变化: {record.summary}")
            else:
                log.info("主机配置已更新到Joplin笔记（无变化）")
        else:
            log.error(f"主机配置更新到Joplin笔记失败: {message}")

        return success, record

    def _compare_with_previous(self, snapshot: ConfigSnapshot) -> Tuple[bool, str]:
        """与本地配置文件对比，返回(是否有变化, 变化摘要)"""
        device_id = snapshot.system.get("device_id", "")
        fp = self.storage.config_dir / f"{device_id}.json"

        if not fp.exists():
            return True, "首次收集"

        try:
            with open(fp, "r", encoding="utf-8") as f:
                old_snap = ConfigSnapshot.from_dict(json.load(f))
        except Exception:
            return True, "无法读取旧配置"

        old_dict = old_snap.to_dict()
        new_dict = snapshot.to_dict()

        old_dict.pop("collection_time", None)
        new_dict.pop("collection_time", None)

        if self.storage.configs_are_equal(ConfigSnapshot.from_dict(old_dict),
                                           ConfigSnapshot.from_dict(new_dict)):
            return False, "无变化"

        # 详细比较
        changes = []
        for section in ["system", "python", "libraries", "project"]:
            if old_dict.get(section) != new_dict.get(section):
                changes.append(section)

        summary = f"配置变化: {', '.join(changes)}" if changes else "配置有细微变化"
        return True, summary
