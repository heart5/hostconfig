"""Joplin API薄封装 + 笔记更新流程"""

from typing import Any, Dict, List, Tuple

from hcm.imports import log
from hcm.markdown import generate_table, generate_update_history, parse_table
from hcm.models import ConfigSnapshot, UpdateRecord
from hcm.storage import LocalStorage


class JoplinClient:
    """Joplin Web Clipper API 薄封装"""

    @staticmethod
    def search_notes(title: str, parent_id: str = None) -> List[Any]:
        from func.jpfuncs import searchnotes
        return searchnotes(title, parent_id=parent_id)

    @staticmethod
    def get_note_body(note_id: str) -> str:
        from func.jpfuncs import getnote
        note = getnote(note_id)
        return note.body if note else ""

    @staticmethod
    def update_body(note_id: str, body: str):
        from func.jpfuncs import updatenote_body
        updatenote_body(note_id, body)

    @staticmethod
    def create_note(notebook_id: str, title: str, body: str) -> str:
        from func.jpfuncs import createnote
        return createnote(notebook_id, title, body)

    @staticmethod
    def find_or_create_notebook(title: str) -> str:
        from func.jpfuncs import jpapi, searchnotebook
        notebook_id = searchnotebook(title)
        if not notebook_id:
            notebook_id = jpapi.add_notebook(title=title)
            log.info(f"创建新笔记本: {title}")
        return notebook_id


def update_joplin_note(
    storage: LocalStorage,
    current_snapshot: ConfigSnapshot,
    update_record: UpdateRecord,
) -> Tuple[bool, str]:
    """将当前配置同步到Joplin笔记"""
    from func.jpfuncs import getinivaluefromcloud

    client = JoplinClient()
    try:
        force_update = getinivaluefromcloud("hostconfig", "FORCE_UPDATE")
        if not update_record.has_changes and not force_update:
            log.info(
                f"主机《{current_snapshot.system['device_name']}》的配置无变化，跳过笔记更新"
            )
            return True, "配置无变化，跳过笔记更新"

        existing_notes = client.search_notes("主机配置对比表")
        joplin_configs: Dict[str, ConfigSnapshot] = {}
        joplin_updates: Dict[str, List[UpdateRecord]] = {}

        if existing_notes and len(existing_notes) > 0:
            note_content = existing_notes[0].body
            joplin_configs, joplin_updates = parse_table(note_content)

        if not joplin_updates:
            joplin_updates = {}

        all_updates = dict(joplin_updates)
        device_id = current_snapshot.system["device_id"]
        if device_id not in all_updates:
            all_updates[device_id] = []
        all_updates[device_id].insert(0, update_record)
        if len(all_updates[device_id]) > 100:
            all_updates[device_id] = all_updates[device_id][:100]

        storage.save_all_update_records(all_updates)

        all_configs = dict(joplin_configs)
        if device_id not in all_configs:
            all_configs[device_id] = current_snapshot
        else:
            existing = all_configs[device_id]
            if not storage.configs_are_equal(existing, current_snapshot):
                all_configs[device_id] = storage._merge_single(existing, current_snapshot)
            else:
                log.info("配置实质相同，跳过笔记更新")
                return True, "配置无变化"

        storage.save_smart(all_configs)

        content = generate_table(all_configs) + generate_update_history(all_updates)

        notebook_id = client.find_or_create_notebook("ewmobile")
        existing = client.search_notes("主机配置对比表", parent_id=notebook_id)

        if existing:
            client.update_body(existing[0].id, content)
            log.info("更新现有笔记: 主机配置对比表")
        else:
            client.create_note(notebook_id, "主机配置对比表", content)
            log.info("创建新笔记: 主机配置对比表")

        return True, "笔记更新成功"

    except Exception as e:
        log.error(f"更新Joplin笔记失败: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False, f"更新失败: {str(e)}"
