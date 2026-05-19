# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # 主机配置收集与对比工具
# 收集各主机配置信息，存储到本地JSON，同步到Joplin笔记

# %% [markdown]
# ## 导入模块

# %%
import pathmagic

with pathmagic.context():
    from hcm.collector import HostConfigCollector
    from hcm.imports import log, not_IPython
    from hcm.joplin_sync import update_joplin_note
    from hcm.storage import LocalStorage
    from hcm.sync_service import SyncService

# %% [markdown]
# ## 主函数

# %%
def main():
    service = SyncService(HostConfigCollector(), LocalStorage())
    return service.run()

# %% [markdown]
# ## 程序入口

# %%
if __name__ == "__main__":
    if not_IPython():
        log.info(f"开始运行文件\t{__file__}")

    success, update_record = main()

    if not_IPython():
        status = "成功" if success else "失败"
        changes = (
            f"，变化: {update_record.summary}"
            if update_record.has_changes
            else "（无变化）"
        )
        log.info(f"文件执行{status}{changes}\t{__file__}")
