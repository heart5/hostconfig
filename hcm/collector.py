"""配置收集器：采集系统/Python/库/项目信息，返回 ConfigSnapshot"""

import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

from hcm.imports import execcmd, getdeviceid, getdevicename, getdirmain, gethostuser, log, timethis
from hcm.models import ConfigSnapshot, LibsConfig
from hcm.utils import get_libs_config_from_cloud


class HostConfigCollector:
    """主机配置收集器（纯收集，不持久化）"""

    def __init__(self, libs_config: LibsConfig = None):
        self.libs_config = libs_config  # None 表示惰性加载，首次 collect_all 时获取

    @timethis
    def collect_all(self) -> ConfigSnapshot:
        """收集全部主机配置信息，返回 ConfigSnapshot"""
        if self.libs_config is None:
            self.libs_config = get_libs_config_from_cloud()
        try:
            device_name = getdevicename()
        except Exception:
            device_name = getdeviceid()
        log.info(f"开始收集本地主机配置信息: {device_name}")
        return ConfigSnapshot(
            system=self._collect_system_info(),
            python=self._collect_python_info(),
            libraries=self._collect_library_versions(),
            project=self._collect_project_info(),
            collection_time=datetime.now().isoformat(),
        )

    def _collect_system_info(self) -> Dict:
        system_info = {
            "device_id": getdeviceid(),
            "device_name": getdevicename(),
            "host_user": gethostuser(),
            "timestamp": datetime.now().isoformat(),
            "system": {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
            },
        }
        try:
            if platform.system() == "Linux":
                try:
                    with open("/etc/os-release", "r") as f:
                        for line in f:
                            if line.startswith("PRETTY_NAME="):
                                system_info["system"]["distro"] = (
                                    line.split("=")[1].strip().strip('"')
                                )
                                break
                except Exception:
                    pass
                try:
                    system_info["system"]["kernel"] = execcmd("uname -r").strip()
                except Exception:
                    pass
            elif platform.system() == "Windows":
                try:
                    system_info["system"]["windows_edition"] = platform.win32_edition()
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"获取额外系统信息失败: {e}")
        return system_info

    @timethis
    def _collect_python_info(self) -> Dict:
        python_info = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "python_build": platform.python_build(),
        }
        try:
            result = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True, timeout=5
            )
            python_info["conda_version"] = (
                result.stdout.strip() if result.returncode == 0
                else "Not installed or not in PATH"
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            python_info["conda_version"] = "Not installed"
        try:
            result = subprocess.run(
                ["pip", "--version"], capture_output=True, text=True, timeout=5
            )
            python_info["pip_version"] = (
                result.stdout.strip() if result.returncode == 0
                else "Not installed or not in PATH"
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            python_info["pip_version"] = "Not installed"
        python_info["virtual_env"] = os.environ.get("VIRTUAL_ENV", "N/A")
        python_info["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV", "base")
        return python_info

    @timethis
    def _collect_library_versions(self) -> Dict[str, str]:
        lib_versions = {}
        all_libs = set()
        for lib_list in [
            self.libs_config.required_libs,
            self.libs_config.optional_libs,
            self.libs_config.ai_libs,
        ]:
            all_libs.update(lib_list)
        for lib_name in all_libs:
            try:
                if lib_name == "jupyter":
                    result = subprocess.run(
                        ["jupyter", "--version"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0:
                        for sonline in result.stdout.strip().split("\n")[1:]:
                            parts = sonline.split(":")
                            if len(parts) >= 2:
                                lib_versions[parts[0].strip()] = parts[1].strip()
                    else:
                        lib_versions["jupyter"] = "Not installed"
                else:
                    module = __import__(lib_name)
                    lib_versions[lib_name] = getattr(module, "__version__", "Unknown")
            except ImportError:
                lib_versions[lib_name] = "Not installed"
            except Exception as e:
                lib_versions[lib_name] = f"Error: {str(e)[:50]}"
        return lib_versions

    def _collect_project_info(self) -> Dict:
        project_info = {
            "project_path": str(getdirmain()),
            "codebase_path": str(getdirmain()),
            "config_files": {},
        }
        for config_file in [
            "pyproject.toml", "requirements.txt", "environment.yml",
            "setup.py", "README.md",
        ]:
            file_path = getdirmain() / config_file
            if file_path.exists():
                project_info["config_files"][config_file] = {
                    "exists": True,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                }
            else:
                project_info["config_files"][config_file] = "Not found"
        return project_info
