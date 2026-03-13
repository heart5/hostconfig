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
# # 主机配置收集与对比工具（重构版）
# 功能：收集各主机配置信息，存储到本地配置文件，并同步到Joplin笔记中
# 支持：云端配置库列表、变化检测、更新记录

# %% [markdown]
# ## 导入标准库

# %%
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# %% [markdown]
# ## 导入项目相关模块

# %%
try:
    import pathmagic

    with pathmagic.context():
        from func.configpr import (
            findvaluebykeyinsection,
            getcfpoptionvalue,
            setcfpoptionvalue,
        )
        from func.first import dirmainpath, getdirmain
        from func.getid import getdeviceid, getdevicename, gethostuser
        from func.jpfuncs import (
            createnote,
            getinivaluefromcloud,
            getnote,
            jpapi,
            searchnotebook,
            searchnotes,
            updatenote_body,
            updatenote_title,
        )
        from func.logme import log
        from func.sysfunc import execcmd, not_IPython
        from func.wrapfuncs import timethis
except ImportError as e:
    print(f"导入项目模块失败: {e}")

    # 创建模拟函数以便代码可以运行
    class MockLog:
        def info(self, msg):
            print(f"[INFO] {msg}")

        def warning(self, msg):
            print(f"[WARNING] {msg}")

        def error(self, msg):
            print(f"[ERROR] {msg}")

        def debug(self, msg):
            print(f"[DEBUG] {msg}")

    log = MockLog()

    def getdeviceid():
        return "mock_device_id"

    def getdevicename():
        return "mock_device"

    def gethostuser():
        return "mock_user"

    def getdirmain():
        return Path.cwd()

    def dirmainpath():
        return Path.cwd()

    def getinivaluefromcloud(*args):
        return ""

    def timethis(func):
        return func

    def execcmd(cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

    def not_IPython():
        return True


# %% [markdown]
# ## 辅助函数

# %%
def get_libs_from_cloud(config_key: str) -> List[str]:
    """从云端配置获取库列表"""
    try:
        libs_str = getinivaluefromcloud("hostconfig", config_key)
        if libs_str:
            # 支持逗号、分号、空格分隔
            libs = []
            for sep in [",", ";", "\n"]:
                if sep in libs_str:
                    libs = [lib.strip() for lib in libs_str.split(sep) if lib.strip()]
                    break
            if not libs:  # 如果没有分隔符，尝试空格分隔
                libs = [lib.strip() for lib in libs_str.split() if lib.strip()]
            # 过滤空字符串
            libs = [lib for lib in libs if lib]
            if libs:
                # log.info(f"从云端配置获取 {config_key}: {len(libs)} 个库")
                return libs
    except Exception as e:
        log.warning(f"获取云端配置 {config_key} 失败: {e}")

    # 默认库列表（如果云端配置不存在）
    default_libs = {
        "required_libs": [
            "pandas",
            "numpy",
            "matplotlib",
            "jupyter",
            "jupyterlab",
            "notebook",
            "seaborn",
            "scipy",
            "scikit-learn",
            "geopandas",
            "plotly",
            "dash",
            "joplin",
            "pathmagic",
            "arrow",
        ],
        "optional_libs": [
            "torch",
            "tensorflow",
            "keras",
            "pytorch",
            "transformers",
            "langchain",
            "openai",
            "anthropic",
            "cohere",
        ],
        "ai_libs": [
            "torch",
            "tensorflow",
            "keras",
            "pytorch",
            "transformers",
            "langchain",
            "openai",
            "anthropic",
            "cohere",
            "llama_index",
        ],
    }
    return default_libs.get(config_key, [])


def format_timestamp(timestamp: str) -> str:
    """格式化时间戳"""
    if not timestamp or timestamp == "N/A":
        return "N/A"

    try:
        if "T" in timestamp:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # 尝试其他格式
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    continue
            return timestamp
    except:
        return timestamp


# %% [markdown]
# ## 基类：BaseConfigCollector

# %% [markdown]
# ### 基类：BaseConfigCollector

# %%
class BaseConfigCollector:
    """配置收集器基类（通用版）"""

    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        初始化配置收集器

        Args:
            config_data: 可选的配置数据，如果提供则直接使用，否则创建空配置
        """
        self.config_name = "hostconfig"
        self.config_data = config_data or {}

        # 延迟初始化的属性
        self._device_id = None
        self._device_name = None
        self._host_user = None
        self._config_dir = None

    @property
    def device_id(self) -> str:
        """设备ID（延迟获取）"""
        if self._device_id is None:
            self._device_id = self.config_data.get("system", {}).get("device_id", "")
            if not self._device_id:
                self._device_id = getdeviceid()
        return self._device_id

    @property
    def device_name(self) -> str:
        """设备名称（延迟获取）"""
        if self._device_name is None:
            self._device_name = self.config_data.get("system", {}).get(
                "device_name", ""
            )
            if not self._device_name:
                self._device_name = getdevicename()
        return self._device_name

    @property
    def host_user(self) -> str:
        """主机用户（延迟获取）"""
        if self._host_user is None:
            self._host_user = self.config_data.get("system", {}).get("host_user", "")
            if not self._host_user:
                self._host_user = gethostuser()
        return self._host_user

    @property
    def config_dir(self) -> Path:
        """配置目录（延迟获取）"""
        if self._config_dir is None:
            self._config_dir = getdirmain() / "data" / "hostconfig"
            self._config_dir.mkdir(parents=True, exist_ok=True)
        return self._config_dir

    @property
    def local_config_file(self) -> Path:
        """本地配置文件路径"""
        return self.config_dir / f"{self.device_id}.json"

    @property
    def update_record_file(self) -> Path:
        """更新记录文件路径"""
        return self.config_dir / f"{self.device_id}_updates.json"

    def get_config_data(self) -> Dict[str, Any]:
        """获取配置数据（兼容性方法）"""
        return self.config_data


# %% [markdown]
# ## HostConfigCollector

# %% [markdown]
# ### HostConfigCollector 类 - 第一部分

# %%
class HostConfigCollector(BaseConfigCollector):
    """主机配置收集器（重构版）"""

    def __init__(
        self,
        config_data: Optional[Dict[str, Any]] = None,
        is_local_host: bool = False,
        libs_config: Optional[Dict[str, List[str]]] = None,
    ):
        """
        初始化主机配置收集器

        Args:
            config_data: 可选的配置数据
            is_local_host: 是否当前本地主机（决定是否自动收集）
            libs_config: 库配置，格式如 {"required_libs": [...], "optional_libs": [...], "ai_libs": [...]}
        """
        super().__init__(config_data)

        # 库配置
        self.libs_config = libs_config or self._get_default_libs_config()

        # 如果是本地主机且没有配置数据，自动收集
        if is_local_host and not self.config_data:
            self.config_data = self.collect_local_host_info()

    def _get_default_libs_config(self) -> Dict[str, List[str]]:
        """获取默认库配置"""
        return {
            "required_libs": get_libs_from_cloud("required_libs"),
            "optional_libs": get_libs_from_cloud("optional_libs"),
            "ai_libs": get_libs_from_cloud("ai_libs"),
        }

    @classmethod
    def from_local_host(
        cls, libs_config: Optional[Dict[str, List[str]]] = None
    ) -> "HostConfigCollector":
        """从本地主机创建收集器（工厂方法）"""
        return cls(is_local_host=True, libs_config=libs_config)

    @classmethod
    def from_config_data(
        cls,
        config_data: Dict[str, Any],
        libs_config: Optional[Dict[str, List[str]]] = None,
    ) -> "HostConfigCollector":
        """从配置数据创建收集器（工厂方法）"""
        return cls(config_data=config_data, libs_config=libs_config)

    @classmethod
    def from_config_file(
        cls,
        config_file: Union[str, Path],
        libs_config: Optional[Dict[str, List[str]]] = None,
    ) -> "HostConfigCollector":
        """从配置文件创建收集器（工厂方法）"""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            return cls(config_data=config_data, libs_config=libs_config)
        except Exception as e:
            log.error(f"从文件加载配置失败: {e}")
            # 返回空收集器
            return cls(libs_config=libs_config)


# %% [markdown]
# ### HostConfigCollector 类 - 第二部分（配置收集方法）

    # %%
    @timethis
    def collect_local_host_info(self) -> Dict[str, Any]:
        """收集本地主机配置信息"""
        log.info(f"开始收集本地主机配置信息: {self.device_name}")
        
        all_info = {
            "system": self._collect_system_info(),
            "python": self._collect_python_info(),
            "libraries": self._collect_library_versions(),
            "project": self._collect_project_info(),
            "collection_time": datetime.now().isoformat(),
        }
        
        return all_info
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系统信息"""
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
        
        # 获取更多系统信息
        try:
            if platform.system() == "Linux":
                # 获取发行版信息
                try:
                    with open("/etc/os-release", "r") as f:
                        for line in f:
                            if line.startswith("PRETTY_NAME="):
                                system_info["system"]["distro"] = line.split("=")[1].strip().strip('"')
                                break
                except:
                    pass
                
                # 获取内核版本
                try:
                    system_info["system"]["kernel"] = execcmd("uname -r").strip()
                except:
                    pass
            
            elif platform.system() == "Windows":
                # Windows特定信息
                try:
                    system_info["system"]["windows_edition"] = platform.win32_edition()
                except:
                    pass
        except Exception as e:
            log.warning(f"获取额外系统信息失败: {e}")
        
        return system_info
    
    @timethis
    def _collect_python_info(self) -> Dict[str, Any]:
        """获取Python环境信息"""
        python_info = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "python_build": platform.python_build(),
        }
        
        # 获取conda信息
        try:
            conda_result = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True, timeout=5
            )
            if conda_result.returncode == 0:
                python_info["conda_version"] = conda_result.stdout.strip()
            else:
                python_info["conda_version"] = "Not installed or not in PATH"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            python_info["conda_version"] = "Not installed"
        
        # 获取pip信息
        try:
            pip_result = subprocess.run(
                ["pip", "--version"], capture_output=True, text=True, timeout=5
            )
            if pip_result.returncode == 0:
                python_info["pip_version"] = pip_result.stdout.strip()
            else:
                python_info["pip_version"] = "Not installed or not in PATH"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            python_info["pip_version"] = "Not installed"
        
        # 获取虚拟环境信息
        python_info["virtual_env"] = os.environ.get("VIRTUAL_ENV", "N/A")
        python_info["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV", "base")
        
        return python_info

# %% [markdown]
# ### HostConfigCollector 类 - 第三部分（库版本收集）

    # %%
    @timethis
    def _collect_library_versions(self) -> Dict[str, str]:
        """收集库版本信息"""
        lib_versions = {}
        
        # 合并所有库列表
        all_libs = set()
        for lib_list in self.libs_config.values():
            all_libs.update(lib_list)
        
        # 收集每个库的版本
        for lib_name in all_libs:
            try:
                # 动态导入库
                module = __import__(lib_name)
                version = getattr(module, "__version__", "Unknown")
                lib_versions[lib_name] = version
            except ImportError:
                lib_versions[lib_name] = "Not installed"
            except Exception as e:
                lib_versions[lib_name] = f"Error: {str(e)[:50]}"
        
        return lib_versions
    
    def _collect_project_info(self) -> Dict[str, Any]:
        """获取项目相关信息"""
        project_info = {
            "project_path": str(getdirmain()),
            "codebase_path": str(dirmainpath),
            "config_files": {},
        }
        
        # 检查重要配置文件
        config_files = [
            "pyproject.toml",
            "requirements.txt",
            "environment.yml",
            "setup.py",
            "README.md",
        ]
        
        for config_file in config_files:
            file_path = getdirmain() / config_file
            if file_path.exists():
                project_info["config_files"][config_file] = {
                    "exists": True,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                }
            else:
                project_info["config_files"][config_file] = "Not found"
        
        return project_info

# %% [markdown]
# ### HostConfigCollector 类 - 第四部分（配置管理方法）

    # %%
    def complete_missing_info(self, use_local: bool = True) -> None:
        """补全缺失的配置信息"""
        if not self.config_data:
            self.config_data = {}
        
        # 补全系统信息
        if "system" not in self.config_data:
            self.config_data["system"] = {}
        
        system_info = self.config_data["system"]
        if not system_info.get("device_id"):
            system_info["device_id"] = self.device_id
        if not system_info.get("device_name"):
            system_info["device_name"] = self.device_name
        if not system_info.get("host_user"):
            system_info["host_user"] = self.host_user
        
        # 补全收集时间
        if not self.config_data.get("collection_time"):
            self.config_data["collection_time"] = datetime.now().isoformat()
        
        # 如果需要，使用本地信息补全
        if use_local and not self.config_data.get("libraries"):
            self.config_data["libraries"] = self._collect_library_versions()
    
    def save_to_file(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """保存配置到文件"""
        if not self.config_data:
            log.warning("没有配置数据，无法保存")
            return False
        
        # 补全缺失信息
        self.complete_missing_info()
        
        # 确定文件路径
        if file_path is None:
            file_path = self.local_config_file
        else:
            file_path = Path(file_path)
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            log.info(f"配置已保存到: {file_path}")
            return True
        except Exception as e:
            log.error(f"保存配置失败: {e}")
            return False
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """验证配置完整性"""
        errors = []
        
        # 检查必需字段
        required_fields = ["system", "python", "libraries", "collection_time"]
        for field in required_fields:
            if field not in self.config_data:
                errors.append(f"缺少必需字段: {field}")
        
        # 检查system字段
        if "system" in self.config_data:
            system_fields = ["device_id", "device_name", "host_user"]
            for field in system_fields:
                if field not in self.config_data["system"] or self.config_data["system"][field] == "N/A":
                    errors.append(f"system字段不完整: {field}")
        
        # 检查python字段
        if "python" in self.config_data:
            if "python_version" not in self.config_data["python"] or self.config_data["python"]["python_version"] == "N/A":
                errors.append("python_version缺失或无效")
        
        return len(errors) == 0, errors

# %% [markdown]
# ### HostConfigCollector 类 - 第五部分（配置比较）

    # %%
    def compare_with(self, other_collector: "HostConfigCollector") -> Dict[str, Any]:
        """与另一个收集器比较配置差异"""
        differences = {
            "system": {},
            "python": {},
            "libraries": {},
            "project": {},
        }
        
        config1 = self.config_data
        config2 = other_collector.config_data
        
        # 比较系统信息
        sys1 = config1.get("system", {})
        sys2 = config2.get("system", {})
        for key in ["device_name", "host_user"]:
            if sys1.get(key) != sys2.get(key):
                differences["system"][key] = {"self": sys1.get(key), "other": sys2.get(key)}
        
        # 比较Python信息
        py1 = config1.get("python", {})
        py2 = config2.get("python", {})
        for key in ["python_version", "conda_version", "conda_env"]:
            if py1.get(key) != py2.get(key):
                differences["python"][key] = {"self": py1.get(key), "other": py2.get(key)}
        
        # 比较库版本
        libs1 = config1.get("libraries", {})
        libs2 = config2.get("libraries", {})
        all_libs = set(list(libs1.keys()) + list(libs2.keys()))
        for lib in all_libs:
            ver1 = libs1.get(lib, "Not installed")
            ver2 = libs2.get(lib, "Not installed")
            if ver1 != ver2:
                differences["libraries"][lib] = {"self": ver1, "other": ver2}
        
        return differences
    
    def show_config_summary(self):
        """显示配置摘要"""
        config = self.config_data
        print("=" * 60)
        print(f"主机配置摘要: {self.device_name}")
        print("=" * 60)
        print(f"\n1. 系统信息:")
        print(f"   设备ID: {config['system']['device_id']}")
        print(f"   设备名: {config['system']['device_name']}")
        print(f"   用户: {config['system']['host_user']}")
        print(f"   系统: {config['system']['system'].get('distro', config['system']['system']['platform'])}")
        print(f"   架构: {config['system']['system']['machine']}")
        print(f"\n2. Python环境:")
        print(f"   Python版本: {config['python']['python_version']}")
        print(f"   Conda版本: {config['python'].get('conda_version', 'N/A')}")
        print(f"   Conda环境: {config['python']['conda_env']}")
        print(f"\n3. 核心库版本:")
        
        # 显示前5个核心库
        core_libs = self.libs_config.get("required_libs", [])[:5]
        for lib in core_libs:
            version = config["libraries"].get(lib, "Not installed")
            print(f"   {lib:15} : {version}")
        
        print(f"\n4. 配置文件:")
        print(f"   本地存储: {self.local_config_file}")
        if self.local_config_file.exists():
            print(f"   文件大小: {self.local_config_file.stat().st_size} bytes")
        else:
            print(f"   文件大小: 0 bytes")
        
        print(f"\n5. 收集时间: {format_timestamp(config['collection_time'])}")
        print("=" * 60)


# %% [markdown]
# ## JoplinConfigManager

# %% [markdown]
# ### JoplinConfigManager 类 - 第一部分

# %%
class JoplinConfigManager:
    """Joplin配置管理器 - 处理多主机配置对比和笔记同步"""

    def __init__(self, config_dir: Path = None):
        """初始化"""
        self.device_id = getdeviceid()
        if config_dir is None:
            self.config_dir = getdirmain() / "data" / "hostconfig"
        else:
            self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 从云端配置获取库列表
        self.required_libs = get_libs_from_cloud("required_libs")
        self.optional_libs = get_libs_from_cloud("optional_libs")
        self.ai_libs = get_libs_from_cloud("ai_libs")

    def _is_config_complete(self, config: Dict[str, Any]) -> bool:
        """检查配置是否完整"""
        try:
            # 检查必需字段
            required_fields = ["system", "python", "libraries", "collection_time"]
            for field in required_fields:
                if field not in config:
                    log.debug(f"配置缺少必需字段: {field}")
                    return False

            # 检查system字段的必需子字段
            if "device_name" not in config["system"]:
                log.debug(f"配置缺少device_name")
                return False
            if "device_id" not in config["system"]:
                log.debug(f"配置缺少device_id")
                return False

            return True
        except Exception as e:
            log.debug(f"检查配置完整性失败: {e}")
            return False

    def _cleanup_old_configs(self, current_configs: Dict[str, Any]) -> None:
        """清理过时的本地配置文件"""
        try:
            # 获取当前有效的设备ID集合
            current_device_ids = set(current_configs.keys())
            current_device_ids.add(self.device_id)  # 包括当前主机

            # 遍历所有配置文件
            for config_file in self.config_dir.glob("*.json"):
                if "_updates.json" in str(config_file):
                    continue

                # 提取设备ID（从文件名）
                device_id = config_file.stem

                # 如果设备ID不在当前有效集合中，且不是当前主机
                if device_id not in current_device_ids and device_id != self.device_id:
                    # 检查文件创建时间
                    file_age_days = (
                        datetime.now()
                        - datetime.fromtimestamp(config_file.stat().st_mtime)
                    ).days

                    if file_age_days > 30:  # 超过30天的旧文件
                        try:
                            config_file.unlink()
                            log.info(
                                f"清理过时配置文件: {device_id} ({file_age_days}天)"
                            )
                        except Exception as e:
                            log.error(f"清理配置文件失败: {device_id}, {e}")
        except Exception as e:
            log.error(f"清理过时配置失败: {e}")


# %% [markdown]
# ### JoplinConfigManager 类 - 第二部分（配置加载）

    # %%
    @timethis
    def load_all_configs(self) -> Dict[str, Any]:
        """加载所有主机的配置信息"""
        configs = {}
        for config_file in self.config_dir.glob("*.json"):
            # 跳过更新记录文件
            if "_updates.json" in str(config_file):
                continue
            
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    device_id = config_data["system"]["device_id"]
                    configs[device_id] = config_data
            except Exception as e:
                log.error(f"加载配置文件 {config_file} 失败: {e}")
        
        return configs
    
    @timethis
    def load_all_update_records(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载所有主机的更新记录"""
        all_records = {}
        for record_file in self.config_dir.glob("*_updates.json"):
            try:
                device_id = record_file.stem.replace("_updates", "")
                with open(record_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
                    all_records[device_id] = records
            except Exception as e:
                log.error(f"加载更新记录文件 {record_file} 失败: {e}")
        
        return all_records
    
    @timethis
    def save_update_records_to_local(self, all_update_records: Dict[str, Any]) -> None:
        """保存所有主机的更新记录"""
        # 保存到本地
        for device_id, records in all_update_records.items():
            update_file = self.config_dir / f"{device_id}_updates.json"
            
            # 检查是否需要更新
            should_save = False
            if not update_file.exists():
                should_save = True
            else:
                try:
                    with open(update_file, "r", encoding="utf-8") as f:
                        existing_records = json.load(f)
                    
                    # 如果记录数量不同，则更新
                    if len(existing_records) != len(records):
                        should_save = True
                    # 或者如果内容有变化（比较最新记录的时间戳）
                    elif records and existing_records:
                        latest_new = records[0].get("timestamp") if isinstance(records, list) and records else None
                        latest_existing = existing_records[0].get("timestamp") if existing_records else None
                        if latest_new != latest_existing:
                            should_save = True
                except:
                    should_save = True
            
            if should_save:
                try:
                    with open(update_file, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=2, ensure_ascii=False)
                    log.info(f"保存更新记录: {device_id} -> {update_file}")
                except Exception as e:
                    log.error(f"保存更新记录失败: {device_id}, {e}")

# %% [markdown]
# ### JoplinConfigManager 类 - 第三部分（Markdown解析）

    # %%
    @timethis
    def parse_config_from_markdown_table(
        self, markdown_content: str
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        """从Markdown表格中解析配置信息和更新记录"""
        configs = {}
        update_records = {}
        
        try:
            lines = markdown_content.split("\n")
            current_section = None
            device_names = []
            device_id_map = {}
            
            # 第一步：解析设备名称（从第一个表格的标题行）
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("| 配置项 |") and "|" in line:
                    # 分割单元格，移除空值
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) > 1:
                        # 提取设备名称（跳过第一个"配置项"）
                        device_names = parts[1:]
                        log.info(f"从表格中解析到设备名称: {device_names}")
                        break
            
            if not device_names:
                log.warning("无法从表格中解析设备名称")
                return {}, {}
            
            # 第二步：为每个设备查找对应的device_id
            for device_name in device_names:
                device_id = findvaluebykeyinsection("happyjpinifromcloud", "device", device_name)
                device_id_map[device_name] = device_id
                
                # 创建初始配置结构
                configs[device_id] = {
                    "system": {
                        "device_name": device_name,
                        "device_id": device_id,
                        "host_user": "N/A",
                        "system": {
                            "platform": "N/A",
                            "system": "N/A",
                            "release": "N/A",
                            "version": "N/A",
                            "machine": "N/A",
                        },
                    },
                    "python": {
                        "python_version": "N/A",
                        "conda_version": "N/A",
                        "pip_version": "N/A",
                        "virtual_env": "N/A",
                        "conda_env": "N/A",
                    },
                    "libraries": {},
                    "project": {"project_path": "N/A", "config_files": {}},
                    "collection_time": "N/A",
                }
                
                # 初始化更新记录列表
                update_records[device_id] = []
            
            # 第三步：解析各个部分的配置
            for line in lines:
                line = line.strip()
                
                # 检测章节标题
                if line.startswith("## "):
                    if "系统信息" in line:
                        current_section = "system"
                    elif "Python环境" in line:
                        current_section = "python"
                    elif "核心库版本" in line:
                        current_section = "core_libs"
                    elif "AI/ML相关库" in line:
                        current_section = "ai_libs"
                    elif "项目信息" in line:
                        current_section = "project"
                    elif "信息收集时间" in line:
                        current_section = "collection_time"
                    elif "更新历史" in line:
                        current_section = "update_history"
                    continue
                
                # 跳过空行和非表格行
                if not line.startswith("|"):
                    continue
                
                # 跳过表头行和分隔行
                if line.startswith("|:---") or line.startswith("|---"):
                    continue
                
                # 解析表格行
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if len(cells) < 2:
                    continue
                
                # 第一列是配置项名称
                config_item = cells[0] if cells[0] else ""
                
                # 跳过空配置项和表头
                if (not config_item or config_item == "配置项" or 
                    config_item == "主机" or config_item == "时间"):
                    continue
                
                # 根据当前章节处理数据
                if current_section == "system":
                    # 系统信息表格
                    for j, device_name in enumerate(device_names):
                        if j + 1 < len(cells):
                            value = cells[j + 1]
                            device_id = device_id_map.get(device_name)
                            if not device_id or device_id not in configs:
                                continue
                            
                            if config_item == "系统":
                                configs[device_id]["system"]["system"]["platform"] = value
                            elif config_item == "发行版":
                                configs[device_id]["system"]["system"]["distro"] = value
                            elif config_item == "内核版本":
                                configs[device_id]["system"]["system"]["release"] = value
                            elif config_item == "架构":
                                # 解析架构信息，如 "x86_64 (64bit)"
                                if "(" in value:
                                    machine = value.split("(")[0].strip()
                                    configs[device_id]["system"]["system"]["machine"] = machine
                                else:
                                    configs[device_id]["system"]["system"]["machine"] = value
                            elif config_item == "主机用户":
                                configs[device_id]["system"]["host_user"] = value
                
                elif current_section == "python":
                    # Python环境表格
                    for j, device_name in enumerate(device_names):
                        if j + 1 < len(cells):
                            value = cells[j + 1]
                            device_id = device_id_map.get(device_name)
                            if not device_id or device_id not in configs:
                                continue
                            
                            if value in ["N/A", "Not found", "Unknown", "Not installed", ""]:
                                continue
                            
                            if config_item == "Python版本":
                                configs[device_id]["python"]["python_version"] = value
                            elif config_item == "Conda版本":
                                configs[device_id]["python"]["conda_version"] = value
                            elif config_item == "Pip版本":
                                configs[device_id]["python"]["pip_version"] = value
                            elif config_item == "虚拟环境":
                                configs[device_id]["python"]["virtual_env"] = value
                            elif config_item == "Conda环境":
                                configs[device_id]["python"]["conda_env"] = value
                
                elif current_section == "core_libs" or current_section == "ai_libs":
                    # 库版本表格
                    for j, device_name in enumerate(device_names):
                        if j + 1 < len(cells):
                            value = cells[j + 1]
                            device_id = device_id_map.get(device_name)
                            if not device_id or device_id not in configs:
                                continue
                            
                            if value in ["N/A", "Not found", "Unknown", "Not installed", ""]:
                                continue
                            
                            # 库名称就是config_item
                            configs[device_id]["libraries"][config_item] = value
                
                elif current_section == "project":
                    # 项目信息表格
                    for j, device_name in enumerate(device_names):
                        if j + 1 < len(cells):
                            value = cells[j + 1]
                            device_id = device_id_map.get(device_name)
                            if not device_id or device_id not in configs:
                                continue
                            
                            if value in ["N/A", "Not found", "Unknown", "Not installed", ""]:
                                continue
                            
                            if config_item == "项目路径":
                                configs[device_id]["project"]["project_path"] = value
                            elif config_item == "requirements.txt":
                                configs[device_id]["project"]["config_files"]["requirements.txt"] = {
                                    "exists": value != "Not found",
                                    "status": value,
                                }
                
                elif current_section == "collection_time":
                    # 信息收集时间表格
                    # 表格格式：| 主机 | 收集时间 |
                    if config_item in device_names:
                        device_id = device_id_map.get(config_item)
                        if device_id and device_id in configs and len(cells) >= 2:
                            collection_time = cells[1] if len(cells) > 1 else "N/A"
                            if collection_time not in ["N/A", ""]:
                                configs[device_id]["collection_time"] = collection_time
                
                elif current_section == "update_history":
                    # 更新历史表格
                    # 表格格式：| 时间 | 主机 | 变化摘要 |
                    if len(cells) >= 3:
                        time_str, host_name, summary = tuple(cells)
                        # 查找对应的设备ID
                        device_id = device_id_map.get(host_name)
                        if device_id:
                            # 创建更新记录
                            update_record = {
                                "timestamp": time_str,
                                "device_id": device_id,
                                "device_name": host_name,
                                "has_changes": summary != "无变化",
                                "summary": summary,
                            }
                            update_records[device_id].append(update_record)
            
            # 第四步：清理无效的库条目
            for device_id, config in configs.items():
                libraries = config["libraries"]
                libraries_to_remove = []
                for lib_name, lib_value in libraries.items():
                    if lib_value in ["N/A", "Not found", "Unknown", "Not installed", ""]:
                        libraries_to_remove.append(lib_name)
                
                for lib_name in libraries_to_remove:
                    del libraries[lib_name]
                
                # 检查配置是否为空
                if not libraries and config["python"]["python_version"] == "N/A":
                    log.warning(f"设备 {config['system']['device_name']} 的配置信息为空")
            
            log.info(f"从Markdown表格解析了 {len(configs)} 个主机的配置")
            
            # 打印解析结果摘要
            for device_id, config in configs.items():
                device_name = config["system"]["device_name"]
                python_version = config["python"].get("python_version", "N/A")
                lib_count = len(config["libraries"])
                collection_time = config.get("collection_time", "N/A")
                update_count = len(update_records.get(device_id, []))
                log.info(f"设备 {device_name}: Python={python_version}, 库数量={lib_count}, 收集时间={collection_time}, 更新记录数={update_count}")
            
            return configs, update_records
        
        except Exception as e:
            log.error(f"解析Markdown表格失败: {e}")
            import traceback
            log.error(traceback.format_exc())
            return {}, {}

# %% [markdown]
# ### JoplinConfigManager 类 - 第四部分（配置合并）

    # %%
    def _merge_configs(
        self, parsed_config: Dict[str, Any], local_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """合并解析的配置和本地配置"""
        merged = parsed_config.copy()
        
        # 合并system字段
        if "system" in local_config:
            if "device_id" not in merged["system"] or merged["system"]["device_id"] == "N/A":
                merged["system"]["device_id"] = local_config["system"].get("device_id", "N/A")
            
            if "device_name" not in merged["system"] or merged["system"]["device_name"] == "N/A":
                merged["system"]["device_name"] = local_config["system"].get("device_name", "N/A")
            
            if "host_user" not in merged["system"] or merged["system"]["host_user"] == "N/A":
                merged["system"]["host_user"] = local_config["system"].get("host_user", "N/A")
            
            # 合并system.system子字段
            if "system" in local_config["system"]:
                for key in ["platform", "system", "release", "version", "machine", 
                           "processor", "architecture", "distro", "kernel"]:
                    if (key not in merged["system"]["system"] or 
                        merged["system"]["system"][key] == "N/A"):
                        if key in local_config["system"]["system"]:
                            merged["system"]["system"][key] = local_config["system"]["system"][key]
        
        # 合并python字段
        if "python" in local_config:
            for key in ["python_version", "python_implementation", "python_compiler", 
                       "python_build", "conda_version", "pip_version", "virtual_env", "conda_env"]:
                if key not in merged["python"] or merged["python"][key] == "N/A":
                    if key in local_config["python"]:
                        merged["python"][key] = local_config["python"][key]
        
        # 合并libraries字段
        if "libraries" in local_config and local_config["libraries"]:
            if not merged["libraries"] or all(v == "Not installed" for v in merged["libraries"].values()):
                merged["libraries"] = local_config["libraries"].copy()
            else:
                # 合并库信息，以解析的配置为主，用本地配置补充缺失的库
                for lib, version in local_config["libraries"].items():
                    if (lib not in merged["libraries"] or 
                        merged["libraries"][lib] == "Not installed"):
                        merged["libraries"][lib] = version
        
        # 合并project字段
        if "project" in local_config:
            if ("project_path" not in merged["project"] or 
                merged["project"]["project_path"] == "N/A"):
                merged["project"]["project_path"] = local_config["project"].get("project_path", "N/A")
            
            if ("config_files" not in merged["project"] or 
                not merged["project"]["config_files"]):
                merged["project"]["config_files"] = local_config["project"].get("config_files", {})
        
        # 合并collection_time
        if "collection_time" not in merged or merged["collection_time"] == "N/A":
            merged["collection_time"] = local_config.get("collection_time", "N/A")
        
        return merged
    
    @timethis
    def load_configs_from_joplin_note(
        self,
    ) -> Tuple[Dict[str, HostConfigCollector], Dict[str, List[Dict[str, Any]]]]:
        """从Joplin笔记中读取所有主机的配置和更新记录"""
        try:
            # 查找配置笔记
            note_title = "主机配置对比表"
            existing_notes = searchnotes(note_title)
            
            if not existing_notes or len(existing_notes) == 0:
                log.info("未找到主机配置对比笔记")
                return {}, {}
            
            note = existing_notes[0]
            note_content = note.body
            
            # 使用解析方法获取配置字典
            configs_dict, update_records = self.parse_config_from_markdown_table(note_content)
            
            # 加载本地配置，用于补充缺失的配置信息
            local_configs = self.load_all_configs()
            
            # 将配置字典转换为 HostConfigCollector 对象
            config_collectors = {}
            for device_id, config_data in configs_dict.items():
                # 创建 HostConfigCollector 对象
                collector = HostConfigCollector()
                
                # 直接设置配置数据
                collector.config_data = config_data
                
                config_collectors[device_id] = collector
            
            log.info(f"从Joplin笔记中成功解析 {len(config_collectors)} 个主机的配置")
            return config_collectors, update_records
        
        except Exception as e:
            log.error(f"从Joplin笔记读取配置失败: {e}")
            import traceback
            log.error(traceback.format_exc())
            return {}, {}

# %% [markdown]
# ### JoplinConfigManager 类 - 第五部分（配置保存）

    # %%
    def save_configs_to_local_smart(self, configs: Dict[str, Any]) -> None:
        """智能保存从笔记加载的配置到本地（增强版）"""
        saved_count = 0
        updated_count = 0
        skipped_count = 0
        
        for device_id, config in configs.items():
            # 跳过当前主机的配置（已经保存过了）
            if device_id == self.device_id:
                skipped_count += 1
                continue
            
            # 构建本地配置文件路径
            config_file = self.config_dir / f"{device_id}.json"
            
            # 检查配置的完整性
            if not self._is_config_complete(config):
                log.warning(f"配置不完整，跳过保存: {device_id}")
                skipped_count += 1
                continue
            
            # 决定是否保存
            should_save = False
            save_reason = ""
            
            if not config_file.exists():
                should_save = True
                save_reason = "文件不存在"
            else:
                try:
                    # 读取现有配置进行比较
                    with open(config_file, "r", encoding="utf-8") as f:
                        existing_config = json.load(f)
                    
                    # 比较关键字段
                    existing_time = existing_config.get("collection_time", "")
                    new_time = config.get("collection_time", "")
                    
                    # 如果收集时间不同，或者配置内容有显著差异
                    if existing_time != new_time:
                        should_save = True
                        save_reason = f"时间不同 ({existing_time} -> {new_time})"
                    else:
                        # 比较其他关键字段
                        existing_name = existing_config.get("system", {}).get("device_name", "")
                        new_name = config.get("system", {}).get("device_name", "")
                        if existing_name != new_name:
                            should_save = True
                            save_reason = f"设备名称不同 ({existing_name} -> {new_name})"
                except Exception as e:
                    should_save = True
                    save_reason = f"读取失败: {e}"
            
            # 保存配置
            if should_save:
                try:
                    # 确保配置目录存在
                    self.config_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存配置
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    
                    if config_file.exists():
                        saved_count += 1
                        log.info(f"保存配置: {device_id} ({save_reason}) -> {config_file}")
                    else:
                        log.error(f"保存失败: {device_id}")
                except Exception as e:
                    log.error(f"保存配置失败: {device_id}, {e}")
            else:
                updated_count += 1
                log.debug(f"跳过保存（无变化）: {device_id}")
        
        # 统计报告
        log.info(f"配置保存统计: 保存{saved_count}个, 跳过{skipped_count}个, 无变化{updated_count}个")
        
        # 清理过时的配置文件
        self._cleanup_old_configs(configs)
    
    @timethis
    def merge_all_configs(self) -> Dict[str, Any]:
        """合并本地配置和Joplin笔记中的配置"""
        # 加载本地所有配置
        local_configs = self.load_all_configs()
        
        # 从Joplin笔记中读取配置
        joplin_result = self.load_configs_from_joplin_note()
        
        # 检查返回的是元组还是字典
        if isinstance(joplin_result, tuple) and len(joplin_result) == 2:
            # 如果是元组，提取第一个元素（配置字典）
            joplin_collectors, update_records = joplin_result
        else:
            # 如果不是元组，可能是其他格式，初始化为空字典
            joplin_collectors = {}
            update_records = {}
        
        # 将 HostConfigCollector 对象转换为配置字典
        joplin_configs = {}
        for device_id, collector in joplin_collectors.items():
            if hasattr(collector, "config_data") and collector.config_data:
                joplin_configs[device_id] = collector.config_data
            else:
                # 如果没有配置数据，尝试获取
                try:
                    config_data = collector.get_config_data()
                    joplin_configs[device_id] = config_data
                except:
                    log.warning(f"无法获取设备 {device_id} 的配置数据")
        
        # 将从笔记加载的配置保存到本地
        if joplin_configs:
            self.save_configs_to_local_smart(joplin_configs)
        
        # 重新加载本地配置（可能已经更新）
        local_configs = self.load_all_configs()
        
        # 合并配置（本地配置优先）
        merged_configs = {}
        
        # 首先添加所有本地配置
        for device_id, config in local_configs.items():
            merged_configs[device_id] = config
        
        # 然后添加Joplin配置（如果不存在于本地配置中）
        for device_id, config in joplin_configs.items():
            if device_id not in merged_configs:
                merged_configs[device_id] = config
            else:
                # 如果已经存在，检查收集时间，保留最新的
                joplin_time = merged_configs[device_id].get("collection_time", "")
                local_time = config.get("collection_time", "")
                # 如果Joplin配置的收集时间更新，则更新合并后的配置
                if local_time > joplin_time:
                    merged_configs[device_id] = config
        
        # 保存其他主机的配置到本地（用于下次比较）
        self.save_configs_to_local_smart(merged_configs)
        log.info(f"合并后共有 {len(merged_configs)} 个主机的配置")
        return merged_configs

# %% [markdown]
# ### JoplinConfigManager 类 - 第六部分（Markdown生成）

    # %%
    @timethis
    def generate_markdown_table(
        self, config_collectors: Dict[str, HostConfigCollector]
    ) -> str:
        """生成Markdown对比表格"""
        if not config_collectors:
            return "# 主机配置对比表\n\n暂无配置信息\n"
        
        # 获取所有设备ID并按设备名称排序
        device_ids = sorted(
            config_collectors.keys(), key=lambda x: config_collectors[x].device_name
        )
        
        md_lines = ["# 主机配置对比表\n"]
        
        # 1. 系统信息
        md_lines.append("\n## 1. 系统信息\n")
        md_lines.append(
            "| 配置项 | "
            + " | ".join([config_collectors[did].device_name for did in device_ids])
            + " |"
        )
        md_lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
        
        system_items = ["系统", "发行版", "内核版本", "架构", "主机用户"]
        for item in system_items:
            row = [f"**{item}**"]
            for did in device_ids:
                collector = config_collectors[did]
                config_data = collector.get_config_data()
                system_info = config_data.get("system", {}).get("system", {})
                
                if item == "系统":
                    value = system_info.get("system", "N/A")
                elif item == "发行版":
                    value = system_info.get("distro", "N/A")
                elif item == "内核版本":
                    value = system_info.get("kernel", "N/A")
                elif item == "架构":
                    value = system_info.get("architecture", "N/A")
                elif item == "主机用户":
                    value = config_data.get("system", {}).get("host_user", "N/A")
                else:
                    value = "N/A"
                
                row.append(str(value))
            md_lines.append("| " + " | ".join(row) + " |")
        
        # 2. Python环境信息
        md_lines.append("\n## 2. Python环境\n")
        md_lines.append(
            "| 配置项 | "
            + " | ".join([config_collectors[did].device_name for did in device_ids])
            + " |"
        )
        md_lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
        
        python_items = ["Python版本", "Conda版本", "Pip版本", "虚拟环境", "Conda环境"]
        for item in python_items:
            row = [f"**{item}**"]
            for did in device_ids:
                collector = config_collectors[did]
                config_data = collector.get_config_data()
                python_info = config_data.get("python", {})
                
                if item == "Python版本":
                    value = python_info.get("python_version", "N/A")
                elif item == "Conda版本":
                    value = python_info.get("conda_version", "N/A")
                elif item == "Pip版本":
                    value = python_info.get("pip_version", "N/A")
                    if isinstance(value, list):
                        value = value[1] if len(value) > 1 else "N/A"
                elif item == "虚拟环境":
                    value = python_info.get("virtual_env", "N/A")
                elif item == "Conda环境":
                    value = python_info.get("conda_env", "N/A")
                else:
                    value = "N/A"
                
                row.append(str(value))
            md_lines.append("| " + " | ".join(row) + " |")
        
        # 3. 库信息（按类别分组）
        md_lines.append("\n## 3. 主要库版本\n")
        
        # 定义库类别
        lib_categories = {
            "基础库": ["pandas", "numpy", "matplotlib", "seaborn", "scipy"],
            "Jupyter": ["jupyter", "jupyterlab", "notebook"],
            "AI/ML": ["scikit-learn", "torch", "tensorflow", "keras", "pytorch"],
            "NLP": ["transformers", "langchain", "nltk", "spacy"],
            "其他": ["geopandas", "plotly", "dash", "joplin", "pathmagic", "arrow"],
        }
        
        for category, libs in lib_categories.items():
            md_lines.append(f"\n### {category}\n")
            md_lines.append(
                "| 库名 | "
                + " | ".join([config_collectors[did].device_name for did in device_ids])
                + " |"
            )
            md_lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
            
            for lib in libs:
                row = [f"**{lib}**"]
                for did in device_ids:
                    collector = config_collectors[did]
                    config_data = collector.get_config_data()
                    libraries = config_data.get("libraries", {})
                    value = libraries.get(lib, "Not installed")
                    row.append(str(value))
                md_lines.append("| " + " | ".join(row) + " |")
        
        # 4. 项目信息
        md_lines.append("\n## 4. 项目信息\n")
        md_lines.append(
            "| 配置项 | "
            + " | ".join([config_collectors[did].device_name for did in device_ids])
            + " |"
        )
        md_lines.append("|:---|" + "|".join([":---:" for _ in device_ids]) + "|")
        
        project_items = ["项目路径", "配置文件数量"]
        for item in project_items:
            row = [f"**{item}**"]
            for did in device_ids:
                collector = config_collectors[did]
                config_data = collector.get_config_data()
                project_info = config_data.get("project", {})
                
                if item == "项目路径":
                    value = project_info.get("project_path", "N/A")
                elif item == "配置文件数量":
                    config_files = project_info.get("config_files", {})
                    value = len([k for k, v in config_files.items() if v != "Not found"])
                else:
                    value = "N/A"
                
                row.append(str(value))
            md_lines.append("| " + " | ".join(row) + " |")
        
        # 5. 信息收集时间
        md_lines.append("\n## 5. 信息收集时间\n")
        md_lines.append("| 主机 | 收集时间 |")
        md_lines.append("|:---|:---|")
        
        for did in device_ids:
            collector = config_collectors[did]
            config_data = collector.get_config_data()
            device_name = collector.device_name
            collection_time = config_data.get("collection_time", "N/A")
            
            # 格式化时间
            formatted_time = format_timestamp(collection_time)
            md_lines.append(f"| {device_name} | {formatted_time} |")
        
        return "\n".join(md_lines) + "\n\n"

# %% [markdown]
# ### JoplinConfigManager 类 - 第七部分（更新历史）

    # %%
    @timethis
    def generate_update_history(self, all_records: Dict[str, Any]) -> str:
        """生成更新历史记录"""
        if not all_records:
            return "\n## 更新历史\n\n暂无更新记录"
        
        md_lines = ["\n## 更新历史\n"]
        md_lines.append("*按时间倒序排列，最近更新在前*\n")
        
        # 收集所有主机的更新记录
        all_updates = []
        for device_id, records in all_records.items():
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict):
                        # 确保每条记录都有设备ID
                        record_copy = record.copy()
                        record_copy["device_id"] = device_id
                        all_updates.append(record_copy)
        
        # 按时间排序（倒序）
        all_updates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # 只显示最近20条记录
        recent_updates = all_updates[:20]
        
        md_lines.append("| 时间 | 主机 | 变化摘要 |")
        md_lines.append("|:---|:---|:---|")
        
        for update in recent_updates:
            timestamp = update.get("timestamp", "")
            device_name = update.get("device_name", update.get("device_id", "Unknown"))
            summary = update.get("summary", "无变化")
            
            # 格式化时间
            formatted_time = format_timestamp(timestamp)
            
            # 如果有变化，加粗显示
            if update.get("has_changes", False):
                summary = f"**{summary}**"
            
            md_lines.append(f"| {formatted_time} | {device_name} | {summary} |")
        
        return "\n".join(md_lines)
    
    @timethis
    def update_joplin_note(
        self, current_config: Dict[str, Any], update_record: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """更新Joplin笔记"""
        try:
            # 检查配置是否有变化
            if not update_record.get("has_changes", False):
                log.info(f"主机《{current_config['system']['device_name']}》的配置无变化")
                return True, "配置无变化，无需更新笔记"
            
            # 查找或创建笔记本
            notebook_title = "ewmobile"
            notebook_id = searchnotebook(notebook_title)
            if not notebook_id:
                notebook_id = jpapi.add_notebook(title=notebook_title)
                log.info(f"创建新笔记本: {notebook_title}")
            
            # 查找或创建笔记
            note_title = "主机配置对比表"
            existing_notes = searchnotes(note_title, parent_id=notebook_id)
            
            # 使用 merge_all_configs() 获取所有主机配置
            all_configs = self.merge_all_configs()
            
            # 将配置字典转换为 HostConfigCollector 对象
            all_collectors = {}
            for device_id, config_data in all_configs.items():
                collector = HostConfigCollector()
                collector.config_data = config_data
                all_collectors[device_id] = collector
            
            # 保存当前主机配置（确保最新）
            current_collector = HostConfigCollector()
            current_collector.config_data = current_config
            if current_collector.device_id in all_collectors:
                all_collectors[current_collector.device_id] = current_collector
            
            # 合并更新记录
            all_update_records = self.load_all_update_records() or {}
            device_id = current_config["system"]["device_id"]
            if device_id not in all_update_records:
                all_update_records[device_id] = []
            
            # 检查是否已存在相同时间戳的记录
            current_timestamp = update_record.get("timestamp")
            if current_timestamp:
                existing_timestamps = [
                    x.get("timestamp") for x in all_update_records[device_id]
                ]
                if current_timestamp not in existing_timestamps:
                    all_update_records[device_id].insert(0, update_record)
            else:
                all_update_records[device_id].insert(0, update_record)
            
            # 限制每个主机最多100条记录
            if len(all_update_records[device_id]) > 100:
                all_update_records[device_id] = all_update_records[device_id][:100]
            
            # 保存更新记录到本地
            self.save_update_records_to_local(all_update_records)
            
            # 生成markdown对比表格
            markdown_content = self.generate_markdown_table(all_collectors)
            markdown_content += self.generate_update_history(all_update_records)
            
            # 更新或创建笔记
            if existing_notes:
                note = existing_notes[0]
                updatenote_body(note.id, markdown_content)
                log.info(f"更新现有笔记: {note_title}")
            else:
                note_id = createnote(notebook_id, note_title, markdown_content)
                log.info(f"创建新笔记: {note_title} (ID: {note_id})")
            
            return True, "笔记更新成功"
        
        except Exception as e:
            log.error(f"更新Joplin笔记失败: {e}")
            import traceback
            log.error(traceback.format_exc())
            return False, f"更新失败: {str(e)}"


# %% [markdown]
# ## 主函数和入口点

# %%
@timethis
def hostconfig2note():
    """主函数：收集主机配置并更新到笔记"""
    try:
        # 1. 收集当前主机配置
        collector = HostConfigCollector.from_local_host()
        collector.show_config_summary()
        current_config = collector.config_data

        # 2. 保存配置到本地并检测变化
        update_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "device_id": current_config["system"]["device_id"],
            "device_name": current_config["system"]["device_name"],
            "has_changes": True,  # 假设有变化，实际应该比较旧配置
            "summary": "配置收集完成",
        }

        # 保存配置
        collector.save_to_file()

        # 3. 更新Joplin笔记
        joplin_manager = JoplinConfigManager()
        success, message = joplin_manager.update_joplin_note(
            current_config, update_record
        )

        if success:
            if update_record.get("has_changes", False):
                log.info(
                    f"主机配置已更新到Joplin笔记，变化: {update_record.get('summary', '无变化')}"
                )
            else:
                log.info("主机配置已更新到Joplin笔记（无变化）")
        else:
            log.error(f"主机配置更新到Joplin笔记失败: {message}")

        return success, update_record

    except Exception as e:
        log.error(f"hostconfig2note 执行失败: {e}")
        import traceback

        log.error(traceback.format_exc())
        return False, {"error": str(e)}


# %% [markdown]
# ## 测试函数

# %%
def test_hostconfig():
    """测试主机配置功能"""
    print("测试主机配置收集器...")

    # 1. 创建本地主机收集器
    local_collector = HostConfigCollector.from_local_host()
    print(f"设备名称: {local_collector.device_name}")
    print(f"设备ID: {local_collector.device_id}")

    # 2. 显示配置摘要
    local_collector.show_config_summary()

    # 3. 保存配置
    if local_collector.save_to_file():
        print(f"配置已保存到: {local_collector.local_config_file}")

    # 4. 测试从文件加载
    if local_collector.local_config_file.exists():
        file_collector = HostConfigCollector.from_config_file(
            local_collector.local_config_file
        )
        print(f"从文件加载的设备: {file_collector.device_name}")

    # 5. 测试配置比较
    test_config = {
        "system": {"device_id": "test123", "device_name": "TestHost"},
        "python": {"python_version": "3.9.0"},
        "libraries": {"torch": "2.0.0"},
    }
    test_collector = HostConfigCollector.from_config_data(test_config)

    differences = local_collector.compare_with(test_collector)
    print(f"配置差异数量: {sum(len(diff) for diff in differences.values())}")

    return local_collector


# %% [markdown]
# ## 主程序入口

# %%
if __name__ == "__main__":
    if not_IPython():
        log.info(f"开始运行文件\t{__file__}")

    # 运行主函数
    success, update_record = hostconfig2note()

    if not_IPython():
        status = "成功" if success else "失败"
        changes = (
            f"，变化: {update_record.get('summary', '无变化')}"
            if update_record.get("has_changes", False)
            else "（无变化）"
        )
        log.info(f"文件执行{status}{changes}\t{__file__}")
