# hostconfig

主机配置自动搜集对比工具。收集各主机（Linux/Windows/macOS）的系统信息、Python环境、库版本和项目配置，保存到本地JSON文件，并自动同步到Joplin笔记中，生成多主机配置对比表格。

---

## 目录结构

```
hostconfig/
├── hostconfig.py         # 主程序：配置收集器 + Joplin笔记管理器
├── hostconfig.ipynb      # Jupyter Notebook 版本（jupytext 同步）
├── pathmagic.py           # 路径魔法模块，扩展 sys.path
├── pathmagic.ipynb        # Notebook 版本
├── rootfile               # 项目根目录定位文件（func 子模块依赖）
├── README.md              # 项目文档
├── LICENSE                # 许可证
├── .gitignore             # Git 忽略规则
├── .gitmodules            # Git 子模块配置（func）
├── data/                  # 数据目录
│   ├── happyjp.ini        # Joplin API 基础配置
│   ├── happyjphard.ini    # Joplin API 硬件相关配置
│   ├── happyjpinifromcloud.ini  # 云端同步的设备映射配置
│   ├── happyjpsys.ini     # 系统参数配置
│   └── hostconfig/        # 主机配置数据存储目录
│       ├── *.json          # 各设备的配置快照
│       └── *_updates.json  # 各设备的更新记录
├── log/                   # 日志目录
│   └── happyjoplin.log    # 运行日志
└── func/                  # Git 子模块：常用工具函数库
    ├── common/utils.py    # Shell 命令执行工具
    ├── configpr.py        # INI 配置文件读写
    ├── datatools.py       # 数据处理工具
    ├── datetimetools.py   # 日期时间处理
    ├── evernttest.py      # 事件测试
    ├── filedatafunc.py    # 文件数据函数
    ├── first.py           # 项目路径初始化（getdirmain/dirmainpath）
    ├── getid.py           # 设备ID/名称/用户获取
    ├── jpfuncs.py         # Joplin API 封装（搜索/创建/更新笔记）
    ├── litetools.py       # 轻量工具集
    ├── logme.py           # 日志模块（log.info/warning/error/debug）
    ├── nettools.py        # 网络工具
    ├── pdtools.py         # Pandas 数据处理工具
    ├── sysfunc.py         # 系统函数（execcmd/not_IPython）
    ├── termuxtools.py     # Termux 工具
    └── wrapfuncs.py       # 装饰器工具（timethis等）
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone --recurse-submodules https://github.com/heart5/hostconfig.git
cd hostconfig
```

`--recurse-submodules` 确保同时拉取 `func` 子模块。如果已克隆但缺少子模块：

```bash
git submodule update --init --recursive
```

### 2. 配置 Joplin API

在 `data/` 目录下配置 INI 文件：

- **happyjp.ini** — Joplin Web Clipper API 地址和 token
- **happyjphard.ini** — 硬件相关 Joplin 配置
- **happyjpinifromcloud.ini** — 云端设备名称到 device_id 的映射
- **happyjpsys.ini** — 系统参数（如 FORCE_UPDATE 等）

### 3. 创建定位文件

在项目根目录创建空文件 `rootfile`（func 子模块用它来定位项目根目录）：

```bash
touch rootfile
```

### 4. 运行

```bash
python hostconfig.py
```

在 Jupyter 中打开 `hostconfig.ipynb` 直接运行所有单元格亦可。

---

## 核心架构

### 类与职责

| 类名 | 职责 |
|:---|:---|
| `BaseConfigCollector` | 配置收集器基类，管理设备ID/名称/用户/配置目录/文件路径等属性 |
| `HostConfigCollector` | 主机配置收集器，继承基类，实现系统/Python/库/项目信息的收集、保存、比较 |
| `JoplinConfigManager` | Joplin笔记管理器，负责多主机配置的加载、合并、Markdown表格生成、笔记同步 |

### 关键函数

| 函数 | 说明 |
|:---|:---|
| `hostconfig2note()` | 主入口，串联收集→保存→对比→更新Joplin笔记全流程 |
| `get_libs_from_cloud(config_key)` | 从云端INI配置获取待监测的库列表，支持逗号/分号/空格/换行分隔 |
| `format_timestamp(timestamp)` | 统一格式化ISO时间戳为 `YYYY-MM-DD HH:MM:SS` |
| `collect_local_host_info()` | 收集系统信息+Python环境+库版本+项目信息，返回完整字典 |
| `save_to_file(file_path)` | 补全缺失信息后保存配置到JSON文件 |
| `compare_with(other)` | 与另一台主机的配置逐项比较差异 |
| `validate_config()` | 校验配置完整性，返回 (通过, 错误列表) |
| `generate_markdown_table(collectors)` | 生成多主机Markdown对比表格（系统/Python/库/项目/收集时间） |
| `generate_update_history(records)` | 生成所有主机的更新历史Markdown表格 |
| `update_joplin_note(config, record)` | 将最新配置同步到Joplin笔记，合并历史记录 |
| `parse_configs_updates_from_markdown_table(md)` | 从Markdown表格反向解析配置和更新记录 |
| `load_configs_updates_from_joplin_note()` | 从Joplin笔记加载所有主机配置和更新记录 |
| `merge_configs(parsed, local)` | 合并笔记配置和本地配置（本地优先） |
| `save_collectors_to_local_smart(configs)` | 智能保存配置到本地（比较时间戳和内容变化） |


```mermaid
graph TD
    A[开始运行 hostconfig.py] --> B[hostconfig2note 主函数]
    
    subgraph "1. 配置收集阶段"
        B --> C[创建 HostConfigCollector<br/>from_local_host]
        C --> D[collect_local_host_info]
        D --> E[_collect_system_info]
        D --> F[_collect_python_info]
        D --> G[_collect_library_versions]
        D --> H[_collect_project_info]
        E --> I[系统信息<br/>device_id, device_name, host_user,<br/>platform, distro, kernel等]
        F --> J[Python信息<br/>python_version, conda_version,<br/>pip_version, conda_env等]
        G --> K[库版本信息<br/>pandas, numpy, matplotlib等]
        H --> L[项目信息<br/>project_path, config_files]
        I --> M[组装 all_info 字典]
        J --> M
        K --> M
        L --> M
        M --> N[config_data 完整配置]
    end
    
    subgraph "2. 本地保存阶段"
        N --> O[save_to_file]
        O --> P[complete_missing_info]
        P --> Q[补全系统信息<br/>device_id, device_name, host_user]
        P --> R[补全收集时间]
        P --> S[补全库版本]
        Q --> T[写入JSON文件<br/>device_id.json]
    end
    
    subgraph "3. Joplin笔记更新阶段"
        N --> U[JoplinConfigManager<br/>update_joplin_note]
        U --> V[merge_all_configs]
        
        subgraph "3.1 合并配置"
            V --> W[load_all_configs<br/>加载本地所有配置]
            V --> X[load_configs_from_joplin_note]
            X --> Y[searchnotes 查找笔记]
            Y --> Z[parse_config_from_markdown_table<br/>解析Markdown表格]
            Z --> AA[提取设备名称和配置]
            AA --> AB[转换配置字典]
            W --> AC[合并配置<br/>本地配置优先]
            AB --> AC
            AC --> AD[save_configs_to_local_smart<br/>保存到本地]
        end
        
        V --> AE[生成 HostConfigCollector 对象集合]
        AE --> AF[generate_markdown_table]
        AF --> AG[生成系统信息表格]
        AF --> AH[生成Python环境表格]
        AF --> AI[生成库版本表格]
        AF --> AJ[生成项目信息表格]
        AG --> AK[Markdown内容]
        AH --> AK
        AI --> AK
        AJ --> AK
        
        U --> AL[generate_update_history]
        AL --> AM[加载所有更新记录]
        AM --> AN[合并当前更新记录]
        AN --> AO[按时间排序]
        AO --> AP[生成更新历史表格]
        
        AK --> AQ[组合完整Markdown内容]
        AP --> AQ
        AQ --> AR[updatenote_body 或 createnote]
    end
    
    subgraph "4. 数据流变化"
        style I fill:#e1f5fe
        style J fill:#f3e5f5
        style K fill:#e8f5e8
        style L fill:#fff3e0
        style N fill:#fce4ec
        style T fill:#f1f8e9
        style AA fill:#e0f2f1
        style AC fill:#f9fbe7
        style AK fill:#fff8e1
        style AQ fill:#f5f5f5
    end
    
    B --> AS[输出执行结果]
    AS --> AT[结束]
    
    subgraph "关键数据结构"
        direction LR
        subgraph "配置数据结构"
            CFG1[system<br/>├── device_id<br/>├── device_name<br/>├── host_user<br/>└── system<br/>    ├── platform<br/>    ├── distro<br/>    ├── kernel<br/>    └── ...]
            CFG2[python<br/>├── python_version<br/>├── conda_version<br/>├── pip_version<br/>└── conda_env]
            CFG3[libraries<br/>├── pandas: 2\.3\.1<br/>├── numpy: 1\.26\.4<br/>└── ...]
            CFG4[project<br/>├── project_path<br/>└── config_files]
            CFG5[collection_time]
        end
        
        subgraph "更新记录结构"
            UPD1[timestamp]
            UPD2[device_id]
            UPD3[device_name]
            UPD4[has_changes]
            UPD5[summary]
        end
    end
    
    style A fill:#4fc3f7
    style B fill:#29b6f6
    style U fill:#ff9800
    style V fill:#ffb74d
    style AR fill:#4caf50
    style AS fill:#66bb6a
    style AT fill:#81c784

```


```mermaid
stateDiagram-v2
    [*] --> 初始化
    初始化 --> 收集配置: 创建HostConfigCollector
    
    收集配置 --> 系统信息: _collect_system_info
    收集配置 --> Python环境: _collect_python_info  
    收集配置 --> 库版本: _collect_library_versions
    收集配置 --> 项目信息: _collect_project_info
    
    系统信息 --> 组装配置: 返回system字典
    Python环境 --> 组装配置: 返回python字典
    库版本 --> 组装配置: 返回libraries字典
    项目信息 --> 组装配置: 返回project字典
    
    组装配置 --> 保存本地: save_to_file
    保存本地 --> 补全信息: complete_missing_info
    补全信息 --> 写入文件: JSON序列化
    
    写入文件 --> 更新Joplin: update_joplin_note
    更新Joplin --> 合并配置: merge_all_configs
    
    合并配置 --> 加载本地: load_all_configs
    合并配置 --> 加载Joplin: load_configs_from_joplin_note
    加载Joplin --> 解析Markdown: parse_config_from_markdown_table
    
    合并配置 --> 生成表格: generate_markdown_table
    合并配置 --> 生成历史: generate_update_history
    
    生成表格 --> 更新笔记: updatenote_body
    生成历史 --> 更新笔记: updatenote_body
    
    更新笔记 --> 完成: 返回成功状态
    完成 --> [*]
    
    state 错误处理 {
        收集配置 --> 错误: 异常捕获
        保存本地 --> 错误: IO异常
        更新Joplin --> 错误: 网络异常
        错误 --> 记录日志: log.error
        记录日志 --> 返回失败: 返回False
    }

```

---

## 数据结构

### 配置快照 JSON (`{device_id}.json`)

```json
{
  "system": {
    "device_id": "0x1070835fe6e5d115",
    "device_name": "my-server",
    "host_user": "baiyefeng",
    "timestamp": "2026-03-22T16:30:00",
    "system": {
      "platform": "Linux-4.15.0",
      "system": "Linux",
      "release": "4.15.0-249-generic",
      "version": "#249-Ubuntu SMP",
      "machine": "x86_64",
      "processor": "x86_64",
      "architecture": ["64bit", "ELF"],
      "distro": "Ubuntu 20.04 LTS",
      "kernel": "4.15.0-249-generic"
    }
  },
  "python": {
    "python_version": "3.10.0",
    "python_implementation": "CPython",
    "conda_version": "conda 23.1.0",
    "pip_version": "pip 23.0.1",
    "virtual_env": "N/A",
    "conda_env": "base"
  },
  "libraries": {
    "pandas": "2.3.1",
    "numpy": "1.26.4",
    "matplotlib": "3.8.0"
  },
  "project": {
    "project_path": "/data/codebase/hostconfig",
    "config_files": {
      "requirements.txt": {"exists": true, "size": 256, "modified": "..."}
    }
  },
  "collection_time": "2026-03-22T16:30:00"
}
```

### 更新记录 JSON (`{device_id}_updates.json`)

```json
[
  {
    "timestamp": "2026-03-22 16:30",
    "device_id": "0x1070835fe6e5d115",
    "device_name": "my-server",
    "has_changes": true,
    "summary": "配置变化: system, python, libraries"
  }
]
```

---

## 配置文件说明

| 文件 | 格式 | 用途 |
|:---|:---|:---|
| `data/happyjp.ini` | INI | Joplin API 连接配置（URL、token） |
| `data/happyjphard.ini` | INI | 硬件相关的 Joplin 参数 |
| `data/happyjpinifromcloud.ini` | INI | 云端设备映射表（设备名→device_id），支持多平台同步 |
| `data/happyjpsys.ini` | INI | 系统开关，如 `FORCE_UPDATE=true` 强制更新笔记 |
| `rootfile` | 空文件 | 项目根目录定位标记，`func/first.py` 向上查找此文件确定 `getdirmain()` |

---

## 子模块说明

`func/` 是 Git 子模块，地址 `https://github.com/heart5/func`，提供通用工具函数库。本项目中使用到的关键模块：

| 模块 | 功能 |
|:---|:---|
| `func/first.py` | 项目路径初始化，`getdirmain()` / `dirmainpath()` |
| `func/getid.py` | `getdeviceid()` / `getdevicename()` / `gethostuser()` |
| `func/jpfuncs.py` | Joplin API 完整封装（搜索笔记本、笔记、创建/更新笔记内容） |
| `func/configpr.py` | INI 配置文件解析（`findvaluebykeyinsection`等） |
| `func/logme.py` | 统一日志输出 `log.info/warning/error/debug` |
| `func/sysfunc.py` | 系统封装 `execcmd()` / `not_IPython()` |
| `func/wrapfuncs.py` | `@timethis` 装饰器计时 |
| `func/common/utils.py` | 底层 Shell 命令执行 `execute()` |

---

## Jupytext 说明

项目中 `.py` 和 `.ipynb` 文件通过 [jupytext](https://github.com/mwouts/jupytext) 双向同步。编辑 `.py` 文件后，在 Jupyter 中打开 `.ipynb` 会自动同步内容。核心文件：

- `hostconfig.py` ↔ `hostconfig.ipynb`
- `pathmagic.py` ↔ `pathmagic.ipynb`
- `func/*.py` ↔ `func/*.ipynb`

