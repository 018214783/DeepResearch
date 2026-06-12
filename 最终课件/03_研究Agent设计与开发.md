# DeepAgents框架和本项目的Agent设计与开发

## 1.DeepAgents概述

### 1.1 和langchain、langgraph的区别

 Deep Agents 构建在 LangChain 的 Agent 基础组件之上，并使用 LangGraph runtime 提供持久执行、流式输出、人类介入等能力。

相比直接使用 `create_agent`，DeepAgents 默认集成了任务规划、虚拟文件系统、上下文压缩、子智能体、长期记忆、权限和人类审批等能力，更适合复杂、多步骤、长上下文任务。

三者关系可以这样理解：

| 层级          | 组件                       | 主要解决的问题                              | 适合场景                   |
| ----------- | ------------------------ | ------------------------------------ | ---------------------- |
| 应用级 harness | DeepAgents               | 开箱即用地构建复杂 Agent，内置规划、文件系统、子智能体和上下文管理 | 研究、编码、长任务、多步骤自动化       |
| Agent 框架    | LangChain `create_agent` | 构建标准模型-工具循环，并通过 middleware 扩展        | 轻量工具调用 Agent、需要自定义少量能力 |
| 编排运行时       | LangGraph                | 自定义有状态工作流、节点、边、持久化和中断恢复              | 复杂状态机、强控制流、多 Agent 图编排 |

对本项目来说，选择 DeepAgents 的原因不是“不能用 LangChain 或 LangGraph”，而是研究报告生成天然包含长任务规划、检索材料沉淀、子任务委托和上下文压缩。

### 1.2 核心原理

DeepAgents 的核心原理可以概括为：在标准模型-工具循环外面，增加一组面向长任务的默认能力，让 Agent 不只是“会调用工具”，还能够规划任务、隔离上下文、管理中间文件、委托子任务并在必要时持久化状态。

DeepAgents 在langchain的`create_agent`外面增加了几类关键能力：

| 能力    | 官方文档中的对应机制                                                             | 作用                                       |
| ----- | ---------------------------------------------------------------------- | ---------------------------------------- |
| 任务规划  | `write_todos`                                                          | 把复杂任务拆成可跟踪步骤，并随着新信息更新计划                  |
| 文件系统  | `ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep` 等文件工具          | 把大输入、中间材料和大工具结果放到虚拟文件系统中                 |
| 上下文管理 | context compression、offloading、summarization                           | 降低长任务中上下文膨胀的风险                           |
| 子智能体  | `task` 工具和 `subagents` 配置                                              | 把检索、代码审查、数据分析等子任务委托给隔离上下文的 Agent         |
| 后端存储  | `StateBackend`、`FilesystemBackend`、`StoreBackend`、`CompositeBackend` 等 | 控制文件写到线程状态、本地文件系统、LangGraph store 或自定义后端 |
| 人类介入  | human-in-the-loop / interrupt                                          | 对敏感工具调用进行审批、修改或拒绝                        |

在本项目中，最重要的是前三类能力：todo、虚拟文件系统和子智能体。

```mermaid
flowchart TD
    Main[主研究智能体]
    Todo[Todo 规划]
    FS[(虚拟文件系统)]
    Task[task 工具]
    SearchSub[信息检索子智能体]
    Tools[搜索 / 网页读取 / RAGFlow]
    Save[save_research_section]

    Main --> Todo
    Main --> FS
    Main --> Task
    Task --> SearchSub
    SearchSub --> Tools
    Tools --> SearchSub
    SearchSub --> Main
    Main --> Save
    Main --> FS
```

DeepAgents 的“文件系统”不等于默认直接操作服务器真实目录。官方 Backends 文档说明，Deep Agents 通过可插拔 backend 暴露文件系统表面。默认 backend 是线程级的 `StateBackend`，文件存在 LangGraph agent state 中；如果使用 `FilesystemBackend` 才会读写本地磁盘，并且官方明确提醒本地文件系统访问有安全风险，生产环境要使用沙箱、权限规则或更受控的 backend。

因此，本项目中把 `/research/task_payload.json` 和 `/research/workspace/` 设计成 Agent 工作区，是为了让 Agent 在虚拟文件系统中管理大输入和中间材料，而不是把所有内容塞进单条用户消息。

### 1.3 核心价值

DeepAgents 对本项目的价值主要体现在三个方面。

第一，多智能体开发更自然。官方 Subagents 文档说明，Deep Agents 可以通过 `subagents` 参数配置自定义子智能体，主智能体通过内置 `task` 工具委托任务。

子智能体适合处理会污染主上下文的多步骤任务、需要专门工具的任务，或者需要不同模型能力的任务。本项目正好把“研究管理”和“信息检索”拆开：主智能体负责研究策略和章节写作，检索子智能体负责搜索、网页读取、RAGFlow 检索和事实整理。

第二，规划能力更适合长任务。研究报告不是一次问答，而是“理解任务 -> 设计大纲 -> 拆解章节 -> 检索证据 -> 写正文 -> 保存结果”的链路。

DeepAgents 内置 `write_todos` 规划工具，可以让 Agent 在执行前维护任务清单，并随着缺失章节、检索结果或用户补充要求调整计划。

第三，上下文管理更稳定。官方 Context Engineering 文档把 context compression、文件 offloading、summarization 和 subagent isolation 都列为长任务上下文管理机制。

对研究场景来说，搜索结果、网页正文、事实卡片、引用来源和章节草稿都可能很长。如果全部放在消息历史里，模型容易遗漏章节、混淆来源，甚至把搜索摘要当事实。使用虚拟文件系统和子智能体隔离后，主智能体可以只接收整理后的结论和证据结构。

可以把传统单 Agent 和 DeepAgents 多 Agent 的差别画成下面这样：

```mermaid
flowchart LR
    subgraph Single[单 Agent 写法]
        U1[用户任务] --> A1[一个 Agent]
        A1 --> T1[所有工具]
        T1 --> A1
        A1 --> R1[一次性长输出]
    end

    subgraph Deep[DeepAgents 写法]
        U2[用户任务] --> M[主智能体]
        M --> P[Todo 计划]
        M --> S[检索子智能体]
        S --> T2[专属检索工具]
        T2 --> S
        S --> E[事实和来源]
        E --> M
        M --> DB[逐章节落库]
    end
```

本项目采用 DeepAgents，不是为了追求“Agent 数量越多越好”，而是为了把研究链路中的不同风险隔离开：

| 风险            | DeepAgents 能力             | 本项目落点                               |
| ------------- | ------------------------- | ----------------------------------- |
| 检索材料太长，挤占主上下文 | 子智能体上下文隔离、文件系统 offloading | 检索子智能体只返回来源、事实和冲突                   |
| 长报告一次性输出容易漏章节 | todo 规划、逐章节工具调用           | 主智能体调用 `save_research_section` 保存每章 |
| 工具结果和正文混在一起   | 虚拟文件系统和结构化输出              | 中间材料进 `/research/workspace/`，最终结果落库 |
| 报告渲染阶段二次编造事实  | 职责边界                      | 渲染阶段只做确定性 HTML 转换                   |

这些价值都依赖工程约束配合。DeepAgents 提供的是能力，不会自动保证来源可靠、章节完整或工具安全。因此后续章节会继续通过 Prompt、Pydantic 模型、工具校验和确定性渲染来收紧边界。

### 1.4 编码示例

下面的示例只用于说明 DeepAgents 的基础用法。正式项目代码会在后续章节中结合 `ResearchAgent` 、Prompt 文件、检索工具和保存工具展开。

#### 1.4.1 构建子智能体示例

官方 Subagents 文档说明，自定义子智能体通常通过字典配置，核心字段包括 `name`、`description`、`system_prompt`、`tools`，可选字段包括 `model`、`middleware`、`response_format`、`permissions` 等。主智能体会根据 `description` 判断什么时候通过 `task` 工具委托给子智能体。

一个最小化的研究检索子智能体可以这样写：

```python
from deepagents import create_deep_agent


def external_search(query: str, max_results: int = 5) -> dict:
    """Search public web sources for a research question."""
    return {
        "query": query,
        "results": [],
    }


search_subagent = {
    "name": "search-agent",
    "description": "负责公开互联网检索、网页读取、内部知识库检索和证据整理。",
    "system_prompt": (
        "你是信息检索智能体。只负责检索、读取、整理来源和抽取事实，"
        "不要编写最终报告正文。输出必须包含 sources、fact_cards 和 conflicts。"
    ),
    "tools": [external_search],
}

manager_agent = create_deep_agent(
    model="openai:gpt-5.4",
    tools=[],
    system_prompt="你是研究管理智能体，负责规划研究任务并委托检索子智能体。",
    subagents=[search_subagent],
    name="research-manager-agent",
)
```

在真实项目中，`external_search` 会替换为 Tavily 搜索工具，子智能体还会拿到 `read_web_page` 和 `ragflow_search`。主智能体不会直接调用这些检索工具，而是通过子智能体完成证据收集，这样主上下文里保留的是整理后的事实结果，而不是所有网页正文和搜索过程。

#### 1.4.2 使用文件系统示例

DeepAgents 的文件系统能力用于管理长任务中的上下文。官方 Backends 文档说明，默认 backend 是线程级状态中的虚拟文件系统；文件会随同一个 thread 的 checkpoint 在多轮中保留，但不会跨 thread 共享。也可以显式使用 `FilesystemBackend`、`StoreBackend` 或 `CompositeBackend`。

本项目更推荐把大输入作为初始文件传给 Agent，而不是直接放进消息正文：

```python
import json

from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data


agent = create_deep_agent(
    model="deepseek:deepseek-chat",
    system_prompt="你是研究管理智能体。先读取任务文件，再规划研究步骤。",
)

task_payload = {
    "task_name": "generate_report",
    "project_id": "project-001",
    "topic": "企业级 Deep Research 系统设计",
    "required_section_ids": ["1", "2", "3"],
}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "请读取 /research/task_payload.json，先使用 todo 规划步骤，"
                    "中间材料写入 /research/workspace/，最终只返回严格 JSON。"
                ),
            }
        ],
        "files": {
            "/research/task_payload.json": create_file_data(
                json.dumps(task_payload, ensure_ascii=False, indent=2)
            ),
            "/research/workspace/README.md": create_file_data(
                "该目录用于保存检索摘要、来源整理、事实卡片和章节草稿。"
            ),
        },
    }
)
```

如果需要接入真实磁盘，需要显式配置 backend，并谨慎处理权限：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend


agent = create_deep_agent(
    model="deepseek:deepseek-chat",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir="/absolute/path/to/project-workspace",
                virtual_mode=True,
            ),
        },
    ),
)
```

这里的设计含义是：Agent 内部临时材料仍走 `StateBackend`，只有 `/workspace/` 路径映射到指定本地目录。`virtual_mode=True` 用于把访问限制在 `root_dir` 之下。官方文档也提醒，本地文件系统访问可能读取敏感文件或造成不可逆修改，生产环境中应优先使用更受控的状态后端、存储后端、沙箱和权限规则。

## 2. Agent总设计

当前项目中，仅在大纲制定环节和研究执行环节使用智能体。报告生成阶段不再额外构建“报告写作 Agent”，而是使用确定性的报告渲染工具，把已经落库的结构化研究结果转换成 HTML。

整个 Agent 链路可以拆成两层：

| 层级   | 名称      | 作用                                        |
| ---- | ------- | ----------------------------------------- |
| 主智能体 | 研究管理智能体 | 理解研究任务、生成大纲、修改大纲、拆解章节、协调检索、整理事实和洞察、写出章节正文 |
| 子智能体 | 信息检索智能体 | 围绕主智能体分派的问题进行公开搜索、网页读取、内部知识库检索和事实整理       |

报告阶段的职责边界如下：

| 阶段         | 是否使用 LLM Agent      | 产物                                                |
| ---------- | ------------------- | ------------------------------------------------- |
| 生成研究任务书和大纲 | 使用研究管理智能体           | `research_brief`、`outline`                        |
| 修改大纲       | 使用研究管理智能体           | 修订后的 `outline`                                    |
| 执行研究       | 使用研究管理智能体 + 信息检索智能体 | `sections`、`sources`、`fact_cards`、`insight_cards` |
| 渲染报告       | 不使用报告 Agent         | HTML、目录、引用、参考来源                                   |

这个设计的核心原因是：研究过程需要 LLM 进行理解、拆解、检索规划、归纳和写作；但报告渲染阶段主要是结构转换和页面展示，不应该让 LLM 重新补写事实、改写证据链或生成新的来源。

整体流程如下：

```mermaid
flowchart TD
    后台任务[后台任务]
    Agent门面[ResearchAgent 门面]
    研究管理[研究管理智能体]
    信息检索[信息检索智能体]
    外部搜索[external_search]
    网页读取[read_web_page]
    知识库检索[ragflow_search]
    章节保存[save_research_section]
    项目仓储[(research_projects)]
    报告渲染[write_html_report]
    报告仓储[(report_versions)]

    后台任务 --> Agent门面
    Agent门面 --> 研究管理
    研究管理 --> 信息检索
    信息检索 --> 外部搜索
    信息检索 --> 网页读取
    信息检索 --> 知识库检索
    研究管理 --> 章节保存
    章节保存 --> 项目仓储
    Agent门面 --> 报告渲染
    报告渲染 --> 报告仓储
```

## 3. 研究Agent设计

在前面的架构设计环节，整个研究过程已经拆成两个 Agent：一个是主研究管理者，另一个专门负责收集信息、整理来源并构建事实证据链。

这里要注意一个边界：信息检索智能体不是“帮忙写报告”的智能体，它只负责证据材料；主研究智能体才负责把证据组织成章节正文和研究结果。

### 3.1 主研究智能体的职责

主研究智能体负责：

- 基于用户的问题，构建研究任务书和研究大纲。

- 基于用户对大纲的修改意见，修改大纲。

- 基于已确认大纲，拆解每个章节需要回答的问题。

- 将检索问题分派给信息检索智能体。

- 整理来源、事实卡片、冲突信息和洞察卡片。

- 写出每个章节的完整正文。

- 为关键判断构建证据链。

- 调用 `save_research_section` 工具，将章节研究结果保存至数据库。

主研究智能体不负责：

- 不直接调用互联网搜索工具。

- 不直接读取网页正文。

- 不直接调用 RAGFlow。

- 不直接编写最终 HTML。

- 不调用报告写作 Agent。

- 不保存项目状态和任务状态。

主研究智能体的核心输入输出如下：

| 任务类型                      | 输入                 | 输出         |
| ------------------------- | ------------------ | ---------- |
| `generate_research_brief` | 项目主题、目标、读者、地域和时间范围 | 研究任务书、大纲草案 |
| `revise_outline`          | 当前大纲、用户修改意见        | 修订后的大纲     |
| `generate_report`         | 已确认大纲、项目设定、用户补充要求  | 逐章节保存的研究结果 |

项目中使用 `ResearchAgent` 类作为业务门面，后台任务不直接操作 DeepAgents 对象：

```python
class ResearchAgent:
    """研究智能体业务门面。"""

    def __init__(self, manager_agent: Any | None = None, report_agent: Any | None = None) -> None:
        self.manager_agent = manager_agent
        self.report_agent = report_agent

    async def generate_research_brief(self, project: dict[str, Any] | None) -> ResearchBriefResult:
        payload = self._build_generate_research_brief_input(project=project)
        raw_result = await self._invoke_manager_agent(
            task_name="generate_research_brief",
            payload=payload,
        )
        return self._parse_research_brief_result(raw_result=raw_result, project=project)

    async def revise_outline(
        self,
        project: dict[str, Any] | None,
        outline: list[OutlineNode],
        revision_instruction: str,
    ) -> list[OutlineNode]:
        payload = self._build_revise_outline_input(
            project=project,
            outline=outline,
            revision_instruction=revision_instruction,
        )
        raw_result = await self._invoke_manager_agent(
            task_name="revise_outline",
            payload=payload,
        )
        return self._parse_outline_result(raw_result=raw_result, fallback_outline=outline)
```

研究执行阶段不是一次性要求模型返回一个巨大的 `research_result`，而是要求主智能体逐章节调用工具落库：

```python
async def generate_research_result(
    self,
    project: dict[str, Any] | None,
    outline: list[OutlineNode],
    user_instruction: str | None,
) -> ResearchResult:
    project_id = self._get_project_id(project=project)
    await research_project_repository.clear_research_sections(project_id=project_id)
    expected_section_ids = self._expected_research_section_ids(outline=outline)
    sections: list[dict[str, Any]] = []
    missing_section_ids = sorted(expected_section_ids)

    for attempt in range(1, 5):
        payload = self._build_generate_research_result_input(
            project=project,
            outline=outline,
            user_instruction=user_instruction,
            required_section_ids=sorted(expected_section_ids),
            missing_section_ids=missing_section_ids,
            attempt=attempt,
        )
        await self._invoke_manager_agent(task_name="generate_report", payload=payload)
        sections = await research_project_repository.get_research_sections(project_id=project_id)
        saved_section_ids = {
            str(section.get("section_id"))
            for section in sections
            if isinstance(section, dict) and section.get("section_id")
        }
        missing_section_ids = sorted(expected_section_ids - saved_section_ids)
        if not missing_section_ids:
            break

    return self._build_research_result_from_saved_sections(
        sections=sections,
        sources=await research_project_repository.get_research_sources(project_id=project_id),
        project=project,
        outline=outline,
    )
```

这里有一个重要设计：如果部分章节没有被保存，系统会把缺失的 `section_id` 重新传给主智能体，要求它补写缺失章节。这样可以避免一次大输出中遗漏章节。

主研究智能体的 Prompt 核心片段如下：

```markdown
你是 AI 研究报告工作台中的研究管理智能体。

你的职责是完成研究本身：理解任务、设计大纲、协调信息检索、整理事实、形成洞察、写出完整章节正文，并产出可落库的结构化研究结果。

你不是报告渲染智能体。你不负责把研究结果渲染为最终 HTML。报告渲染由后端确定性渲染流程基于你落库的 `research_result` 完成。
```

针对 `generate_report` 任务，Prompt 明确要求主智能体逐章节保存结果：

```markdown
目标：根据已确认大纲完成逐章节研究，并通过 `save_research_section` 工具把每个有正文的章节写入数据库。不要一次性输出完整 `research_result`。

流程要求：

1. 基于已确认大纲识别需要写正文的章节。
2. 如果任务载荷中存在 `missing_section_ids`，本轮只处理这些章节，不要重写已保存章节。
3. 对每个章节拆解检索问题，委托信息检索智能体获取公开来源和可复核事实。
4. 写出该章节完整正文、关键发现、证据链、表格/图表结构、风险说明和本章来源详情。
5. 调用 `save_research_section(project_id, section)` 保存该章节。
```

### 3.2 信息检索智能体的职责

信息检索智能体负责：

- 基于主研究智能体分派的问题，构造搜索关键词。

- 使用公开互联网搜索工具发现资料来源。

- 使用网页读取工具读取关键网页正文和元数据。

- 按需使用 RAGFlow 工具检索内部知识库。

- 对来源进行去重和相关性判断。

- 从来源中提取可复核事实。

- 标注每条事实对应的来源。

- 识别不同来源之间的冲突、口径差异和不确定性。

信息检索智能体不负责：

- 不设计完整研究大纲。

- 不生成最终 HTML 报告。

- 不编造 URL、日期、机构名称或数据。

- 不把搜索摘要直接当作最终事实。

- 不保存数据库状态。

信息检索智能体可以使用的工具如下：

| 工具                | 文件                             | 作用                  |
| ----------------- | ------------------------------ | ------------------- |
| `external_search` | `app/tools/external_search.py` | 调用 Tavily 搜索公开互联网资料 |
| `read_web_page`   | `app/tools/web_reader.py`      | 读取网页正文、标题、发布时间线索    |
| `ragflow_search`  | `app/tools/ragflow_search.py`  | 检索 RAGFlow 内部知识库    |

公开互联网搜索工具的核心结构如下：

```python
async def external_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    time_range: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    normalized_query = query.strip()
    if not normalized_query:
        return {
            "status": "error",
            "provider": "tavily",
            "query": query,
            "results": [],
            "error": "query 不能为空",
        }

    settings = get_settings()
    if not settings.tavily_api_key:
        return {
            "status": "skipped",
            "provider": "tavily",
            "query": normalized_query,
            "results": [],
            "error": "TAVILY_API_KEY 未配置",
        }
```

网页读取工具的核心结构如下：

```python
async def read_web_page(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> dict[str, Any]:
    normalized_url = url.strip()
    if not normalized_url.startswith(("http://", "https://")):
        return {
            "status": "error",
            "url": url,
            "title": None,
            "published_at": None,
            "content": "",
            "error": "仅支持 http 或 https URL",
        }

    html, final_url, content_type = await asyncio.to_thread(_fetch_html, normalized_url)
```

RAGFlow 检索工具的核心结构如下：

```python
async def ragflow_search(
    query: str,
    dataset_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
    page: int = 1,
    page_size: int = 10,
    similarity_threshold: float = 0.2,
    vector_similarity_weight: float = 0.3,
    top_k: int = 1024,
    keyword: bool = False,
) -> dict[str, Any]:
    normalized_query = query.strip()
    if not normalized_query:
        return {
            "status": "error",
            "provider": "ragflow",
            "query": query,
            "chunks": [],
            "error": "query 不能为空",
        }
```

信息检索智能体的 Prompt 核心片段如下：

```markdown
你是 AI 研究报告工作台中的信息检索智能体。

你的职责是围绕研究管理智能体分派的问题进行资料检索、网页读取、内部知识库检索、事实整理和证据链输出。

搜索工具用于发现来源，不把搜索摘要直接当作最终事实。
网页读取工具用于获取可追溯正文和来源元数据。
RAGFlow 工具用于检索内部知识库，不把内部资料伪装成公开来源。
```

信息检索智能体的最终输出必须是严格 JSON：

```json
{
  "sources": [
    {
      "source_id": "source-1",
      "title": "来源标题",
      "url": "https://example.com",
      "published_at": "2026-01-01",
      "source_type": "public_web",
      "summary": "来源摘要"
    }
  ],
  "fact_cards": [
    {
      "fact_id": "fact-1",
      "statement": "可复核事实",
      "source_ids": ["source-1"],
      "confidence": "medium",
      "evidence_summary": "证据摘要"
    }
  ],
  "conflicts": []
}
```

### 3.3 其他可扩展的智能体

在实际生产环境下，本项目还可以继续扩展更多智能体，但第一版不需要一次性拆太多 Agent。拆分智能体的原则是：只有当某类任务有独立工具、独立上下文和独立输出结构时，才适合拆成单独智能体。

可扩展方向包括：

- 问数智能体：面向企业内部结构化数据库，负责 SQL 生成、指标查询和数据解释。

- 竞品分析智能体：专门跟踪公司、产品、融资、价格、渠道和客户案例。

- 政策分析智能体：专门检索政策文件、监管动态、官方解读和政策影响。

- 财务分析智能体：专门处理财报、公告、经营数据和估值指标。

- 图表规划智能体：根据研究结果规划适合展示的图表类型和数据结构。

这些智能体都不应该改变当前系统的主边界：研究管理智能体负责协调研究过程，报告最终由确定性渲染流程生成。

## 4. 开发

### 4.1 多智能体架构的难点

多智能体架构的难点不在于“创建多个 Agent”，而在于职责边界和数据边界是否清晰。

本项目需要处理几个问题：

1. 谁负责拆解研究问题？

2. 谁负责检索资料？

3. 谁负责判断来源是否可用？

4. 谁负责把事实写成章节正文？

5. 谁负责保存研究结果？

6. 谁负责生成最终 HTML？

如果边界不清晰，很容易出现以下问题：

| 问题                | 结果                   |
| ----------------- | -------------------- |
| 主智能体也搜索，检索智能体也写报告 | 职责混乱，输出不可控           |
| 报告阶段继续让 LLM 补内容   | 来源和事实链条断裂            |
| 所有结果一次性塞进上下文      | 内容过长，模型容易遗漏章节        |
| 只返回自然语言结果         | 后端无法稳定落库和渲染          |
| 工具返回异常没有结构化       | Agent 不知道应该重试、跳过还是降级 |

因此，本项目采用以下约束：

- 主研究智能体只协调研究过程，不直接执行搜索和网页读取。

- 信息检索智能体只处理来源、事实和冲突，不写最终报告。

- 主研究智能体必须通过 `save_research_section` 工具逐章节落库。

- 报告渲染工具只做 HTML 展示转换，不新增事实、来源和结论。

- 所有关键产物都使用 Pydantic 结构描述。

核心结构包括：

```python
class FactCard(BaseModel):
    fact_id: str
    statement: str
    source_ids: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class InsightCard(BaseModel):
    insight_id: str
    title: str
    summary: str
    supporting_fact_ids: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    claim: str
    fact_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    confidence: str = "medium"
```

章节研究结果结构如下：

```python
class ResearchSection(BaseModel):
    section_id: str
    title: str
    summary: str | None = None
    body: str
    key_findings: list[str] = Field(default_factory=list)
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
```

完整研究结果结构如下：

```python
class ResearchResult(BaseModel):
    title: str
    executive_summary: str | None = None
    sections: list[ResearchSection] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    fact_cards: list[FactCard] = Field(default_factory=list)
    insight_cards: list[InsightCard] = Field(default_factory=list)
    synthesis: ResearchSynthesis | None = None
```

这些结构就是研究阶段和报告渲染阶段之间的边界对象。

### 4.2 DeepAgents框架的介绍

DeepAgents 用于构建能够规划任务、调用工具、委托子智能体并维护上下文文件系统的 Agent。

在本项目中，DeepAgents 主要提供四类能力：

| 能力   | 在本项目中的作用                                                                      |
| ---- | ----------------------------------------------------------------------------- |
| 主智能体 | 构建研究管理智能体                                                                     |
| 子智能体 | 构建信息检索智能体                                                                     |
| 工具调用 | 调用 `save_research_section`、`external_search`、`read_web_page`、`ragflow_search` |
| 文件系统 | 把大规模检索结果、来源列表和中间材料写入 `/research/workspace/`，避免上下文膨胀                           |

主智能体构建代码如下：

```python
def _build_deepagents_manager_agent() -> Any | None:
    settings: Settings = get_settings()
    model_name = _build_model_name(settings=settings)
    subagents = [_build_search_subagent(model_name=model_name)]
    return create_deep_agent(
        model=model_name,
        tools=[save_research_section],
        system_prompt=_load_prompt(RESEARCH_MANAGER_PROMPT_PATH),
        subagents=subagents,
        name="research-manager-agent",
        checkpointer=MemorySaver(),
    )
```

信息检索子智能体构建代码如下：

```python
def _build_search_subagent(model_name: str) -> dict[str, Any]:
    settings: Settings = get_settings()
    if settings.enable_ragflow:
        tools = [external_search, read_web_page, ragflow_search]
    else:
        tools = [external_search, read_web_page]
    return {
        "name": "search-agent",
        "description": "负责公开互联网检索、网页读取、RAGFlow 内部知识库检索和证据整理。",
        "system_prompt": _load_prompt(SEARCH_AGENT_PROMPT_PATH),
        "tools": tools,
        "model": model_name,
    }
```

Prompt 不写死在代码中，而是维护在外部 Markdown 文件：

```python
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
RESEARCH_MANAGER_PROMPT_PATH = PROMPT_DIR / "research_manager.md"
SEARCH_AGENT_PROMPT_PATH = PROMPT_DIR / "search_agent.md"

def _load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()
```

模型名称通过配置生成：

```python
def _build_model_name(settings: Settings) -> str:
    provider = settings.llm_provider.lower()
    if provider == "deepseek":
        return f"deepseek:{settings.llm_model_name}"
    if provider == "openai":
        return f"openai:{settings.llm_model_name}"
    return f"{provider}:{settings.llm_model_name}"
```

调用 DeepAgents 时，大 payload 不直接塞进消息正文，而是写入虚拟文件系统：

```python
def _build_deepagents_input(self, payload: dict[str, Any]) -> dict[str, Any]:
    task_json = json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_default)
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "请执行 /research/task_payload.json 中的研究任务。"
                    "先使用 todo 规划步骤；大规模检索结果和报告中间稿请写入"
                    " /research/workspace/ 下的文件；最终只返回严格 JSON。"
                ),
            }
        ],
        "files": {
            "/research/task_payload.json": create_file_data(task_json),
            "/research/workspace/README.md": create_file_data(
                "该目录用于保存检索摘要、来源整理、事实卡片、洞察卡片和报告草稿。"
            ),
        },
    }
```

这里的设计重点是：消息里只告诉智能体任务文件路径，大量输入数据放到 `/research/task_payload.json`，中间材料放到 `/research/workspace/`。

### 4.3 编码

Agent 编码可以按下面的顺序实现：

1. 定义结构化输出模型。

2. 编写工具函数。

3. 编写 Prompt 文件。

4. 构建 DeepAgents 主智能体和子智能体。

5. 编写 `ResearchAgent` 业务门面。

6. 在后台任务中调用 `ResearchAgent`。

#### 1. 定义结构化输出模型

研究任务书结构：

```python
class ResearchBrief(BaseModel):
    topic: str
    research_goal: str
    target_audience: str
    scope_summary: str
    key_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
```

研究任务书和大纲生成结果：

```python
class ResearchBriefResult(BaseModel):
    research_brief: ResearchBrief
    outline: list[OutlineNode]
```

报告渲染结果：

```python
class ReportGenerationResult(BaseModel):
    title: str
    html: str
    sources: list[ReportSource] = Field(default_factory=list)
    fact_cards: list[FactCard] = Field(default_factory=list)
    insight_cards: list[InsightCard] = Field(default_factory=list)
```

#### 2. 编写章节保存工具

主研究智能体生成章节正文后，需要调用 `save_research_section` 保存章节。这个工具会校验章节是否完整，避免占位内容进入报告。

```python
async def save_research_section(project_id: str, section: dict[str, Any]) -> dict[str, Any]:
    errors = await _validate_section(project_id=project_id, section=section)
    if errors:
        return {"ok": False, "errors": errors}

    normalized_sources = await _normalize_section_sources(
        project_id=project_id,
        section=section,
    )
    normalized_section = _normalize_section(section, sources=normalized_sources)
    await research_project_repository.upsert_research_section(
        project_id=project_id,
        section=normalized_section,
    )
    if normalized_sources:
        await research_project_repository.upsert_research_sources(
            project_id=project_id,
            sources=normalized_sources,
        )
    return {
        "ok": True,
        "project_id": project_id,
        "section_id": normalized_section["section_id"],
        "sources_saved": len(normalized_sources),
        "message": "research section saved",
    }
```

工具校验规则包括：

- `section_id` 必须存在，并且属于已确认大纲。

- `body` 必须是完整正文，不能是占位文案。

- `key_findings` 至少有一条非空关键发现。

- `evidence_chain` 至少有一条证据链。

- `evidence_chain.source_ids` 引用的来源必须能在 `section.sources` 或项目已有来源中找到。

#### 3. 构建 Agent 单例

后台任务通过 `get_research_agent()` 获取当前进程内复用的 Agent 门面：

```python
_research_agent: ResearchAgent | None = None

def build_research_agent() -> ResearchAgent:
    manager_agent = _build_deepagents_manager_agent()
    return ResearchAgent(manager_agent=manager_agent, report_agent=None)

def get_research_agent() -> ResearchAgent:
    global _research_agent
    if _research_agent is None:
        _research_agent = build_research_agent()
        logger.info("研究智能体门面已初始化")
    return _research_agent
```

#### 4. 后台任务调用 Agent

生成大纲任务中调用 `generate_research_brief`：

```python
project = await research_project_repository.get_project(project_id=project_id)
research_agent = get_research_agent()
result = await research_agent.generate_research_brief(project=project)

await research_project_repository.save_research_brief_and_outline(
    project_id=project_id,
    research_brief=result.research_brief,
    outline=result.outline,
)
```

生成报告任务中先执行研究，再渲染报告：

```python
project = await research_project_repository.get_project(project_id=project_id)
outline = await research_project_repository.get_confirmed_outline(project_id=project_id)
research_agent = get_research_agent()

research_result = await research_agent.generate_research_result(
    project=project,
    outline=outline,
    user_instruction=user_instruction,
)
await research_project_repository.save_research_result(
    project_id=project_id,
    research_result=research_result,
)

project_with_research_result = await research_project_repository.get_project(project_id=project_id)
result = await research_agent.generate_report(
    project=project_with_research_result,
    outline=outline,
    user_instruction=user_instruction,
)
await report_repository.save_report_version(
    project_id=project_id,
    title=result.title,
    html=result.html,
    sources=result.sources,
)
```

#### 5. 确定性报告渲染

`generate_report` 方法内部不调用报告写作 Agent，而是调用 `write_html_report`：

```python
async def generate_report(
    self,
    project: dict[str, Any] | None,
    outline: list[OutlineNode],
    user_instruction: str | None,
) -> ReportGenerationResult:
    payload = self._build_generate_report_input(
        project=project,
        outline=outline,
        user_instruction=user_instruction,
    )
    raw_result = await write_html_report(
        research_result=payload["research_result"],
        layout_plan=self._build_default_layout_plan(payload=payload),
    )
    return self._parse_report_generation_result(raw_result=raw_result, project=project)
```

报告渲染工具的边界非常明确：

```python
async def write_html_report(
    research_result: dict[str, Any] | None = None,
    layout_plan: dict[str, Any] | None = None,
    **legacy_kwargs: Any,
) -> dict[str, Any]:
    document_ir = await build_report_document(
        research_result=research_result,
        layout_plan=layout_plan,
    )
    return await render_report_html(document_ir=document_ir)
```

它只做三件事：

1. 把 `research_result` 转成展示用 document IR。

2. 渲染 HTML、目录、章节、引用和来源列表。

3. 返回 `title`、`html` 和 `sources`。

它不做以下事情：

- 不新增事实。

- 不新增来源。

- 不新增结论。

- 不调用搜索工具。

- 不改写证据链。

最终，Agent 开发完成后，整个链路是：

```text
background
  -> get_research_agent()
  -> generate_research_brief / revise_outline / generate_research_result
  -> DeepAgents 研究管理智能体
  -> DeepAgents 信息检索子智能体
  -> tools 检索和读取资料
  -> save_research_section 逐章节落库
  -> write_html_report 确定性渲染
  -> report_repository 保存报告版本
```
