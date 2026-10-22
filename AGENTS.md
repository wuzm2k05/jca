# AGENTS.md

给在本仓库里工作的 AI agent 的项目约束。

**这个代码库的失败模式几乎全是静默的**——错法不报错、图照跑、工具照写，只是行为悄悄不对。
下面每一条都是「不这么说就一定会做错」的，不是风格偏好。

## 先读什么

- `README.md` 是设计记录，**读全文**（518 行，不是 skim）。`document/lesson{1,2,3}.zh.md` 是配套课程，含「为什么」和已经踩过的坑。
- **源码 docstring 是中文的，是真正的设计档案。** 本项目大量约束只存在于注释里（比如 `engine/agent.py` 里那段解释为什么不能加静态出边的注释）。跳过注释会直接踩坑。
- `req.md` 是**已落地工作**的规格书——HEAD 的 `b5b240a` 就是它。别重做。下一步该做什么，看 `document/lesson3.zh.md` 的「七、扩展练习」。

## 跑测试

```bash
uv run python -m tests.test_engine_events    # agent 循环，对抗脚本化的假模型
uv run python -m tests.test_middleware       # 中间件管道与 TodoMiddleware
uv run python -m tests.test_ui_transcript    # 事件进、渲染文本出
uv run python -m tests.test_tools            # 文件与 shell 工具，打真实临时目录
```

- **新增测试函数必须手动加进该文件 `main()` 里的列表。** 那个手写列表就是 runner，漏了是**静默不跑**。
- **pytest 不是声明依赖**（`pyproject.toml` 的 `dev` 组只有 pyinstaller）。要用就 `uv add --dev pytest` 或 `uv run --with pytest pytest tests/`。
- `tests/test_tools.py` 的临时目录必须建在 **cwd 之内**：`safe_path` 只放行 workspace 下的路径，用系统 temp 目录会测出「这个 agent 永远做不到的事」。

## 会静默失败的红线

**图的结构**

- `engine/agent.py:199-201` 三个 `add_node` 上的 `input_schema=state_schema`，**一个都不能删**。不写的话 LangGraph 会拿节点函数第一个参数的类型标注当输入 schema，于是中间件加的字段在进节点前被悄悄滤掉：图照跑、工具照写、checkpointer 里也存得住，**只有节点里读不到，一句报错都没有**。
- `agent` 节点**没有静态出边**，下一步完全由它返回的 `Command(goto=...)` 决定。**不要加** `add_edge("agent", "after_model")`：`Command.goto` 是**追加分支**而不是改道，被内容判定拦下的重试轮会在**同一个超步**里跑完两个钩子节点，只要任何中间件在两个钩子里写同一个 state 键，就直接 `InvalidUpdateError`。
- **`after_model` 不能往 `messages` 追加任何消息。** 下一跳 `tools_condition` 读的是**最后一条**消息——塞一条 `HumanMessage` 等于告诉它「模型没要求调工具」，图直接结束，模型刚请求的工具永不执行。静默。
- **`before_model` 不能塞 `AIMessage`。** `_translate_update` 会把**任何**节点产出的 `AIMessage` 当成「模型申请调工具」上报给界面。
- **钩子顺序有语义**：`before_model` 按声明序、`after_model` **逆序**（包起来的东西按逆序拆）。越靠后声明的中间件越靠近提示词末尾，也就优先级最高。**写反了照样跑**，只是某个中间件在 after 里会读到旧值。
- 中间件的 `state_schema` 与 `tools` 是**类属性**，不是构造参数——建图时就要读到，那时还没有实例。

**提示词与状态**

- **system prompt 只活在 `ModelRequest` 里**，每次调用重建、用完即弃，**绝不入 state**。入 state 会每轮重发、烧 token，而且「谁在什么时候改的」永远查不清。
- `wrap_model` 里必须用 `dataclasses.replace` 造**新** request，不要就地改——就地改会污染同一次调用里排在**前面**的中间件。
- `todos` 字段带 reducer，所以**初始值是 `[]` 而不是不存在**。`state.get("todos") is None` 这种写法是错的。
- 声明同一个 state 键两次是**启动期 `ValueError`**，不是静默覆盖。合并读的是 `get_type_hints(..., include_extras=True)`——reducer 在 `Annotated` 里面，默认会被剥掉。

**工具**

- **中文 docstring、中文返回值，出错返回说明而不是抛异常。** 工具**名**也会进请求体，所以是 `run_cmd` 而不是 `execute_bash`。这不是文风问题：docstring 是写给模型看的产品说明书，而且服务端有内容过滤（见 `document/issues.md`）。
- **中间件自己的工具跟着中间件放**（如 `engine/middleware/todo.py` 的 `write_todos`），不要塞进 `tools/`。加**常驻**工具则要同时改 `FILE_TOOLS`（`tools/files.py:556`）和 `build_tools()`，由 `test_new_tools_are_registered` 守着。
- 需要写状态的工具要返回 `Command` 并配一条 `ToolMessage`（`ToolNode` 会拒收没有回执的 command）。

## 注意

- **README 里写的 `JCA_ALLOW_SHELL=1` 开关在代码中不存在**：`tools/__init__.py` 无条件挂载 `run_cmd`。这是 `lesson2.zh.md` 记录在案的**已知文档/代码不一致**。要动它先问，别默认「把开关补回去」。
- **默认 workspace 就是你启动程序的目录**，所以在本仓库根目录跑它自己的 app，等于让模型对本项目有读写权限（`main.py` 也不例外）。用 `JCA_WORKSPACE=workspace`。另外日志默认落在同一个 workspace 的 `logs/` 下——两处默认值是撞的。
- **`safety/guard.py` 的 `RETRY_NOTE` 和 `prompts/todos.py` 的措辞是实测出来的常量**，不同的近似说法过审率天差地别。改之前重新实测，不要凭语感改。
- **`.env` 和 `start_claude_autodl.bat` 内含凭据**，两者都在 `.gitignore` 里（所以不会出现在 `git status` 中，容易被忽略）。不要提交，也不要复制进文档、示例或任何产物。
- 打包是真实目标（`jca.spec`、`dist/jca.exe`）：用 `Path.cwd()` 而不是 `Path(__file__).parent`。
- 生产代码里**没有 `print`**（`ui/` 里的 `console.print` 是 rich 的输出路径，不算）。用 `logger = get_logger(__name__)` 加结构化 kwargs。
