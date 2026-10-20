# jca-simple

A runnable **ReAct coding agent** built with **LangGraph + LangChain**.

This repository is the teaching material for a short course on how agents work. Lessons
are delivered as git tags, so the same repository shows you the project at every stage of
the course.

---

## 1. A teaching project

The goal is to build an agent from small to big. Each lesson adds one idea, and you can check out
that lesson's tag to see exactly what the code looked like when it was taught.

**What you should understand by the end:**

1. **An agent is a loop, not a pipeline.** Think → act → look at the result → think again.
2. **Deciding and doing are separate.** The model only ever *requests* an action; your own
   code performs it. This is what keeps an agent under your control.
3. **Tools are the agent's limits.** The model can only do what you have given it a tool
   for.
4. **The loop ends when the model stops asking for tools** — not because you wrote an
   "is the task finished?" check.

### How the lessons are organized

Every lesson is a git tag beginning with `t`: `t<lesson>.<major>.<minor>`. Lesson 1 is
`t1.0.0`. The `main` branch is the working branch and may be ahead of any lesson, so if
you want to follow along, check out the lesson tag rather than `main`.

```bash
git tag -l 't*'        # list the available lessons
git checkout t1.0.0    # jump to lesson 1
```

---

## 2. Requirements

- **Python 3.13 or newer** — this project targets 3.13
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **An API key** for an OpenAI-compatible chat endpoint serving `DeepSeek-V4.1-Flash`
- **git**, to fetch the lesson tags

If you don't have `uv` yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 3. Running it

### Get the code

```bash
git clone https://github.com/wuzm2k05/jca.git
cd jca
git checkout t1.0.0        # or the lesson you want
```

### Install dependencies

```bash
uv sync
```

This creates `.venv/` and installs everything the project declares. You never need to
activate the environment by hand — `uv run` takes care of it.

### Add your credentials

Copy the template and fill it in:

```bash
cp .env.example .env        # macOS / Linux
copy .env.example .env      # Windows (cmd)
```

`.env.example` is committed and lists **every** variable this project reads — the model
credentials, `JCA_WORKSPACE`, and the `JCA_LOG_*` set — with comments. **Copy the names
exactly.** A misspelled variable name is not an error; `os.getenv` simply returns nothing and
you get the default, which is the kind of silent difference that costs you an afternoon.

Two things worth checking if the program can't reach the model:

- the base URL normally needs the **`/v1` suffix**;
- the model name is set in the code, so if your endpoint serves a different model, edit it
  there.

> **Keep `.env` out of git.** It holds a live key. `.env.example` is the one that gets
> committed; `.gitignore` matches `.env` exactly, so the template is not caught by it.

### Run

```bash
uv run main.py                       # interactive session
uv run main.py --prompt "..."        # run one turn and exit
```

See the next section for what the interactive session looks like.

### The interactive CLI

`uv run main.py` with no arguments drops you into a session. It is a thin terminal
front end over the same engine `--prompt` uses — the agent loop, the tools and the
sandbox are unchanged; only the presentation differs.

```
┌──────────────────────────────── jca ────────────────────────────────┐
│ 模型  DeepSeek-V4.1-Flash                                           │
│ 工作目录  D:\work\src\jca-simple\workspace                          │
│ 日志  D:\work\src\jca-simple\logs\jca.log                           │
└──────────────── Ctrl-C 中断本轮（在提示符处则退出）· /exit 退出 ────┘
› 在 workspace 里建一个 hello.py，打印 Hello，然后运行它

  ● write_file(path=hello.py)
    ⎿ 已新建 hello.py，共写入 15 个字符。
  ● run_cmd(command=python hello.py)
    ⎿ Hello

⏺ 已完成：脚本已创建并运行，输出是 Hello
›
```

**Reading the marks:**

| Mark | Meaning |
| --- | --- |
| `›` | your prompt |
| `⏺` | the model talking (rendered as Markdown) |
| `●` | the model *asked* for a tool — name and arguments, one line |
| `⎿` | what the tool returned (truncated — the full output is in the log file) |
| `⚠` | a notice, e.g. the request was blocked by the server's content filter and is being retried |

The two marks in the middle are the whole point of the section on deciding vs doing:
`●` is a *request*, written by the model; `⎿` is the *result*, produced by your own
code. They never merge into one line.

**The tools** are the whole of what the agent can do (`tools/files.py`, `tools/shell.py`,
plus one supplied by a middleware — see [below](#pre-llm-middleware)):

| Tool | What it does |
| --- | --- |
| `list_dir(path=".", max_depth=2)` | lists a directory tree, indented, skipping `.git`, `.venv`, `node_modules`, `__pycache__`, `dist`, `build`, `.idea`, `.vscode` |
| `find_files(pattern="*", path=".")` | recursive wildcard search — `*.py`, `*auth*`, `tools/*.py` — same skip list, first 100 matches |
| `read_file(path, start_line=1, line_count=None)` | reads a file with a line-number column, in slices if you ask for them |
| `write_file(path, content)` | creates or overwrites |
| `edit_file(path, old_text, new_text)` | one exact-string replacement |
| `replace_lines(path, start_line, end_line, new_content)` | replaces an inclusive line range, by the numbers `read_file` printed |
| `run_cmd(command)` | one command, in the workspace root; output over 2000 chars comes back head+tail, with the full text in `.jca/logs/cmd_last.log` |
| `write_todos(todos)` | replaces the plan — a list of `{content, status}` with `status` one of `pending` / `in_progress` / `completed`. Supplied by `TodoMiddleware`, not by `tools/`; see [the middleware section](#pre-llm-middleware) |

`read_file` and `replace_lines` are deliberately numbered the same way, so the model can
read lines 200-260, decide what is wrong, and write that range back without ever pasting
the text it wants to match. That pairing is what makes long files work: a file too big for
one read comes back as "this chunk ends at line 199, continue with `start_line=200`"
instead of the old silent truncation, and a fifty-line function can be replaced without a
single whitespace mismatch. New content is written back in the file's own encoding and
line-ending style, and an out-of-range line number is refused rather than clamped — a
silently short edit is much harder to notice than an error message.

`run_cmd` handles the same problem on the command side. A build log or a test run can be
tens of thousands of characters, and the part that matters — the traceback, the exit-code
line — is at the *end*. So anything over `MAX_OUTPUT_CHARS` is returned as its first 500
and last 1500 characters joined by a `... [中间省略 N 字符] ...` marker, and the
untruncated text is written to `.jca/logs/cmd_last.log` inside the workspace so the model
can `read_file` it. That path is inside the sandbox on purpose: the notice tells the model
to read it, and a file it cannot reach would just be a dead end. If the write fails
(read-only directory, permissions), the command result is still returned — with a notice
that says so rather than pointing at a file that is not there.

**Commands.** `/exit`, `/quit` and `/q` all leave. `Ctrl-C` interrupts the turn in
progress and keeps the session; at the prompt it exits. `Ctrl-D` (or the end of piped
input) exits too. There is no `/help` and no history/completion — no raw-mode terminal
handling, so nothing to go wrong on Windows consoles. Feel free to add them.

**Multi-turn memory** is kept by the graph's checkpointer, keyed on a `thread_id` — the
UI itself stores no history. Everything you said earlier in the session is visible to
the model, including files it wrote (it reads them back with `read_file` rather than
remembering their contents).

---

### Pre-LLM middleware

`engine/middleware/` is where "how to call the model" lives. It used to be hard-coded in
`engine/agent.py`'s `_agent_node`: one system prompt fixed at construction, tools bound
once with `bind_tools`, a `SystemMessage` pushed into the state history every turn. Anything
you wanted to change about that — mount a tool only when it is needed, inject fresh context,
compact the history, RAG in a skill — meant editing the node itself.

Now a middleware is an object with three hooks, and the graph grew two nodes to run them:

```
START → before_model → agent → after_model ─┬─→ tools ─┐
                                             └─→ END    │
                                   ↑                    │
                                   └────────────────────┘
```

| Hook | Changes | Runs |
| --- | --- | --- |
| `before_model(state)` | **the state** — returns an update dict | in the `before_model` node, in declaration order |
| `after_model(state)` | **the state** — returns an update dict | in the `after_model` node, in *reverse* order |
| `wrap_model(request, handler)` | **this one call** — prompt, tool set, messages | inside the `agent` node, nested outermost-first |

The two hook nodes sit **inside the ReAct loop** — `tools` returns to `before_model`, not
to `agent` — so every "model → tool → model" round recomputes its context. That is
deliberate: a tool result can invalidate something a middleware wanted to say, and the
context is cheap next to a model request.

Two rules come with the two hook nodes, and both are silent when broken:

- **`after_model` must not append messages.** The next hop is `tools_condition`, which
  reads the *last* message: adding a `HumanMessage` there tells it the model did not ask
  for a tool, so the graph ends and the tool the model just requested is never executed.
  Nothing raises.
- **The safety retry never reaches `after_model`.** `agent` has no static out-edge; it
  returns `Command(goto=...)` — `after_model` on success, `before_model` on a retry, since
  a blocked round has no model reply for a post-model hook to see. This is also why the
  retry has to be a *Command* rather than a plain return: `Command.goto` **adds** a branch
  rather than replacing the static edge, so with an `agent → after_model` edge in place the
  blocked round would run both hooks in the same superstep — which crashes the graph
  outright the moment any middleware writes the same state key from both.

`wrap_model` is the one that matters most. It receives a `ModelRequest` — a throwaway object
holding the state, the messages to send, the system prompt and the tool set — and a
`handler` that means "the rest of the chain, ending in the real call". Calling
`handler(request)` hands over; returning an `AIMessage` yourself is also legal (that would
be a middleware deciding *for* the model), but every middleware here only adds to the
request. The request is built fresh on each call and dropped afterwards, which is why the
system prompt is **never stored in the state**: a prompt kept in history gets re-sent on
every subsequent turn, burns tokens, and makes "who changed it, and when" unanswerable.
The one thing that *is* state — the todo list — is re-read from the state and re-rendered
on each call, so the model always sees the current version instead of a stale copy.

Tools follow the same split. `tools/` is the **resident** registry (`build_tools()`):
`list_dir`, `read_file`, `run_cmd` and friends exist in every session. A tool that a
middleware mounts lives next to that middleware instead, because such a tool and the state
field it writes are two halves of one contract — `write_todos` writes the `todos` field
that `TodoMiddleware` declares. The tool set is computed per call
(`MiddlewarePipeline.tools`), so "always mount" and "mount only in these rounds" are one
line apart.

`TodoMiddleware` is the first one. It adds `todos` to the state, mounts `write_todos`, and
appends the current checklist to the system prompt on every call:

```
● write_todos(todos=[{content: "read engine/agent.py", status: "in_progress"}, ...])
⎿ 已更新 3 条待办（1 项进行中，2 项待办）
```

`write_todos` returns a `Command` rather than a string, which is what makes "the tool
synchronises the state" a single step: the same return value writes `todos` **and** hands
the model its `ToolMessage` receipt. There is no `after_model` pass that scans what the
model just called and copies the result into the state — that copy would eventually drift
from the truth. (The matching `ToolMessage` is not optional: `ToolNode` rejects a command
without one.) The `todos` field carries a last-write-wins reducer, because a model calling
`write_todos` twice in one round — which it does, despite the tool description asking it not
to — would otherwise kill the graph with `InvalidUpdateError: Can receive only one value
per step`, surfacing to you as a generic "调用模型失败" error.

The state schema follows the same principle. A middleware declares its own fields in
`state_schema`, and `MiddlewarePipeline.build_state_schema` merges them into the graph's
state at build time — adding a middleware should not mean editing `engine/agent.py`. The
merge reads `get_type_hints(..., include_extras=True)`, because the reducers live *inside*
`Annotated` and the default strips them; a duplicated field name is a startup error rather
than a silent last-one-wins.

**Symbols and old terminals.** On a console that cannot show `⏺`/`⎿` (a GBK code page
in `cmd.exe`, for instance) set `JCA_ASCII=1` and they degrade to `>`, `-`, `!`:

```bash
JCA_ASCII=1 uv run main.py
```

**`--prompt` is deliberately plainer.** One turn, no re-drawing, no colour, no Markdown
— just the answer, the tool lines and the results. That is what you want when the output
is being piped into a file or read by a script. It exits `0` on success and `1` if the
turn failed, so `&&` chains work. Logs still go to the terminal here, as they always did,
so a failing turn shows a stack trace above the one-line error — set
`JCA_LOG_TARGET=file` if you would rather it did not.

**Why the logs move to a file here.** The terminal logger writes to `sys.stdout` directly,
and the live display needs to own `sys.stdout` to redraw in place; both at once gives you
garbled output. So in interactive mode the log target **defaults to `file`** — the banner
tells you where it went, because "my logs disappeared" is harder to diagnose than "the
output looks scrambled". An explicit `JCA_LOG_TARGET` is never overridden: if you set it
to `console` or `both`, that is honoured and you get a one-line warning instead. Under
`--prompt` nothing changes — there is no live display there, so logs still go to the
terminal by default.

**Two known rough edges**, both consequences of the design rather than bugs:

- a turn that fails (network error, safety block) leaves your message in the session
  history, so the next turn may see two consecutive user messages;
- Ctrl-C during a turn keeps whatever was already printed, but the interrupted turn does
  not resume.

---

### Logging

The project logs through **structlog on top of the standard library `logging`** — there is
no `print` left in the code. Every module does:

```python
from logger import get_logger

logger = get_logger(__name__)
logger.info("执行命令", cmd=command)
```

`main.py` calls `setup_logging()` once, right after `load_dotenv()`.

**Where the logs go** is set by environment variables (put them in `.env` next to your API
key, or set them in the shell):

| Variable | Default | Meaning |
| --- | --- | --- |
| `JCA_LOG_TARGET` | `console` (`file` in the interactive CLI) | `console`, `file`, or `both` |
| `JCA_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, … |
| `JCA_LOG_DIR` | `logs` | directory for log files, relative to the working directory |
| `JCA_LOG_FILE` | `jca.log` | file name |
| `JCA_LOG_MAX_BYTES` | `10485760` | rotate once a file passes this size (10 MB) |
| `JCA_LOG_BACKUP_COUNT` | `5` | how many rotated files to keep |

That default is the only place the two modes differ: without `JCA_LOG_TARGET` set, an
interactive session logs to `logs/jca.log` and `--prompt` logs to the terminal. Set the
variable and it means exactly what it says in both. The reason is in
[the CLI section](#the-interactive-cli) — the live display and a terminal logger
fight over the same cursor.

```bash
# macOS / Linux — write to both the terminal and ./logs/
JCA_LOG_TARGET=both JCA_LOG_LEVEL=DEBUG uv run main.py

# Windows (PowerShell)
$env:JCA_LOG_TARGET='both'; $env:JCA_LOG_LEVEL='DEBUG'; uv run main.py
```

An unknown value for any of these is a **startup error, not a silent fallback** — a typo in
`JCA_LOG_LEVEL` should never be the reason your logs went quiet.

**Format.** The terminal gets colourised, human-readable lines; the file gets one JSON
object per line. Two renderers, one event:

```
2026-10-16T10:02:16.410199Z [info     ] 执行命令        [tools.shell] cmd='python app.py'

{"event": "执行命令", "cmd": "python app.py", "level": "info",
 "logger": "tools.shell", "timestamp": "2026-10-16T10:02:16.410199Z"}
```

Colour is only turned on when stdout is a real terminal, so `uv run main.py > out.txt`
gives you a clean file.

**Rotation** is in-process (`logging.handlers.RotatingFileHandler`) — no external
`logrotate` needed. It also means one process per log directory: two processes writing the
same file will fight over each other's rotation.

Two things worth knowing:

- third-party logs (`httpx`, `openai`, `langchain`) land in the same file, through the same
  rotation, because everything is funnelled through the standard library;
- keep the log directory **out of the agent's workspace** (see the next section). It is the
  record of what the agent did; the agent should not be able to read or edit it. The two
  defaults collide — workspace defaults to the run directory and logs default to `logs/`
  inside it — so if you care, set `JCA_WORKSPACE=workspace` (workspace shrinks to a
  subdirectory) or point `JCA_LOG_DIR` somewhere else.

---

## 4. The workspace

Everything the agent can touch lives under one root, the **workspace**. Where that root is,
is set by `JCA_WORKSPACE`:

| `JCA_WORKSPACE` | Workspace root | What it means |
| --- | --- | --- |
| unset (**default**) | the directory you run from | the agent works **in your project directory** |
| `workspace` | `<pwd>/workspace` | the agent gets its own scratch directory |
| `D:\sandbox\agent` (absolute) | `D:\sandbox\agent` | an absolute path overrides `<pwd>` entirely |

```bash
# macOS / Linux — keep the agent in ./workspace/
JCA_WORKSPACE=workspace uv run main.py

# Windows (PowerShell)
$env:JCA_WORKSPACE='workspace'; uv run main.py
```

Or put it in `.env` next to your API key — it is read from the same file:

```ini
JCA_WORKSPACE=workspace
```

With `JCA_WORKSPACE=workspace` the directory is created on first write (a `write_file` into
it makes the parent directories). If you point the workspace at something that does not
exist and the agent then runs a command, `run_cmd` reports the missing directory rather than
creating it — where commands run is your decision, not a side effect.

---

## 5. Tests

`tests/` holds four offline suites. None needs an API key or a terminal, and none
needs pytest — they are plain functions with a `main()`:

```bash
uv run python -m tests.test_engine_events    # the agent loop, against a scripted model
uv run python -m tests.test_middleware       # the middleware pipeline and TodoMiddleware
uv run python -m tests.test_ui_transcript    # events in, rendered text out
uv run python -m tests.test_tools            # the file and shell tools, against real temp dirs
```

They exist because of how this project is split. The engine takes a `BaseChatModel` as a
constructor argument, so a fake model that emits hand-written chunks can drive the real
`StateGraph`, the real `ToolNode` and the real translation layer end to end — no network,
no cost. That is what pins down the parts that are easy to get wrong and hard to see:
that tool arguments come from the parsed message rather than self-assembled chunks, that
the safety retry note is turned into a status line instead of fake assistant text, and
that a failed call becomes an error event rather than a stack trace on the user's screen.

The UI suite feeds synthetic events into the same function the live display uses. What it
checks is that each event lands somewhere sensible — a tool result attaches to its call,
a multi-line result is folded, an unknown event type is ignored rather than crashing.

The middleware suite is the one that tests things you cannot see from outside. A middleware
whose order is reversed still runs; a system prompt that accumulates two copies still gets
an answer; a `write_todos` that returns a string instead of a `Command` still lets the
conversation continue — the model just never sees its own plan again, and nothing raises.
So it takes evidence from the model's side: the fake model records every message list it
is handed and every tool set it is bound to, and the assertions are about those records.
One case drives the whole graph — the model calls `write_todos`, the tool writes the state,
and the **next** call's system prompt contains the checklist — which is four separately
working mechanisms in a row, none of which fails loudly on its own. Three regression cases
came out of reviewing the design and pin failures that are all silent: a hook node that
annotates its `state` parameter with the base state and therefore never sees `todos`; a
retry that runs `after_model` on a round with no model reply; two `write_todos` calls in
one round. Two more count `before_model` invocations across a safety retry and check the
merged state schema keeps its reducers.

The tools suite is the one that touches a disk, so it works in a temporary directory it
creates **inside** the working directory and deletes afterwards — `safe_path` only allows
paths under the workspace, so a system temp directory would be testing something the agent
can never do. It plants a small tree with a nested package and a `.git`/`.venv` decoy in
it, and checks the things that fail quietly: that the ignored names really are gone from
the listing (and that "the directory is empty" and "everything in it was ignored" are
reported as two different things), that `start_line`/`line_count` slices line up with the
numbers `replace_lines` takes, that a CRLF + GBK file survives a line edit unchanged
otherwise, that an out-of-range edit leaves the file untouched, that `ls ../..`,
`cat /etc/passwd` and `type C:\Windows\...` are all refused while `dir /s` is not, and
that a long command's output keeps both its head and its tail while the saved file holds
the whole thing — the tail assertion is the one that matters, since trimming from the
end would satisfy "it got shorter" just as well.

If you prefer pytest, they are named to be collected (`uv add --dev pytest`):
`uv run pytest tests/`.

---

## ⚠️ Safety note

The agent's file tools — `list_dir`, `find_files`, `read_file`, `write_file`, `edit_file`,
`replace_lines` — are confined to the workspace. Every path is resolved and checked before
use, so `../.env` and `C:\Windows\...` are both refused.

**The default root is the directory you run from**, so out of the box the agent can read and
write your project files, `main.py` included. Set `JCA_WORKSPACE=workspace` to give it its
own directory instead.

```bash
# macOS / Linux
JCA_ALLOW_SHELL=1 uv run main.py

# Windows (PowerShell)
$env:JCA_ALLOW_SHELL='1'; uv run main.py
```

Once enabled, that tool (`run_cmd`) still only accepts a **single** command — pipes,
redirects and `&&` are rejected with an explanation, so the model cannot chain its way
around the sandbox. It also reports the exit code, so a silent failure no longer looks like
success.

Its arguments are checked against the workspace too: a token that is an absolute path
(`/etc/passwd`, `C:\Windows\win.ini`, `\\server\share`) or that climbs out with `..`
(`ls ../..`) is refused before anything runs, because `cwd=root` only decides where the
command *starts* — it cannot stop the command naming an absolute path itself.

**This is not a complete sandbox, and it does not pretend to be.** Three limits are worth
knowing:

- the command check is **lexical**, done on the raw text before the shell ever sees it, so
  it catches the obvious spellings and nothing else. `python -c "open('/etc/passwd')"` gets
  through, and so does any path the command assembles at runtime. It is a guard-rail against
  a model wandering out by accident, not a boundary against one trying to;
- one deliberate gap in that check: a single-segment `/name` is treated as an option, not a
  path, so `dir /s` and `tasklist /fi ...` keep working on Windows — at the cost of `/tmp`
  slipping through. Refusing every leading `/` would have made the tool nearly unusable on
  the platform where the `/` option prefix is normal;
- the confinement is path-level, not permission-level — no process isolation, no resource
  limits, no human approval step.

**Run it on a disposable machine or a container.** Do not point it at a machine holding
real credentials or data. Expect the code to ask you to make it safer; that is part of the
exercise — the pieces are deliberately small enough to read and change.


## installer
uv run pyinstaller jca.spec