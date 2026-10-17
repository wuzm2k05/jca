# jca-simple

A runnable **ReAct agent** built with **LangGraph + LangChain**.

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
uv run main.py
```

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
| `JCA_LOG_TARGET` | `console` | `console`, `file`, or `both` |
| `JCA_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, … |
| `JCA_LOG_DIR` | `logs` | directory for log files, relative to the working directory |
| `JCA_LOG_FILE` | `jca.log` | file name |
| `JCA_LOG_MAX_BYTES` | `10485760` | rotate once a file passes this size (10 MB) |
| `JCA_LOG_BACKUP_COUNT` | `5` | how many rotated files to keep |

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

## ⚠️ Safety note

The agent's file tools — `list_dir`, `read_file`, `write_file`, `edit_file` — are confined
to the workspace. Every path is resolved and checked before use, so `../.env` and
`C:\Windows\...` are both refused.

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

**This is not a complete sandbox, and it does not pretend to be.** Two limits are worth
knowing:

- the path check covers the file tools only. With `JCA_ALLOW_SHELL=1`, an absolute path
  inside a command (`type C:\Windows\...`) can still step outside the workspace;
- the confinement is path-level, not permission-level — no process isolation, no resource
  limits, no human approval step.

**Run it on a disposable machine or a container.** Do not point it at a machine holding
real credentials or data. Expect the code to ask you to make it safer; that is part of the
exercise — the pieces are deliberately small enough to read and change.
