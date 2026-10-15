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

Create a file named `.env` in the project root:

```ini
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_API_BASE_URL=https://your-endpoint-here/v1
```

Two things worth checking if the program can't reach the model:

- the base URL normally needs the **`/v1` suffix**;
- the model name is set in the code, so if your endpoint serves a different model, edit it
  there.

> **Keep this file out of git.** It holds a live key. If the lesson you are on does not
> already ignore `.env`, add it to your `.gitignore` before you commit anything.

### Run

```bash
uv run main.py
```
---

## ⚠️ Safety note

The agent's default tools are **file operations confined to `./workspace/`**: `list_dir`,
`read_file`, `write_file`, `edit_file`. Every path is resolved and checked before use, so
`../.env` and `C:\Windows\...` are both refused. The agent gets its own scratch directory
and cannot reach anything outside it.

Running commands on your machine is a **separate, opt-in capability**. It is off unless you
set `JCA_ALLOW_SHELL=1`:

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
  inside a command (`type C:\Windows\...`) can still step outside `./workspace/`;
- the confinement is path-level, not permission-level — no process isolation, no resource
  limits, no human approval step.

**Run it on a disposable machine or a container.** Do not point it at a machine holding
real credentials or data. Expect the code to ask you to make it safer; that is part of the
exercise — the pieces are deliberately small enough to read and change.
