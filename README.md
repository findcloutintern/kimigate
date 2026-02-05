# kimigate

gateway proxy for using claude code with the free kimi k2.5 thinking model.

```
claude code cli -> kimigate (localhost:8082) -> kimi k2.5 api
```

## one-liner install (windows)

```powershell
powershell -c "iwr -useb https://raw.githubusercontent.com/findcloutintern/kimigate/main/install.bat -OutFile install.bat; .\install.bat"
```

## requirements

- windows
- git
- node.js (for claude code)
- claude code cli (`npm install -g @anthropic-ai/claude-code`)
- nvidia nim api key (free at [build.nvidia.com](https://build.nvidia.com))

## usage

after install, open a new terminal and run:

```
kimigate
```

this starts the proxy server and launches claude code.

## configuration

edit `%USERPROFILE%\kimigate\.env`:

| setting | default | description |
|---------|---------|-------------|
| `NVIDIA_NIM_API_KEY` | - | your nvidia nim api key |
| `RATE_LIMIT` | 40 | requests per window |
| `RATE_WINDOW` | 60 | window in seconds |
| `TEMPERATURE` | 1.0 | model temperature |
| `MAX_TOKENS` | 81920 | max output tokens |

## uninstall

```powershell
%USERPROFILE%\kimigate\uninstall.bat
```

## how it works

kimigate translates anthropic api format to openai format that kimi k2.5 expects, then proxies responses back to claude code. includes optimizations to skip unnecessary api calls (quota checks, title generation, etc).

## manual install

```bash
git clone https://github.com/findcloutintern/kimigate.git
cd kimigate
uv sync
cp .env.example .env
# edit .env with your api key
uv run python server.py
```

then in another terminal:
```bash
ANTHROPIC_AUTH_TOKEN=kimigate ANTHROPIC_BASE_URL=http://localhost:8082 claude
```
