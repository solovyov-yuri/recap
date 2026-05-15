# recap

Инструмент для транскрибации аудиозаписей встреч и автоматического составления саммари в формате Telegram.

**Пайплайн:** WAV → [Whisper large-v3] → транскрипция → [LLM] → саммари в Telegram Markdown.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU с CUDA (для Whisper)
- Локальный LLM-сервер или API-ключ OpenAI

## Установка

```bash
git clone https://github.com/solovyov-yuri/recap.git
cd recap
uv sync
```

NVIDIA CUDA-библиотеки (cublas, cudnn) устанавливаются как Python-пакеты внутри venv — дополнительной системной установки не нужно.

## Конфигурация

Приоритет: **CLI-флаги > переменные среды > `config.yaml` > умолчания**.

```bash
cp config.yaml.example config.yaml
```

### config.yaml

```yaml
provider: ollama          # openai | ollama | lm-studio | vllm
model: qwen3.5:latest
language: ru
whisper_model: large-v3
max_transcript_chars: 60000   # транскрипт обрезается до этого лимита перед отправкой в LLM
# api_key: sk-...
# base_url: http://...
```

`config.yaml` не коммитится (в `.gitignore`). Полный список полей — в `config.yaml.example`.

### Переменные среды

Переопределяют любое поле из `config.yaml`:

```bash
MEETING_SUM_PROVIDER=openai
MEETING_SUM_MODEL=gpt-4o-mini
MEETING_SUM_API_KEY=sk-...        # или стандартный OPENAI_API_KEY
MEETING_SUM_MAX_TRANSCRIPT_CHARS=30000
```

### Провайдеры

| Провайдер | base_url (по умолчанию) |
|---|---|
| `openai` | api.openai.com |
| `ollama` | http://localhost:11434/v1 |
| `lm-studio` | http://localhost:1234/v1 |
| `vllm` | http://localhost:8000/v1 |

`base_url` можно переопределить вручную для нестандартных адресов.

**OpenAI:**
```yaml
provider: openai
model: gpt-4o-mini
api_key: sk-...
```

**Ollama:**
```yaml
provider: ollama
model: qwen3.5:latest
```

**LM Studio:**
```yaml
provider: lm-studio
model: local-model
```

**Кастомный endpoint:**
```yaml
provider: openai
model: my-model
base_url: http://my-server:9000/v1
api_key: secret
```

## Использование

### Полный пайплайн

```bash
uv run recap run                            # data/meeting.wav, провайдер из config
uv run recap run path/to/call.wav -p ollama
```

Записывает транскрипцию в `data/transcript.txt`, саммари — в `data/summary.txt`.

### По шагам

**Транскрибация:**
```bash
uv run recap transcribe meeting.wav
uv run recap transcribe call.wav -o call.txt -l en
```

**Саммари из готовой транскрипции:**
```bash
uv run recap summarize
uv run recap summarize call.txt -o call_summary.txt -p openai
uv run recap summarize call.txt -p lm-studio -m llama-3.1-8b
```

### Опции

| Опция | Команды | Описание |
|---|---|---|
| `-o, --output PATH` | transcribe, summarize | Файл вывода |
| `-l, --language TEXT` | transcribe, run | Язык аудио (`ru`, `en`, …) |
| `-p, --provider TEXT` | summarize, run | Провайдер (см. таблицу выше) |
| `-m, --model TEXT` | summarize, run | Модель (переопределяет config) |
| `--transcript PATH` | run | Путь для транскрипции |
| `--summary PATH` | run | Путь для саммари |
| `-v, --verbose` | все | Показывать прогресс |

## Разработка

```bash
uv sync --group dev
uv run pytest -v
```
