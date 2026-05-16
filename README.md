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
RECAP_PROVIDER=openai
RECAP_MODEL=gpt-4o-mini
RECAP_API_KEY=sk-...        # или стандартный OPENAI_API_KEY
RECAP_MAX_TRANSCRIPT_CHARS=30000
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

### Режимы саммари

| Режим | Описание |
|---|---|
| `brief` | 2–3 предложения, только суть, без структуры |
| `medium` | Тема + ключевые обсуждения + решения и задачи (по умолчанию) |
| `detailed` | Полный протокол: участники, ход обсуждения, задачи с ответственными и сроками |

### Полный пайплайн

```bash
uv run recap run                               # data/meeting.wav, режим medium
uv run recap run path/to/call.wav -m brief     # краткое саммари
uv run recap run audio.wav -m detailed -p openai
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
uv run recap summarize call.txt -m brief
uv run recap summarize call.txt -m detailed -p openai --model gpt-4o
```

### Опции

| Опция | Команды | Описание |
|---|---|---|
| `-o, --output PATH` | transcribe, summarize | Файл вывода |
| `-l, --language TEXT` | transcribe, run | Язык аудио (`ru`, `en`, …) |
| `-m, --mode TEXT` | summarize, run | Режим: `brief` \| `medium` \| `detailed` |
| `-p, --provider TEXT` | summarize, run | Провайдер (см. таблицу выше) |
| `--model TEXT` | summarize, run | Модель LLM (переопределяет config) |
| `--transcript PATH` | run | Путь для промежуточной транскрипции |
| `--summary PATH` | run | Путь для саммари |
| `-v, --verbose` | все | Показывать прогресс |

> **Breaking change:** флаг `-m` переназначен с `--model` на `--mode`. Для указания модели используйте `--model`.

## Telegram Markdown

Саммари форматируется в **Telegram Markdown v1** (`parse_mode="Markdown"`):

| Элемент | Синтаксис |
|---|---|
| Жирный | `*текст*` |
| Курсив | `_текст_` |
| Код | `` `код` `` |
| Ссылка | `[текст](url)` |

**Ограничения:** Telegram Markdown v1 не поддерживает экранирование спецсимволов. Если в тексте встречаются несбалансированные `_` или `` ` `` (например, в технических терминах), Telegram может не принять сообщение. Промпт явно запрещает модели использовать `_` для форматирования, поэтому на практике это редко. Для полной совместимости со спецсимволами в будущем предполагается переход на MarkdownV2.

## Приватность

Транскрипт встречи отправляется в LLM только для саммари. Важно понимать, куда он уходит:

| Провайдер | Данные |
|---|---|
| `ollama`, `lm-studio`, `vllm` | Остаются локально — не покидают машину |
| `openai` или внешний `base_url` | Отправляются на внешний сервер |

При использовании внешнего endpoint CLI выводит предупреждение в stderr. Чтобы его отключить, добавьте в `config.yaml`:

```yaml
privacy_ack: true
```

Или через переменную среды: `RECAP_PRIVACY_ACK=true`.

## Разработка

```bash
uv sync --group dev
uv run pytest -v
```

## Troubleshooting

### Windows: ошибки доступа к кешу uv или temp

Если `uv run pytest` падает с ошибкой доступа к `%LOCALAPPDATA%\uv\cache` или системной temp-директории, задайте локальные пути:

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
$env:TMP = ".tmp"
$env:TEMP = ".tmp"
uv run pytest -v
```

Либо добавьте их в `.env` или в настройки профиля PowerShell.
