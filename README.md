# recap

Инструмент для транскрибации аудиозаписей встреч и автоматического составления саммари в формате Telegram.

**Пайплайн:** WAV → [Whisper large-v3] → транскрипция → [Ollama / OpenAI] → саммари в Telegram Markdown.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU с CUDA (для Whisper)
- [Ollama](https://ollama.com/) с загруженной моделью **или** API-ключ OpenAI

```bash
ollama pull qwen3.5          # модель по умолчанию для провайдера ollama
```

## Установка

```bash
git clone https://github.com/solovyov-yuri/recap.git
cd recap
uv sync
```

NVIDIA CUDA-библиотеки (cublas, cudnn) устанавливаются как Python-пакеты внутри venv — дополнительной системной установки не нужно.

## Конфигурация

Настройки читаются в порядке приоритета: **CLI > config.yaml > defaults**.

### config.yaml

Скопируйте шаблон и отредактируйте под себя:

```bash
cp config.yaml.example config.yaml
```

**Ollama (по умолчанию):**
```yaml
provider: ollama
ollama_model: qwen3.5:latest
```

**OpenAI:**
```yaml
provider: openai
openai_model: gpt-4o-mini
openai_api_key: sk-...
```

**Локальный OpenAI-совместимый сервер (LM Studio, vLLM и т.д.):**
```yaml
provider: openai
openai_model: local-model
openai_base_url: http://localhost:1234/v1
```

Остальные поля с дефолтными значениями можно не указывать:
```yaml
language: ru
whisper_model: large-v3
audio: data/meeting.wav
transcript: data/transcript.txt
summary: data/summary.txt
```

`config.yaml` не коммитится (в `.gitignore`). Полный список полей — в `config.yaml.example`.


## Использование

### Полный пайплайн

```bash
uv run recap run                            # data/meeting.wav, провайдер из config
uv run recap run path/to/call.wav -p openai
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
uv run recap summarize call.txt -p openai -m gpt-4o
uv run recap summarize call.txt -p ollama -m gemma4:e4b
```

### Опции

| Опция | Команды | Описание |
|---|---|---|
| `-o, --output PATH` | transcribe, summarize | Файл вывода |
| `-l, --language TEXT` | transcribe, run | Язык аудио (`ru`, `en`, …) |
| `-p, --provider TEXT` | summarize, run | Провайдер: `ollama` или `openai` |
| `-m, --model TEXT` | summarize, run | Модель (переопределяет config) |
| `--transcript PATH` | run | Путь для транскрипции |
| `--summary PATH` | run | Путь для саммари |
| `-v, --verbose` | все | Показывать прогресс |


## Разработка

```bash
uv sync --group dev
uv run pytest -v
```
