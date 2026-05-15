# recap

Инструмент для транскрибации аудиозаписей встреч и автоматического составления саммари в формате Telegram.

**Пайплайн:** WAV → [Whisper large-v3] → транскрипция → [Ollama LLM] → саммари в Telegram Markdown.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU с CUDA (для Whisper)
- [Ollama](https://ollama.com/) с загруженной моделью

```bash
ollama pull qwen3.5          # модель по умолчанию
```

## Установка

```bash
git clone https://github.com/solovyov-yuri/recap.git
cd recap
uv sync
```

NVIDIA CUDA-библиотеки (cublas, cudnn) устанавливаются как Python-пакеты внутри venv — дополнительной системной установки не нужно.

## Использование

### Полный пайплайн

```bash
uv run recap run                        # data/meeting.wav по умолчанию
uv run recap run path/to/call.wav
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
uv run recap summarize call.txt -o call_summary.txt -m gemma4:e4b
```

### Опции

| Опция | Команды | Описание |
|---|---|---|
| `-o, --output PATH` | transcribe, summarize | Файл вывода |
| `-l, --language TEXT` | transcribe, run | Язык аудио (`ru`, `en`, …) |
| `-m, --model TEXT` | summarize, run | Ollama-модель |
| `--transcript PATH` | run | Путь для транскрипции |
| `--summary PATH` | run | Путь для саммари |
| `-v, --verbose` | все | Показывать прогресс |

### Переменные окружения

Значения по умолчанию можно переопределить:

```bash
MEETING_SUM_AUDIO=data/meeting.wav
MEETING_SUM_TRANSCRIPT=data/transcript.txt
MEETING_SUM_SUMMARY=data/summary.txt
MEETING_SUM_LANGUAGE=ru
MEETING_SUM_WHISPER_MODEL=large-v3
MEETING_SUM_OLLAMA_MODEL=qwen3.5:latest
```

## Разработка

```bash
uv sync --group dev
uv run pytest -v
```
