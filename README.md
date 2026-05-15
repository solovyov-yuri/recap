# meeting-sum

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
git clone https://github.com/solovyov-yuri/meeting-sum.git
cd meeting-sum
uv sync
```

NVIDIA CUDA-библиотеки (cublas, cudnn) устанавливаются как Python-пакеты внутри venv — дополнительной системной установки не нужно.

## Использование

### Полный пайплайн

```bash
uv run meeting-sum run meeting.wav
```

Записывает транскрипцию в `transcript.txt`, саммари — в `summary.txt`.

### По шагам

**Транскрибация:**
```bash
uv run meeting-sum transcribe meeting.wav
uv run meeting-sum transcribe call.wav -o call.txt -l en
```

**Саммари из готовой транскрипции:**
```bash
uv run meeting-sum summarize
uv run meeting-sum summarize call.txt -o call_summary.txt -m gemma4:e4b
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
MEETING_SUM_AUDIO=standup.wav
MEETING_SUM_LANGUAGE=en
MEETING_SUM_OLLAMA_MODEL=gemma4:e4b
MEETING_SUM_WHISPER_MODEL=large-v3
```

## Разработка

```bash
uv sync --group dev
uv run pytest -v
```
