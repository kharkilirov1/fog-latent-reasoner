# FOG v3: простое обучение

Этот файл — практическая инструкция без необходимости вручную собирать десятки аргументов `train_real.py`.

## Самая простая проверка

Из корня репозитория:

```bash
python train_v3_easy.py --install --recipe smoke
```

Скрипт сам:

1. проверит/поставит training-зависимости;
2. найдёт `tokenizer/tinystories_3k_bpe.json`;
3. возьмёт `checkpoints/fog_machine_v3_10m_init.pt`;
4. использует вложенный TinyStories 3k как smoke-data;
5. сохранит `last.pt` / `best.pt`;
6. при повторном запуске автоматически продолжит незаконченный этап или пропустит уже законченный.

Smoke нужен **только для проверки pipeline**. Вложенные 3000 TinyStories слишком малы для нормального предобучения.

## Первый нормальный запуск на своих текстах

Локальный JSONL должен содержать поле `text`:

```json
{"text": "Some training document..."}
{"text": "Another document..."}
```

Запуск:

```bash
python train_v3_easy.py \
  --install \
  --recipe starter \
  --device cuda \
  --text-data /path/to/train.jsonl \
  --text-eval-data /path/to/validation.jsonl
```

Для CUDA `precision=auto`: BF16 выбирается, если GPU его поддерживает, иначе FP16.

## Если датасет на Hugging Face

```bash
python train_v3_easy.py \
  --install \
  --recipe starter \
  --device cuda \
  --dataset-id roneneldan/TinyStories
```

Можно добавить `--dataset-config ...` и `--revision ...`.

## Подключить reasoning/SFT после lexical pretrain

SFT JSONL:

```json
{"prompt": "Question ...", "response": "Answer ..."}
{"prompt": "Question ...", "response": "Answer ..."}
```

Один запуск:

```bash
python train_v3_easy.py \
  --recipe starter \
  --device cuda \
  --text-data /path/to/train.jsonl \
  --text-eval-data /path/to/validation.jsonl \
  --sft-data /path/to/reasoning_train.jsonl \
  --sft-eval-data /path/to/reasoning_validation.jsonl
```

По умолчанию `starter` делает:

- lexical pretrain: 2000 optimizer steps;
- SFT R=1, `decoder_mode=full`: 300 steps — более мягкий semantic warmup;
- SFT R=2, `memory-only`: 500 steps;
- SFT R=4, `memory-only`: 800 steps.

То есть сложность curriculum спрятана внутри wrapper.

> Важно: semantic binding естественного языка остаётся исследовательской частью FOG v3. Поэтому staged SFT — экспериментальный этап, а не доказанный рецепт SOTA-обучения.

## Режимы

- `--recipe smoke` — проверить установку и pipeline;
- `--recipe starter` — первый рабочий запуск;
- `--recipe serious` — длинный запуск после того, как starter-метрики выглядят нормально.

Не начинайте с `serious`, пока `starter` не прошёл validation без нестабильности.

## Resume

Ничего делать не нужно. Просто повторите **ту же команду**.

Например, если обучение остановилось:

```bash
python train_v3_easy.py --recipe starter --device cuda --text-data data/train.jsonl
```

Если `runs/v3_easy/01_pretrain/last.pt` существует, wrapper передаст его как `--resume`. Уже завершённые этапы будут пропущены.

## Где лежат веса

По умолчанию:

```text
runs/v3_easy/
├── 01_pretrain/
│   ├── best.pt
│   └── last.pt
├── 02_sft_r1_full/
├── 03_sft_r2_memory_only/
├── 04_sft_r4_memory_only/
└── pipeline_manifest.json
```

В конце wrapper печатает путь к текущему финальному checkpoint.

## Если мало VRAM

Уменьшите batch и компенсируйте accumulation:

```bash
python train_v3_easy.py \
  --recipe starter \
  --device cuda \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --text-data data/train.jsonl
```

Если всё ещё OOM:

```bash
--sequence-length 128
```

## Только lexical pretrain

Просто не передавайте `--sft-data`.

Это хороший первый реальный запуск: сначала обучить backbone, посмотреть validation loss/perplexity, а потом подключать reasoning data.

## Кастомная глубина SFT

Например:

```bash
--sft-depths '1:500:full,2:800:memory-only,4:1200:memory-only,8:1500:memory-only'
```

Формат:

```text
reasoning_steps:max_steps:decoder_mode
```

## Два удобных shell-скрипта

Smoke:

```bash
./scripts/train_v3_smoke.sh
```

Starter:

```bash
./scripts/train_v3_starter.sh --device cuda --text-data /path/to/train.jsonl
```
