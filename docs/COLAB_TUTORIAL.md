# Подробный туториал: запуск MC-LLM в Google Colab

Здесь по шагам: как загрузить проект в Colab и запустить обучение на GPU.

---

## Шаг 0. Что нужно заранее

У тебя на компьютере (Windows) лежит папка проекта `LyraAi` (или `mega-cyber-llm`)
со всем кодом. Нам нужно перенести её в Colab одним из трёх способов (Шаг 2).

---

## Шаг 1. Подготовить архив проекта (на твоём компьютере)

Проще всего перенести проект одним ZIP-архивом.

**Способ A — через проводник:**
1. Открой папку, где лежит проект.
2. Зайди **внутрь** папки проекта, выдели все файлы и папки
   (`Ctrl+A`).
3. Правый клик → **Отправить → Сжатая ZIP-папка**.
4. Назови его, например, `mega-cyber-llm.zip`.

**Способ B — через PowerShell** (запусти в папке проекта):
```powershell
# находясь ВНУТРИ папки проекта:
Compress-Archive -Path * -DestinationPath ..\mega-cyber-llm.zip -Force
```

> Важно: архивируй **содержимое** проекта (так, чтобы внутри zip сразу были
> `model/`, `scripts/`, `configs/`, `README.md` и т.д.), а не саму папку.
> Это избавит от лишнего уровня вложенности в Colab.

---

## Шаг 2. Загрузить проект в Colab (3 способа — выбери один)

### Способ 1 — Загрузить ZIP через интерфейс (проще всего)

1. Открой [colab.research.google.com](https://colab.research.google.com) →
   **New notebook** (или File → New notebook).
2. Слева нажми значок папки **Files** (📁).
3. Нажми кнопку **Upload** (стрелка вверх), выбери `mega-cyber-llm.zip`.
4. Жди, пока загрузится (появится в списке слева).
5. В первой ячейке кода выполни:

```python
import zipfile, os

with zipfile.ZipFile('/content/mega-cyber-llm.zip', 'r') as z:
    z.extractall('/content/mega-cyber-llm')

%cd /content/mega-cyber-llm
!ls
```

После этого ты должен увидеть список `model/ scripts/ configs/ README.md ...`.

### Способ 2 — Через Google Drive (удобно сохранять чекпоинты)

1. Залей `mega-cyber-llm.zip` в свой Google Drive (drag&drop в drive.google.com).
2. В Colab выполни:

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os
with zipfile.ZipFile('/content/drive/MyDrive/mega-cyber-llm.zip', 'r') as z:
    z.extractall('/content/mega-cyber-llm')

%cd /content/mega-cyber-llm
```

Бонус: в конце обучения можно скопировать чекпоинты обратно на Drive (см. Шаг 7).

### Способ 3 — Через GitHub (если залил проект в репозиторий)

```python
# замени URL на свой
!git clone https://github.com/ТВОЙ_ЛОГИН/mega-cyber-llm.git
%cd mega-cyber-llm
```

---

## Шаг 3. Включить GPU

1. В меню Colab: **Runtime → Change runtime type**.
2. **Hardware accelerator** → выбери **T4 GPU** (или A100, если есть Colab Pro).
3. Нажми **Save**.

Проверь, что GPU виден:

```python
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Должно вывести `CUDA: True` и название GPU.

---

## Шаг 4. Установить зависимости

```python
!pip install -q pyyaml tqdm psutil
```

`torch` и `numpy` в Colab уже установлены.

---

## Шаг 5. Запустить обучение

Полный pipeline: **генерация диалогов → токенизация → обучение → генерация**.

```python
!python scripts/colab_train.py --conv-tokens 2000000 --steps 500
```

Что произойдёт:
1. Создастся conversational датасет (многоходовые русские диалоги) — до 2 млн токенов.
2. Обучится CyberTokenizer (если его ещё нет).
3. Диалоги токенизируются в `data/shards/`.
4. Обучится модель (~100M параметров) на GPU.
5. Напечатаются примеры генерации.

### Параметры, которые стоит менять

| Флаг | Что делает | Пример |
|------|-----------|--------|
| `--conv-tokens` | лимит токенов датасета (0 = без лимита, до упора ресурсов) | `--conv-tokens 10000000` |
| `--steps` | число шагов обучения | `--steps 2000` |
| `--batch` | размер батча | `--batch 8` |
| `--seq-len` | длина контекста | `--seq-len 512` |
| `--hidden` / `--layers` | размер модели | `--layers 16` |
| `--lr` | learning rate | `--lr 3e-4` |

### Только генерация датасета (без обучения)

```python
!python scripts/generate_conversation_dataset.py --max-tokens 5000000
```

---

## Шаг 6. Что увидишь в процессе

В логе будет примерно такое:

```
[colab] GPU: Tesla T4 (14.7 GB VRAM)
[gen] trained tokenizer on phrase bank: vocab=...
[gen] flushed shard 0 (2000 dial, ... tok, ... MB)
CONVERSATIONAL DATASET REPORT
Total dialogues: ...
Total tokens: ...
...
[MC-LLM] device: Tesla T4 (14.7 GB), dtype=float16
[MC-LLM] parameters: 88,000,000
step 20/500 loss 5.12 lr ...
...
Q: Привет! Расскажи о себе.
A: ...
```

`loss` должен падать со временем.

---

## Шаг 7. Сохранить результаты (иначе Colab всё сотрёт)

Colab удаляет файлы после закрытия вкладки. Сохрани чекпоинты:

**Вариант A — на Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r checkpoints /content/drive/MyDrive/mc_llm_checkpoints
!cp -r tokenizer /content/drive/MyDrive/mc_llm_tokenizer
```

**Вариант B — скачать на компьютер:**
```python
from google.colab import files
import shutil
shutil.make_archive('mc_llm_checkpoints', 'zip', 'checkpoints')
files.download('mc_llm_checkpoints.zip')
```

---

## Шаг 8. Возобновить генерацию датасета после перезапуска

Если Colab перезапустился, диалоги не потеряны (они в `data/conversation/shards/`
на... тоже сотрутся! поэтому шарды тоже стоит копировать на Drive).

```python
!python scripts/resume_dataset_generation.py --max-tokens 10000000
```

Система сама найдёт последний сохранённый shard и продолжит с него.

---

## Частые проблемы

| Проблема | Решение |
|----------|---------|
| `CUDA: False` | Не включён GPU: Runtime → Change runtime type → T4 |
| `No module named 'model'` | Ты не в папке проекта: выполни `%cd /content/mega-cyber-llm` |
| `CUDA out of memory` | Уменьши `--batch` или `--seq-len`, или `--layers` |
| Долго генерируется датасет | Это нормально, датасет считается на CPU; можно ограничить `--conv-tokens` |
| Потерялись файлы после закрытия | Сразу копируй на Drive (Шаг 7) |
| `corpus too small` | Слишком мало диалогов для `--seq-len`; увеличь `--conv-tokens` |

---

## Быстрый старт (всё в одной ячейке)

```python
# 1. распаковать проект (если загрузил zip)
import zipfile
with zipfile.ZipFile('/content/mega-cyber-llm.zip', 'r') as z:
    z.extractall('/content/mega-cyber-llm')
%cd /content/mega-cyber-llm

# 2. зависимости
!pip install -q pyyaml tqdm psutil

# 3. запуск
!python scripts/colab_train.py --conv-tokens 2000000 --steps 500
```
