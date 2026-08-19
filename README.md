# <div align="center">PaperBanana 🍌</div>
<div align="center">Давэй Чжу (Dawei Zhu), Жуй Мэн (Rui Meng), Йель Сун (Yale Song), Сию Вэй (Xiyu Wei), Суцзянь Ли (Sujian Li), Томас Пфистер (Tomas Pfister) и Чжинсунг Юн (Jinsung yoon)
<br><br></div>

<div align="center">
<a href="https://huggingface.co/papers/2601.23265"><img src="assets/paper-page-xl.svg" alt="Страница статьи на HF"></a>
<a href="https://huggingface.co/datasets/dwzhu/PaperBananaBench"><img src="assets/dataset-on-hf-xl.svg" alt="Датасет на HF"></a>
<a href="https://huggingface.co/spaces/dwzhu/PaperBanana"><img src="assets/spaces-on-hf-xl.png" height="48" alt="Демо на HF Spaces"></a>
</div>

> Всем привет! Оригинальная версия PaperBanana уже открыта в исходном коде под эгидой Google-Research как [PaperVizAgent](https://github.com/google-research/papervizagent). 
Этот репозиторий является форком того репозитория и нацелен на дальнейшее развитие для лучшей поддержки создания академических иллюстраций к статьям — хотя мы и достигли значительного прогресса, впереди еще долгий путь к более надежной генерации и поддержке более разнообразных и сложных сценариев. PaperBanana задуман как полностью открытый проект, призванный облегчить создание академических иллюстраций для всех исследователей. Наша цель — принести пользу сообществу, поэтому в настоящее время у нас нет планов использовать его в коммерческих целях.

## Последние новости
- **2026-03-24**: PaperBanana теперь [размещен на Hugging Face Spaces](https://huggingface.co/spaces/dwzhu/PaperBanana). Большое спасибо команде Hugging Face за их поддержку.
- **2026-03-11**: Опубликован PaperBanana как [навык ClawHub](https://clawhub.ai/skills/paperbanana) — установите с помощью `clawhub install paperbanana`.
- **2026-03-11**: Добавлен выбор модели в UI Streamlit — теперь поддерживается выбор как основной модели (VLM), так и модели генерации изображений, с предустановленными параметрами и пользовательским вводом.
- **2026-03-11**: Добавлена поддержка OpenRouter — используйте модели от OpenAI, Anthropic и других провайдеров через единый API.
- **2026-03-11**: Добавлен раздел "Авторы" (Contributors) с поддержкой бота all-contributors.

## Список задач (TODO)
- [ ] Добавить поддержку использования вручную выбранных примеров. Предоставить **дружественный** пользовательский интерфейс.
- [ ] Загрузить код для генерации статистических графиков.
- [ ] Загрузить код для улучшения существующих диаграмм на основе руководства по стилю.
- [ ] Расширить набор эталонных изображений для поддержки большего количества областей, помимо компьютерных наук.

**PaperBanana** — это многоагентный фреймворк на основе референсов (эталонов) для автоматизированной генерации академических иллюстраций. Действуя как творческая команда специализированных агентов, он преобразует исходный научный контент в диаграммы и графики качества публикаций с помощью организованного конвейера из агентов: **Retriever (Искатель), Planner (Планировщик), Stylist (Стилист), Visualizer (Визуализатор) и Critic (Критик)**. Фреймворк использует контекстное обучение на эталонных примерах и итеративное улучшение для создания эстетически привлекательных и семантически точных научных иллюстраций.

Вот некоторые примеры диаграмм и графиков, сгенерированных PaperBanana:
![Примеры](assets/teaser_figure.jpg)

## Обзор PaperBanana

![Структура фреймворка PaperBanana](assets/method_diagram.png)

PaperBanana достигает высокого качества генерации академических иллюстраций за счет координации пяти специализированных агентов в структурированном конвейере:

1. **Агент Retriever (Искатель)**: Ищет наиболее релевантные эталонные диаграммы из тщательно подобранной коллекции для руководства последующими агентами.
2. **Агент Planner (Планировщик)**: Переводит содержание методов и коммуникативное намерение во всеобъемлющие текстовые описания, используя контекстное обучение.
3. **Агент Stylist (Стилист)**: Дорабатывает описания в соответствии с академическими эстетическими стандартами, используя автоматически синтезированные руководства по стилю.
4. **Агент Visualizer (Визуализатор)**: Преобразует текстовые описания в визуальные результаты, используя передовые модели генерации изображений.
5. **Агент Critic (Критик)**: Образует механизм усовершенствования с замкнутым циклом (closed-loop) вместе с Visualizer посредством многократных итеративных улучшений.

## Быстрый старт

### Шаг 1: Клонирование репозитория
```bash
git clone https://github.com/dwzhu-pku/PaperBanana.git
cd PaperBanana
```

### Шаг 2: Настройка
PaperBanana поддерживает настройку API-ключей через конфигурационный файл YAML или переменные окружения. 

Мы рекомендуем продублировать файл `configs/model_config.template.yaml` в `configs/model_config.yaml`, чтобы вынести все пользовательские конфигурации. Этот файл игнорируется git для сохранения в секрете ваших ключей API и конфигураций. В `model_config.yaml` не забудьте заполнить имена двух моделей (`defaults.main_model_name` и `defaults.image_gen_model_name`) и задать **хотя бы один** API-ключ в разделе `api_keys` — например, только `google_api_key` (Gemini) или только `openrouter_api_key` (OpenRouter). **Вам не нужны оба; любого одного достаточно.** Если настроены оба, предпочтение отдается OpenRouter (если он доступен) для маршрутизации.

Обратите внимание, что если вам нужно генерировать много кандидатов одновременно, вам потребуется API-ключ, поддерживающий высокую степень параллелизма (concurrency).

### Шаг 3: Загрузка датасета
Сначала скачайте [PaperBananaBench](https://huggingface.co/datasets/dwzhu/PaperBananaBench), затем поместите его в директорию `data` (например, `data/PaperBananaBench/`). Фреймворк разработан таким образом, чтобы корректно работать и без датасета, обходя способность агента Retriever к few-shot обучению. Если вас интересуют оригинальные PDF-файлы, скачайте их с [PaperBananaDiagramPDFs](https://huggingface.co/datasets/dwzhu/PaperBananaDiagramPDFs).

### Шаг 4: Установка окружения
1. Мы используем `uv` для управления пакетами Python. Пожалуйста, установите `uv`, следуя инструкциям [здесь](https://docs.astral.sh/uv/getting-started/installation/).

2. Создайте и активируйте виртуальное окружение
    ```bash
    uv venv # Это создаст виртуальное окружение в текущем каталоге, в папке .venv/
    source .venv/bin/activate  # или .venv\Scripts\activate в Windows
    ```

3. Установите python 3.12
    ```bash
    uv python install 3.12
    ```

4. Установите необходимые пакеты
    ```bash
    uv pip install -r requirements.txt
    ```

### Запуск PaperBanana

#### Вариант 1: Веб-приложение Gradio (Рекомендуется)

**Попробуйте онлайн — настройка не требуется:**  
👉 **[PaperBanana на Hugging Face Spaces](https://huggingface.co/spaces/dwzhu/PaperBanana)**

Чтобы начать, введите свой API-ключ (OpenRouter или Google Gemini), затем настройте желаемые параметры (режим конвейера, количество кандидатов, соотношение сторон и т. д.), вставьте текст раздела "методы" и подпись к рисунку и нажмите **Generate (Сгенерировать)**.

Вы также можете запустить приложение Gradio локально:
```bash
python app.py
```

Настройка **Figure Size (Размер рисунка)** в Gradio сопоставляет `1-3cm` и `4-6cm` с `1k`, `7-9cm` и `10-13cm` с `2k`, а `14-17cm` с `4k` для вызовов генерации изображений Gemini и OpenRouter. Запросы OpenAI `gpt-image` продолжают использовать существующий путь API с фиксированным размером.

#### Вариант 2: Интерактивная демонстрация (Streamlit)
Самый простой способ запустить PaperBanana — это интерактивная демонстрация Streamlit:
```bash
streamlit run demo.py
```

Веб-интерфейс предлагает два основных рабочих процесса:

**1. Вкладка "Generate Candidates" (Сгенерировать кандидатов)**:
- Вставьте содержание раздела "методы" (рекомендуется формат Markdown) и укажите подпись к рисунку.
- Настройте параметры (режим конвейера, настройки поиска (retrieval), количество кандидатов, соотношение сторон, количество раундов критика).
- Нажмите "Generate Candidates" и дождитесь параллельной обработки.
- Просматривайте результаты в виде сетки с таймлайнами эволюции и скачивайте отдельные изображения или все вместе в ZIP-архиве.

**2. Вкладка "Refine Image" (Улучшить изображение)**:
- Загрузите сгенерированного кандидата или любую диаграмму.
- Опишите желаемые изменения или запросите увеличение разрешения (upscale).
- Выберите разрешение (2K/4K) и соотношение сторон.
- Скачайте улучшенный результат в высоком разрешении.

#### Вариант 3: Интерфейс командной строки (CLI)
Вы также можете запустить PaperBanana из командной строки:
```bash
# Базовое использование с настройками по умолчанию
python main.py

# Продвинутое использование с пользовательскими настройками
python main.py \
  --dataset_name "PaperBananaBench" \
  --task_name "diagram" \
  --split_name "test" \
  --exp_mode "dev_full" \
  --retrieval_setting "auto"

# Устаревший способ (Legacy) генерации кода графиков matplotlib без few-shot поиска
python main.py \
  --dataset_name "PaperBananaBench" \
  --task_name "plot" \
  --split_name "test" \
  --exp_mode "vanilla" \
  --retrieval_setting "none"
```

**Доступные опции:**
- `--dataset_name`: Датасет для использования (по умолчанию: `PaperBananaBench`)
- `--task_name`: Тип задачи - `diagram` (диаграмма) или `plot` (график) (по умолчанию: `diagram`)
- `--split_name`: Выборка датасета (по умолчанию: `test`)
- `--exp_mode`: Режим эксперимента (см. раздел ниже)
- `--retrieval_setting`: Стратегия поиска - `auto`, `manual`, `random` или `none` (по умолчанию: `auto`)

**Режимы экспериментов:**
- `vanilla`: Прямая генерация без планирования или улучшения
- `dev_planner`: Retriever → Planner → Visualizer
- `dev_planner_stylist`: Retriever → Planner → Stylist → Visualizer
- `dev_planner_critic`: Retriever → Planner → Visualizer → Critic (несколько раундов)
- `dev_full`: Полный конвейер со всеми агентами
- `demo_planner_critic`: Демонстрационный режим (Retriever → Planner → Visualizer → Critic; без Stylist) без оценки (evaluation)
- `demo_full`: Демонстрационный режим (полный конвейер) без оценки

### Инструменты визуализации

Просмотр эволюции конвейера и промежуточных результатов:
```bash
streamlit run visualize/show_pipeline_evolution.py
```
Просмотр результатов оценки:
```bash
streamlit run visualize/show_referenced_eval.py
```

## Структура проекта
```
├── .venv/
│   └── ...
├── data/
│   └── PaperBananaBench/
│       ├── diagram/
│       │   ├── images/
│       │   ├── pdfs/
│       │   ├── test.json
│       │   └── ref.json
│       └── plot/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── retriever_agent.py
│   ├── planner_agent.py
│   ├── stylist_agent.py
│   ├── visualizer_agent.py
│   ├── critic_agent.py
│   ├── vanilla_agent.py
│   └── polish_agent.py
├── prompts/
│   ├── __init__.py
│   ├── diagram_eval_prompts.py
│   └── plot_eval_prompts.py
├── style_guides/
│   ├── generate_category_style_guide.py
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── paperviz_processor.py
│   ├── eval_toolkits.py
│   ├── generation_utils.py
│   └── image_utils.py
├── visualize/
│   ├── show_pipeline_evolution.py
│   └── show_referenced_eval.py
├── scripts/
│   ├── run_main.sh
│   ├── run_demo.sh
├── configs/
│   └── model_config.template.yaml
├── results/
│   ├── PaperBananaBench_diagram/
│   └── parallel_demo/
├── main.py
├── demo.py
└── README.md
```

## Ключевые особенности

### Многоагентный конвейер
- **На основе референсов (Reference-Driven)**: Обучается на тщательно подобранных примерах с помощью генеративного поиска.
- **Итеративное улучшение (Iterative Refinement)**: Цикл "Критик-Визуализатор" (Critic-Visualizer) для постепенного улучшения качества.
- **Учет стиля (Style-Aware)**: Автоматически синтезированные эстетические рекомендации обеспечивают академическое качество.
- **Гибкие режимы (Flexible Modes)**: Несколько режимов экспериментов для различных сценариев использования.

### Интерактивная демонстрация
- **Параллельная генерация**: Генерация до 20 диаграмм-кандидатов одновременно.
- **Визуализация конвейера**: Отслеживание эволюции на этапах Планировщик → Стилист → Критик (Planner → Stylist → Critic).
- **Улучшение до высокого разрешения**: Масштабирование (upscale) до 2K/4K с использованием API для генерации изображений.
- **Пакетный экспорт**: Скачивание всех кандидатов в формате PNG или в виде ZIP-архива.

### Расширяемый дизайн
- **Модульные агенты**: Каждый агент может быть настроен независимо.
- **Поддержка задач**: Обрабатывает как концептуальные диаграммы, так и графики данных.
- **Фреймворк для оценки (Evaluation Framework)**: Встроенная оценка по сравнению с истинными (ground truth) данными с использованием нескольких метрик.
- **Асинхронная обработка (Async Processing)**: Эффективная пакетная обработка с настраиваемым уровнем параллелизма.

## Поддержка сообщества
На момент выпуска этого репозитория мы заметили несколько попыток сообщества воспроизвести эту работу. Эти попытки привносят уникальные точки зрения, которые мы считаем невероятно ценными. Мы настоятельно рекомендуем ознакомиться с этими отличными материалами (будем рады, если вы добавите то, что мы упустили):
- https://github.com/llmsresearch/paperbanana
- https://github.com/efradeca/freepaperbanana
- https://github.com/elpsykongloo/PaperBanana-Pro — PaperBanana-Pro: постоянно обновляемая улучшенная версия на китайском языке с более стабильным конвейером и более удобным интерфейсом.

Кроме того, параллельно с разработкой этого метода во многих других работах исследовалась та же тема автоматической генерации академических иллюстраций — некоторые даже позволяют создавать редактируемые рисунки. Их вклад имеет важное значение для экосистемы и заслуживает вашего внимания (также будем рады дополнениям):
- https://github.com/ResearAI/AutoFigure-Edit
- https://github.com/OpenDCAI/Paper2Any
- https://github.com/BIT-DataLab/Edit-Banana

В целом, нас воодушевляет то, что базовые возможности современных моделей значительно приблизили нас к решению проблемы автоматизированной генерации академических иллюстраций. При постоянных усилиях сообщества мы верим, что в ближайшем будущем у нас появятся высококачественные инструменты для автоматического рисования, которые ускорят процесс академических исследований и визуальную коммуникацию.

Мы горячо приветствуем вклад сообщества, чтобы сделать PaperBanana еще лучше!

## Авторы (Contributors)

Спасибо всем участникам, которые помогли улучшить PaperBanana, будь то код, сообщения об ошибках, идеи или отзывы!

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/dwzhu-pku"><img src="https://github.com/dwzhu-pku.png?s=100" width="100px;" alt="Dawei Zhu"/><br /><sub><b>Dawei Zhu</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=dwzhu-pku" title="Code">💻</a> <a href="#ideas-dwzhu-pku" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=dwzhu-pku" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/lemon-prog123"><img src="https://github.com/lemon-prog123.png?s=100" width="100px;" alt="lemon-prog123"/><br /><sub><b>lemon-prog123</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=lemon-prog123" title="Code">💻</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/memray"><img src="https://github.com/memray.png?s=100" width="100px;" alt="memray"/><br /><sub><b>memray</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=memray" title="Code">💻</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/elpsykongloo"><img src="https://github.com/elpsykongloo.png?s=100" width="100px;" alt="elpsykongloo"/><br /><sub><b>elpsykongloo</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3Aelpsykongloo" title="Bug reports">🐛</a> <a href="#ideas-elpsykongloo" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/weathon"><img src="https://github.com/weathon.png?s=100" width="100px;" alt="weathon"/><br /><sub><b>weathon</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3Aweathon" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/arashabadi"><img src="https://github.com/arashabadi.png?s=100" width="100px;" alt="arashabadi"/><br /><sub><b>arashabadi</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=arashabadi" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/Ludobico"><img src="https://github.com/Ludobico.png?s=100" width="100px;" alt="Ludobico"/><br /><sub><b>Ludobico</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=Ludobico" title="Code">💻</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/haosenwang1018"><img src="https://github.com/haosenwang1018.png?s=100" width="100px;" alt="haosenwang1018"/><br /><sub><b>haosenwang1018</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=haosenwang1018" title="Code">💻</a> <a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3Ahaosenwang1018" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/stuinfla"><img src="https://github.com/stuinfla.png?s=100" width="100px;" alt="stuinfla"/><br /><sub><b>stuinfla</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=stuinfla" title="Code">💻</a> <a href="#ideas-stuinfla" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/ReturnYG"><img src="https://github.com/ReturnYG.png?s=100" width="100px;" alt="ReturnYG"/><br /><sub><b>ReturnYG</b></sub></a><br /><a href="#ideas-ReturnYG" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/Mylszd"><img src="https://github.com/Mylszd.png?s=100" width="100px;" alt="Mylszd"/><br /><sub><b>Mylszd</b></sub></a><br /><a href="#ideas-Mylszd" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-Mylszd" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/NielsRogge"><img src="https://github.com/NielsRogge.png?s=100" width="100px;" alt="NielsRogge"/><br /><sub><b>NielsRogge</b></sub></a><br /><a href="#ideas-NielsRogge" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/MinyuChan-vem"><img src="https://github.com/MinyuChan-vem.png?s=100" width="100px;" alt="MinyuChan-vem"/><br /><sub><b>MinyuChan-vem</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3AMinyuChan-vem" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/catallactics"><img src="https://avatars.githubusercontent.com/u/223395626?v=4?s=100" width="100px;" alt="catallactics"/><br /><sub><b>catallactics</b></sub></a><br /><a href="#ideas-catallactics" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/ruiguo-bio"><img src="https://avatars.githubusercontent.com/u/20548903?v=4?s=100" width="100px;" alt="Rui Guo"/><br /><sub><b>Rui Guo</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3Aruiguo-bio" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/YXDBright"><img src="https://avatars.githubusercontent.com/u/144319486?v=4?s=100" width="100px;" alt="YXDBright"/><br /><sub><b>YXDBright</b></sub></a><br /><a href="#ideas-YXDBright" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="16.66%"><a href="http://sites.google.com/view/yiming-shen"><img src="https://avatars.githubusercontent.com/u/89332218?v=4?s=100" width="100px;" alt="Yiming Shen"/><br /><sub><b>Yiming Shen</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=shenyimings" title="Code">💻</a></td>
      <td align="center" valign="top" width="16.66%"><a href="http://blog.sukisq.me"><img src="https://avatars.githubusercontent.com/u/87158944?v=4?s=100" width="100px;" alt="Edom"/><br /><sub><b>Edom</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=blessonism" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/issyuNaN"><img src="https://avatars.githubusercontent.com/u/167730146?v=4?s=100" width="100px;" alt="issyuNaN"/><br /><sub><b>issyuNaN</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3AissyuNaN" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="16.66%"><a href="http://tsingloong.xyz"><img src="https://avatars.githubusercontent.com/u/78492333?v=4?s=100" width="100px;" alt="Tsing_loong"/><br /><sub><b>Tsing_loong</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/commits?author=Tsingloong611" title="Code">💻</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/onepercentLI"><img src="https://avatars.githubusercontent.com/u/17852986?v=4?s=100" width="100px;" alt="LiJingfei"/><br /><sub><b>LiJingfei</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3AonepercentLI" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="16.66%"><a href="https://github.com/Konjac-XZ"><img src="https://avatars.githubusercontent.com/u/71483384?v=4?s=100" width="100px;" alt="Konjac-XZ"/><br /><sub><b>Konjac-XZ</b></sub></a><br /><a href="https://github.com/dwzhu-pku/PaperBanana/issues?q=author%3AKonjac-XZ" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Лицензия
Apache-2.0

## Цитирование
Если вы находите этот репозиторий полезным, пожалуйста, процитируйте нашу статью следующим образом:
```bibtex
@article{zhu2026paperbanana,
  title={PaperBanana: Automating Academic Illustration for AI Scientists},
  author={Zhu, Dawei and Meng, Rui and Song, Yale and Wei, Xiyu and Li, Sujian and Pfister, Tomas and Yoon, Jinsung},
  journal={arXiv preprint arXiv:2601.23265},
  year={2026}
}
```

## Отказ от ответственности (Disclaimer)
Это не официально поддерживаемый продукт Google. Этот проект не подпадает под действие программы [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

Наша цель — исключительно принести пользу сообществу, поэтому в настоящее время у нас нет планов использовать его в коммерческих целях. Основная методология была разработана во время моей стажировки в Google, и Google подал заявки на патенты на эти конкретные рабочие процессы. Хотя это не влияет на исследовательскую деятельность в области open-source, это ограничивает сторонние коммерческие приложения, использующие аналогичную логику.