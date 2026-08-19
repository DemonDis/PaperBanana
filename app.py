# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Веб-интерфейс на базе Gradio для PaperBanana.
Заменяет Streamlit demo.py современным тёмным интерфейсом.
"""

import gradio as gr
import asyncio
import base64
import json
import zipfile
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Логотип (base64-кодировка для надёжной загрузки в Gradio)
# ---------------------------------------------------------------------------
_logo_path = Path(__file__).parent / "assets" / "logo.jpg"
if _logo_path.exists():
    LOGO_B64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii")
else:
    LOGO_B64 = ""

# ---------------------------------------------------------------------------
# Импорты проекта (переиспользуем логику demo.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import shutil

configs_dir = Path(__file__).parent / "configs"
config_path = configs_dir / "model_config.yaml"
template_path = configs_dir / "model_config.template.yaml"

if not config_path.exists() and template_path.exists():
    shutil.copy2(template_path, config_path)

from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.vanilla_agent import VanillaAgent
from agents.polish_agent import PolishAgent
from utils import config
from utils.legacy_generation_options import (
    generation_additional_info,
    normalize_legacy_input_content,
)
from utils.legacy_ui_results import build_evolution_stages, resolve_final_output
from utils.paperviz_processor import PaperVizProcessor

model_config_data = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        model_config_data = yaml.safe_load(f) or {}


def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config_data:
        val = model_config_data[section].get(key)
    return val or default


# ---------------------------------------------------------------------------
# Переиспользование основных вспомогательных функций из demo.py
# ---------------------------------------------------------------------------

def clean_text(text):
    if not text:
        return text
    if isinstance(text, str):
        return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text


def base64_to_image(b64_str):
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        return Image.open(BytesIO(base64.b64decode(b64_str)))
    except Exception:
        return None


def create_sample_inputs(
    method_content,
    caption,
    aspect_ratio="16:9",
    figure_size=None,
    num_copies=10,
    max_critic_rounds=3,
    task_name="diagram",
):
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": normalize_legacy_input_content(method_content, task_name),
        "visual_intent": caption,
        "additional_info": generation_additional_info(aspect_ratio, figure_size),
        "max_critic_rounds": max_critic_rounds,
        "task_name": task_name,
    }
    inputs = []
    for i in range(num_copies):
        c = base_input.copy()
        c["filename"] = f"demo_input_candidate_{i}"
        c["candidate_id"] = i
        inputs.append(c)
    return inputs


async def process_parallel_candidates(
    data_list, exp_mode="dev_planner_critic", retrieval_setting="auto",
    main_model_name="", image_gen_model_name="", task_name="diagram",
):
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    exp_config = config.ExpConfig(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        main_model_name=main_model_name,
        image_gen_model_name=image_gen_model_name,
        work_dir=Path(__file__).parent,
    )
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )
    results = []
    async for result_data in processor.process_queries_batch(data_list, max_concurrent=10, do_eval=False):
        result_data["task_name"] = task_name
        results.append(result_data)
    return results


async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K"):
    image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", "")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Путь 1: OpenRouter
    try:
        from utils.generation_utils import call_openrouter_image_generation_with_retry_async
        _has_openrouter = True
    except ImportError:
        _has_openrouter = False
    openrouter_api_key = get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
    if _has_openrouter and openrouter_api_key:
        try:
            contents = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": edit_prompt},
            ]
            cfg = {"system_prompt": "", "temperature": 1.0, "aspect_ratio": aspect_ratio, "image_size": image_size}
            result = await call_openrouter_image_generation_with_retry_async(
                model_name=image_model, contents=contents, config=cfg, max_attempts=3, retry_delay=10, error_context="refine_image",
            )
            if result and result[0] != "Error":
                return base64.b64decode(result[0]), "Image refined successfully! (via OpenRouter)"
        except Exception as e:
            print(f"OpenRouter refine failed: {e}, falling back...")

    # Пути 2 и 3: Нативный SDK Gemini
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "Error: google-genai SDK not installed and OpenRouter unavailable."

    google_api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
    project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")

    if google_api_key:
        client = genai.Client(api_key=google_api_key)
        via = "Google API key"
    elif project_id:
        location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
        client = genai.Client(vertexai=True, project=project_id, location=location)
        via = "Vertex AI"
    else:
        return None, "Error: No API credentials configured."

    try:
        contents = [
            types.Part.from_text(text=edit_prompt),
            types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes),
        ]
        gen_config = types.GenerateContentConfig(
            temperature=1.0, max_output_tokens=8192, response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
        )
        response = await asyncio.to_thread(
            client.models.generate_content, model=image_model, contents=contents, config=gen_config,
        )
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    if isinstance(data, bytes):
                        return data, f"Image refined successfully! (via {via})"
                    elif isinstance(data, str):
                        return base64.b64decode(data), f"Image refined successfully! (via {via})"
        return None, f"No image data found in {via} response"
    except Exception as e:
        return None, f"{via} error: {str(e)}"


def get_evolution_stages(result, exp_mode):
    return build_evolution_stages(result, exp_mode=exp_mode)


def get_final_image(result, exp_mode):
    """Возвращает (PIL.Image, текст_описания) для лучшего доступного этапа."""
    selection = resolve_final_output(result, exp_mode=exp_mode)
    img = base64_to_image(result.get(selection.image_key)) if selection.image_key else None
    desc = clean_text(result.get(selection.text_key, "")) if selection.text_key else ""
    return img, desc


# ---------------------------------------------------------------------------
# Примеры содержимого
# ---------------------------------------------------------------------------

EXAMPLE_METHOD = r"""## Методология: Фреймворк PaperBanana

В этом разделе мы представляем архитектуру PaperBanana — мультиагентной системы, управляемой эталонами, для автоматической академической иллюстрации. Как показано на рисунке \ref{fig:methodology_diagram}, PaperBanana координирует команду из пяти специализированных агентов — Retriever, Planner, Stylist, Visualizer и Critic — для преобразования сырых научных данных в диаграммы и графики уровня публикации. (См. приложение \ref{app_sec:agent_prompts} для промптов)

### Агент Retriever

Задав исходный контекст $S$ и коммуникативное намерение $C$, агент Retriever определяет $N$ наиболее релевантных примеров $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ из фиксированного набора эталонов $\mathcal{R}$ для направления последующих агентов. Как определено в разделе \ref{sec:task_formulation}, каждый пример $E_i \in \mathcal{R}$ является тройкой $(S_i, C_i, I_i)$.
Для использования возможностей рассуждений VLM мы применяем подход генеративного извлечения, при котором VLM выполняет выбор по метаданным кандидатов:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$

### Агент Planner

Агент Planner служит когнитивным ядром системы. Он принимает исходный контекст $S$, коммуникативное намерение $C$ и извлечённые примеры $\mathcal{E}$ в качестве входных данных:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Агент Stylist

Агент Stylist уточняет каждое начальное описание $P$ до стилистически оптимизированной версии $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$

### Агент Visualizer

Агент Visualizer использует модель генерации изображений:
$$
I_t = \text{Image-Gen}(P_t)
$$

### Агент Critic

Агент Critic предоставляет целевую обратную связь и формирует уточнённое описание:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
Цикл Visualizer-Critic выполняется $T=3$ итерации."""

EXAMPLE_CAPTION = "Рисунок 1: Обзор нашего фреймворка PaperBanana. Задав исходный контекст и коммуникативное намерение, мы сначала применяем Фазу линейного планирования для извлечения релевантных эталонных примеров и синтеза стилистически оптимизированного описания. Затем мы используем Итеративный цикл уточнения (состоящий из агентов Visualizer и Critic) для преобразования описания в визуальный вывод и проведения многораундового уточнения для создания финальной академической иллюстрации."

PIPELINE_DESCRIPTIONS = {
    "demo_planner_critic": "Retriever → Planner → Visualizer → Critic → Visualizer (без Stylist)",
    "demo_full": "Retriever → Planner → Stylist → Visualizer → Critic → Visualizer",
}

# ---------------------------------------------------------------------------
# Пользовательский CSS для тёмной темы
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ---- Global ---- */
.gradio-container {
    max-width: 1400px !important;
    width: 100% !important;
    margin: 0 auto !important;
}
.gradio-container > .main {
    max-width: 100% !important;
}

/* ---- Accent colour (orange/amber) ---- */
.accent { color: #f59e0b; }
.orange-btn {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    border-radius: 10px !important;
}
.orange-btn:hover {
    background: linear-gradient(135deg, #d97706, #b45309) !important;
}

/* ---- Section labels ---- */
.section-label {
    text-transform: uppercase;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1.5px;
    color: #f59e0b;
    margin-bottom: 8px;
}

/* ---- Card-like blocks ---- */
.settings-panel, .input-panel, .results-panel {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
}

/* ---- Candidate gallery (orange border) ---- */
.candidate-card {
    border: 2px solid #f59e0b;
    border-radius: 12px;
    padding: 8px;
    text-align: center;
}

/* ---- Footer ---- */
#footer-row {
    text-align: center;
    padding: 12px 0;
    font-size: 13px;
    color: #6b7280;
}
#footer-row a { color: #f59e0b; text-decoration: none; }
#footer-row a:hover { text-decoration: underline; }

/* ---- Evolution timeline ---- */
.evo-stage { margin-bottom: 12px; }
.evo-stage-title { font-weight: 600; color: #f59e0b; }

/* ---- Status ---- */
.status-box {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 16px;
    background: #f9fafb;
    font-size: 14px;
}

/* ---- Left settings column: prevent label truncation ---- */
.left-settings { min-width: 320px; }
.left-settings .gr-block label,
.left-settings .gr-input label,
.left-settings label span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
.left-settings .gradio-dropdown,
.left-settings .gradio-textbox,
.left-settings .gradio-slider,
.left-settings .gradio-number {
    min-width: 0 !important;
}

/* ---- Compact info text ---- */
.gradio-dropdown .wrap .info,
.gradio-textbox .wrap .info { font-size: 0.8em !important; }

/* ---- Header button style (outlined) ---- */
.header-link-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 20px;
    border: 1.5px solid #d1d5db;
    background: #fff;
    color: #374151;
    font-weight: 600;
    font-size: 14px;
    text-decoration: none;
    transition: border-color 0.2s, background 0.2s;
}
.header-link-btn:hover {
    border-color: #f59e0b;
    background: #fffbeb;
    text-decoration: none;
    color: #374151;
}
"""

# ---------------------------------------------------------------------------
# Построение интерфейса Gradio Blocks
# ---------------------------------------------------------------------------

def build_app():

    default_main_model = get_config_val("defaults", "main_model_name", "MAIN_MODEL_NAME", "gemini-3.1-pro-preview")
    default_image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", "gemini-3.1-flash-image-preview")

    with gr.Blocks(title="PaperBanana") as app:
        # ---- State to hold results across interactions ----
        gen_results_state = gr.State([])
        gen_mode_state = gr.State("demo_planner_critic")
        gen_timestamp_state = gr.State("")
        gen_json_path_state = gr.State("")

        # ================================================================
        # HEADER
        # ================================================================
        gr.HTML(f"""
        <div style="background: #fff; border-radius: 16px; padding: 24px 36px; margin-bottom: 16px; width: 100%;
                    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
                    border: 1px solid #e5e7eb;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <img src="data:image/jpeg;base64,{LOGO_B64}" alt="PaperBanana logo"
                     style="height: 60px; width: auto; border-radius: 10px; object-fit: contain;" />
                <div>
                    <p style="font-size: 28px; font-weight: 800; color: #111; margin: 0 0 4px 0;">
                        PaperBanana
                    </p>
                </div>
            </div>
        </div>
        """)

        # ================================================================
        # API KEYS ACCORDION
        # ================================================================
        with gr.Accordion("Ключи API", open=False):
            gr.Markdown(
                "**Оба ключа не нужны.** Заполните **хотя бы один**: **OpenRouter** *или* **Google (Gemini)**. "
                "Если заданы оба, OpenRouter используется приоритетно для автоматической маршрутизации."
            )
            with gr.Row():
                openrouter_key_input = gr.Textbox(
                    label="Ключ OpenRouter API (необязательно)", type="password", placeholder="sk-or-...",
                    value=get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", ""),
                )
                google_key_input = gr.Textbox(
                    label="Ключ Google API (необязательно)", type="password", placeholder="AIza...",
                    value=get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", ""),
                )
            gr.Markdown("*Ключи используются только для текущей сессии и не сохраняются.*")

            def apply_keys(or_key, g_key):
                if or_key:
                    os.environ["OPENROUTER_API_KEY"] = or_key
                if g_key:
                    os.environ["GOOGLE_API_KEY"] = g_key
                from utils.generation_utils import reinitialize_clients
                initialized = reinitialize_clients()
                if initialized:
                    return f"Клиенты инициализированы: {', '.join(initialized)}."
                return (
                    "Внимание: не удалось инициализировать ни одного API-клиента. "
                    "Введите хотя бы один ключ — OpenRouter или Google (Gemini)."
                )

            apply_keys_btn = gr.Button("Применить ключи", size="sm")
            keys_status = gr.Textbox(visible=False)
            apply_keys_btn.click(apply_keys, inputs=[openrouter_key_input, google_key_input], outputs=[keys_status])

        # ================================================================
        # TABS
        # ================================================================
        with gr.Tabs():

            # ============================================================
            # TAB 1 — Generate Candidates
            # ============================================================
            with gr.TabItem("Генерация кандидатов"):
                with gr.Row():
                    # ---------- LEFT COLUMN: SETTINGS ----------
                    with gr.Column(scale=1, min_width=280, elem_classes=["left-settings"]):
                        gr.HTML('<p class="section-label">Настройки</p>')

                        pipeline_mode = gr.Dropdown(
                            choices=["demo_planner_critic", "demo_full"],
                            value="demo_full",
                            label="Режим конвейера",
                            info="Выберите конвейер агентов",
                        )
                        task_name = gr.Dropdown(
                            choices=["diagram", "plot"],
                            value="diagram",
                            label="Тип вывода",
                            info="Генерировать научную диаграмму или статистический график",
                        )
                        pipeline_desc = gr.Textbox(
                            label="Описание конвейера",
                            value=PIPELINE_DESCRIPTIONS["demo_full"],
                            interactive=False, lines=2,
                        )
                        pipeline_mode.change(
                            lambda m: PIPELINE_DESCRIPTIONS.get(m, ""),
                            inputs=[pipeline_mode],
                            outputs=[pipeline_desc],
                        )

                        retrieval_setting = gr.Dropdown(
                            choices=["auto", "manual", "random", "none"],
                            value="auto",
                            label="Настройка извлечения",
                            info="Способ извлечения эталонных примеров",
                        )
                        num_candidates = gr.Number(
                            value=10, minimum=1, maximum=20, step=1,
                            label="Количество кандидатов",
                        )
                        aspect_ratio = gr.Dropdown(
                            choices=["16:9", "21:9", "3:2"],
                            value="21:9",
                            label="Соотношение сторон",
                        )
                        figure_size = gr.Dropdown(
                            choices=["1-3cm", "4-6cm", "7-9cm", "10-13cm", "14-17cm"],
                            value="7-9cm",
                            label="Размер фигуры",
                        )
                        max_critic_rounds = gr.Slider(
                            minimum=1, maximum=5, value=3, step=1,
                            label="Макс. раундов критика",
                        )
                        main_model_name = gr.Textbox(
                            label="Имя модели",
                            info="Модель для рассуждений",
                            value=default_main_model,
                        )
                        image_model_name = gr.Textbox(
                            label="Модель генерации изображений",
                            info="Модель для генерации изображений диаграмм",
                            value=default_image_model,
                        )
                        save_results = gr.Dropdown(
                            choices=["Yes", "No"],
                            value="Yes",
                            label="Сохранять результаты",
                        )

                    # ---------- RIGHT COLUMN: INPUT + OUTPUT ----------
                    with gr.Column(scale=3):
                        gr.HTML('<p class="section-label">Ввод</p>')

                        with gr.Row():
                            method_example = gr.Dropdown(
                                choices=["None", "PaperBanana Framework"],
                                value="PaperBanana Framework",
                                label="Загрузить пример (Метод)",
                            )
                            caption_example = gr.Dropdown(
                                choices=["None", "PaperBanana Framework"],
                                value="PaperBanana Framework",
                                label="Загрузить пример (Подпись)",
                            )

                        with gr.Row():
                            method_content = gr.Textbox(
                                label="Содержание метода / Данные графика",
                                value=EXAMPLE_METHOD,
                                lines=12, max_lines=30,
                            )
                            caption_input = gr.Textbox(
                                label="Подпись к фигуре / Визуальное намерение",
                                value=EXAMPLE_CAPTION,
                                lines=12, max_lines=30,
                            )

                        # Wire example selectors
                        def load_method_example(choice):
                            return EXAMPLE_METHOD if choice == "PaperBanana Framework" else ""
                        def load_caption_example(choice):
                            return EXAMPLE_CAPTION if choice == "PaperBanana Framework" else ""

                        method_example.change(load_method_example, inputs=[method_example], outputs=[method_content])
                        caption_example.change(load_caption_example, inputs=[caption_example], outputs=[caption_input])

                        generate_btn = gr.Button(
                            "✨ Сгенерировать кандидатов", variant="primary",
                            elem_classes=["orange-btn"], size="lg",
                        )

                # ---- Status ----
                status_text = gr.Textbox(label="Статус", interactive=False, lines=1)

                # ---- Результаты ----
                gr.HTML('<p class="section-label" style="margin-top:16px;">Сгенерированные кандидаты</p>')
                results_gallery = gr.Gallery(
                    label="Сгенерированные кандидаты",
                    columns=3, height="auto", object_fit="contain",
                )
                with gr.Accordion("Таймлайн эволюции", open=False):
                    evolution_html = gr.HTML("")
                with gr.Accordion("Скачать всё (ZIP)", open=False):
                    zip_file_output = gr.File(label="Скачивание ZIP")

                # ---- Generate handler ----
                def run_generate(
                    method_text, caption_text, pipe_mode, task_name, ret_setting,
                    n_cands, ar, max_rounds, m_model, img_model,
                    figure_size, save_results,
                    progress=gr.Progress(track_tqdm=True),
                ):
                    if not method_text or not caption_text:
                        raise gr.Error("Укажите содержание метода и подпись.")

                    n_cands = int(n_cands)
                    max_rounds = int(max_rounds)
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                    progress(0, desc="Подготовка входных данных...")
                    input_data = create_sample_inputs(
                        method_content=method_text, caption=caption_text,
                        aspect_ratio=ar, figure_size=figure_size,
                        num_copies=n_cands, max_critic_rounds=max_rounds,
                        task_name=task_name,
                    )

                    progress(0.1, desc=f"Параллельная генерация {n_cands} кандидатов...")
                    try:
                        loop = asyncio.new_event_loop()
                        results = loop.run_until_complete(
                            process_parallel_candidates(
                                input_data, exp_mode=pipe_mode, retrieval_setting=ret_setting,
                                main_model_name=m_model, image_gen_model_name=img_model,
                                task_name=task_name,
                            )
                        )
                        loop.close()
                    except Exception as e:
                        raise gr.Error(f"Ошибка генерации: {e}")

                    progress(0.9, desc="Сохранение результатов...")

                    # Save JSON
                    results_dir = Path(__file__).parent / "results" / "demo"
                    results_dir.mkdir(parents=True, exist_ok=True)
                    json_filename = results_dir / f"demo_{timestamp_str}.json"
                    try:
                        with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                            s = json.dumps(results, ensure_ascii=False, indent=4)
                            s = s.encode("utf-8", "ignore").decode("utf-8")
                            f.write(s)
                    except Exception:
                        json_filename = None

                    # Build gallery images
                    gallery_images = []
                    for idx, res in enumerate(results):
                        img, _ = get_final_image(res, pipe_mode)
                        if img:
                            gallery_images.append((img, f"Кандидат {idx}"))

                    # Build evolution HTML
                    evo_parts = []
                    for idx, res in enumerate(results):
                        stages = get_evolution_stages(res, pipe_mode)
                        if stages:
                            evo_parts.append(f"<h4>Кандидат {idx} ({len(stages)} этапов)</h4>")
                            for st in stages:
                                evo_parts.append(f'<span class="evo-stage-title">{st["name"]}</span>: {st["description"]}<br/>')
                    evo_html = "".join(evo_parts) if evo_parts else "<p>Данные эволюции отсутствуют.</p>"

                    # Build ZIP
                    zip_path = None
                    if save_results != "No":
                        try:
                            zip_filename = results_dir / f"papervizagent_candidates_{timestamp_str}.zip"
                            buf = BytesIO()
                            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                                for idx, res in enumerate(results):
                                    img, _ = get_final_image(res, pipe_mode)
                                    if img:
                                        ib = BytesIO()
                                        img.save(ib, format="PNG")
                                        zf.writestr(f"candidate_{idx}.png", ib.getvalue())
                            buf.seek(0)
                            with open(zip_filename, "wb") as wf:
                                wf.write(buf.getvalue())
                            zip_path = str(zip_filename)
                        except Exception:
                            pass

                    status = f"Сгенерировано {len(results)} кандидатов в {datetime.now().strftime('%H:%M:%S')}."
                    if json_filename and Path(str(json_filename)).exists():
                        status += f" JSON сохранён: {Path(str(json_filename)).name}."

                    progress(1.0, desc="Готово!")
                    return (
                        gallery_images,       # results_gallery
                        evo_html,             # evolution_html
                        zip_path,             # zip_file_output
                        status,               # status_text
                        results,              # gen_results_state
                        pipe_mode,            # gen_mode_state
                        timestamp_str,        # gen_timestamp_state
                    )

                generate_btn.click(
                    fn=run_generate,
                    inputs=[
                        method_content, caption_input, pipeline_mode, task_name, retrieval_setting,
                        num_candidates, aspect_ratio, max_critic_rounds,
                        main_model_name, image_model_name,
                        figure_size, save_results,
                    ],
                    outputs=[
                        results_gallery, evolution_html, zip_file_output, status_text,
                        gen_results_state, gen_mode_state, gen_timestamp_state,
                    ],
                )

            # ============================================================
            # TAB 2 — Refine Image
            # ============================================================
            with gr.TabItem("Уточнение изображения"):
                gr.Markdown("### Уточните и увеличьте разрешение диаграммы до 2K/4K")
                gr.Markdown("Загрузите изображение, опишите изменения и получите версию в высоком разрешении.")

                with gr.Row():
                    with gr.Column():
                        refine_upload = gr.Image(label="Загрузить изображение", type="pil", height=400)
                    with gr.Column():
                        refine_prompt = gr.Textbox(
                            label="Инструкции по редактированию", lines=6,
                            placeholder="Например, «Измените цветовую схему под стиль академической статьи» или «Оставьте всё как есть, но увеличьте разрешение»",
                        )
                        with gr.Row():
                            refine_resolution = gr.Dropdown(choices=["2K", "4K"], value="2K", label="Разрешение")
                            refine_aspect = gr.Dropdown(choices=["21:9", "16:9", "3:2"], value="21:9", label="Соотношение сторон")
                        refine_btn = gr.Button("Уточнить изображение", variant="primary", elem_classes=["orange-btn"])

                refine_status = gr.Textbox(label="Статус", interactive=False)

                with gr.Row():
                    refine_before = gr.Image(label="До", interactive=False, height=400)
                    refine_after = gr.Image(label="После", interactive=False, height=400)
                refine_download = gr.File(label="Скачать уточнённое изображение")

                def run_refine(pil_img, prompt, resolution, ar):
                    if pil_img is None:
                        raise gr.Error("Сначала загрузите изображение.")
                    if not prompt:
                        raise gr.Error("Укажите инструкции по редактированию.")

                    buf = BytesIO()
                    pil_img.save(buf, format="JPEG")
                    image_bytes = buf.getvalue()

                    loop = asyncio.new_event_loop()
                    try:
                        refined_bytes, msg = loop.run_until_complete(
                            refine_image_with_nanoviz(image_bytes, prompt, aspect_ratio=ar, image_size=resolution)
                        )
                    except Exception as e:
                        raise gr.Error(f"Ошибка уточнения: {e}")
                    finally:
                        loop.close()

                    if not refined_bytes:
                        raise gr.Error(msg)

                    refined_img = Image.open(BytesIO(refined_bytes))

                    # Save to temp file for download
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = Path(__file__).parent / "results" / "demo"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"refined_{resolution}_{ts}.png"
                    refined_img.save(str(out_path), format="PNG")

                    return pil_img, refined_img, str(out_path), msg

                refine_btn.click(
                    fn=run_refine,
                    inputs=[refine_upload, refine_prompt, refine_resolution, refine_aspect],
                    outputs=[refine_before, refine_after, refine_download, refine_status],
                )

        # ================================================================
        # FOOTER
        # ================================================================
        gr.HTML("""
        <div id="footer-row">
            <a href="https://github.com/dwzhu-pku/PaperBanana" target="_blank">GitHub</a> &middot;
            <a href="https://arxiv.org/abs/2601.23265" target="_blank">Статья</a><br/>
            PaperBanana &copy; 2026
        </div>
        """)

    return app


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.amber,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
    )
