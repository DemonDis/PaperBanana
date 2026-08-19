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
Параллельный Streamlit Demo для PaperBanana.
Принимает текстовый ввод пользователя, дублирует его 10 раз и запускает параллельную обработку
для генерации нескольких кандидатов диаграмм для сравнения.
"""

import streamlit as st
import asyncio
import base64
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))

print("DEBUG: Importing agents...")
import yaml
import shutil
configs_dir = Path(__file__).parent / "configs"
config_path = configs_dir / "model_config.yaml"
template_path = configs_dir / "model_config.template.yaml"

if not config_path.exists() and template_path.exists():
    print(f"DEBUG: {config_path.name} not found. Auto-generating from template")
    shutil.copy2(template_path, config_path)
try:
    from agents.planner_agent import PlannerAgent
    print("DEBUG: Imported PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("DEBUG: Imported all agents")
    from utils import config
    from utils.legacy_generation_options import (
        generation_additional_info,
        normalize_legacy_input_content,
    )
    from utils.legacy_ui_results import build_evolution_stages, resolve_final_output
    from utils.paperviz_processor import PaperVizProcessor
    print("DEBUG: Imported utils")

    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"DEBUG: Exception during import: {e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperBanana Параллельный Demo",
    page_icon="🍌"
)

def clean_text(text):
    """Очищает текст, удаляя недопустимые символы-суррогаты UTF-8."""
    if not text:
        return text
    if isinstance(text, str):
        # Remove surrogate characters that cause UnicodeEncodeError
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """Конвертирует строку base64 в PIL Image."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def create_sample_inputs(
    method_content,
    caption,
    diagram_type="Pipeline",
    aspect_ratio="16:9",
    figure_size=None,
    num_copies=10,
    max_critic_rounds=3,
    task_name="diagram",
):
    """Создаёт несколько копий входных данных для параллельной обработки."""
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": normalize_legacy_input_content(method_content, task_name),
        "visual_intent": caption,
        "additional_info": generation_additional_info(aspect_ratio, figure_size),
        "max_critic_rounds": max_critic_rounds,  # Управление количеством раундов критика
        "task_name": task_name,
    }
    
    # Создаём num_copies одинаковых входов, каждый с уникальным идентификатором
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)
    
    return inputs

async def process_parallel_candidates(
    data_list,
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    main_model_name="",
    image_gen_model_name="",
    task_name="diagram",
):
    """Обрабатывает несколько кандидатов параллельно с помощью PaperVizProcessor."""
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    # Создаём конфигурацию эксперимента
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
    
    # Инициализируем процессор со всеми агентами
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
    
    # Обрабатываем всех кандидатов параллельно (параллелизм контролируется процессором)
    results = []
    concurrent_num = 10  # Process all 10 in parallel
    
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=concurrent_num, do_eval=False
    ):
        result_data["task_name"] = task_name
        results.append(result_data)
    
    return results

async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K"):
    """
    Уточняет изображение с помощью API редактирования изображений.
    Поддерживает OpenRouter (приоритет), Google API key и Vertex AI ADC как запасной вариант.
    
    Args:
        image_bytes: Данные изображения в байтах
        edit_prompt: Текстовое описание желаемых изменений
        aspect_ratio: Выходное соотношение сторон (21:9, 16:9, 3:2)
        image_size: Выходное разрешение (2K или 4K)
    
    Returns:
        Кортеж (байты_отредактированного_изображения, сообщение_об_успехе)
    """
    image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", "")

    # Кодируем изображение как data URL base64 для OpenRouter
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_b64}"

    # --- Путь 1: OpenRouter (приоритет, как в основном конвейере) ---
    try:
        from utils.generation_utils import call_openrouter_image_generation_with_retry_async
        _has_openrouter = True
    except ImportError:
        _has_openrouter = False
    openrouter_api_key = get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
    if _has_openrouter and openrouter_api_key:
        try:
            contents = [
                {"type": "image", "data": image_b64, "mime_type": "image/jpeg"},
                {"type": "text", "text": edit_prompt},
            ]
            config = {
                "system_prompt": "",
                "temperature": 1.0,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
            }
            result = await call_openrouter_image_generation_with_retry_async(
                model_name=image_model,
                contents=contents,
                config=config,
                max_attempts=3,
                retry_delay=10,
                error_context="refine_image",
            )
            if result and result[0] != "Error":
                return base64.b64decode(result[0]), "✅ Image refined successfully! (via OpenRouter)"
        except Exception as e:
            print(f"OpenRouter refine failed: {e}, falling back to Google API key...")

    # --- Пути 2 и 3: Нативный SDK Gemini (Google API key или Vertex AI ADC) ---
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "❌ Error: google-genai SDK not installed and OpenRouter unavailable."

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
        return None, "❌ Error: No API credentials configured. Set OPENROUTER_API_KEY, GOOGLE_API_KEY, or configure Vertex AI project in configs/model_config.yaml."

    try:
        contents = [
            types.Part.from_text(text=edit_prompt),
            types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes),
        ]
        gen_config = types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=8192,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        )
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=image_model,
            contents=contents,
            config=gen_config,
        )
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    edited_image_data = part.inline_data.data
                    if isinstance(edited_image_data, bytes):
                        return edited_image_data, f"✅ Image refined successfully! (via {via})"
                    elif isinstance(edited_image_data, str):
                        return base64.b64decode(edited_image_data), f"✅ Image refined successfully! (via {via})"
        return None, f"❌ No image data found in {via} response"
    except Exception as e:
        return None, f"❌ {via} error: {str(e)}"


def get_evolution_stages(result, exp_mode):
    """Извлекает все этапы эволюции (изображения и описания) из результата."""
    return build_evolution_stages(result, exp_mode=exp_mode)

def display_candidate_result(result, candidate_id, exp_mode):
    """Отображает результат одного кандидата."""
    selection = resolve_final_output(result, exp_mode=exp_mode)
    final_image_key = selection.image_key
    final_desc_key = selection.text_key
    
    # Отображаем финальное изображение
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(img, use_container_width=True, caption=f"Кандидат {candidate_id} (Финальный)")
            
            # Добавляем кнопку скачивания
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ Скачать",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                use_container_width=True
            )
        else:
            st.error(f"Не удалось декодировать изображение для кандидата {candidate_id}")
    else:
        st.warning(f"Изображение не сгенерировано для кандидата {candidate_id}")
    
    # Показываем таймлайн эволюции в раскрывающемся блоке
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
                with st.expander(f"🔄 Таймлайн эволюции ({len(stages)} этапов)", expanded=False):
                    st.caption("Как диаграмма эволюционировала на разных этапах конвейера")
            
            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])
                
                # Отображаем изображение для этого этапа
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, use_container_width=True)
                
                # Показываем описание
                if stage['desc_key'] in result:
                    with st.expander(f"📝 Описание", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)
                
                # Показываем предложения критика, если есть
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 Предложения критика", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ Изменения не требуются — итерация остановлена.")
                        else:
                            st.write(cleaned_sugg)
                
                # Добавляем разделитель между этапами (кроме последнего)
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # Если только один этап, показываем описание в простом раскрывающемся блоке
        with st.expander(f"📝 Описание", expanded=False):
            if final_desc_key and final_desc_key in result:
                # Очищаем текст от недопустимых символов UTF-8
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("Описание отсутствует")

def main():
    st.title("🍌 PaperBanana Demo")
    st.markdown("Генерация и уточнение научных диаграмм с помощью ИИ")
    
    # Создаём вкладки
    tab1, tab2 = st.tabs(["📊 Генерация кандидатов", "✨ Уточнение изображения"])
    
    # ==================== ВКЛАДКА 1: Генерация кандидатов ====================
    with tab1:
        st.markdown("### Генерация нескольких кандидатов диаграмм из раздела метода и подписи")
        
        # Конфигурация боковой панели для вкладки 1
        with st.sidebar:
            st.title("⚙️ Настройки генерации")
            
            exp_mode = st.selectbox(
                "Pipeline Mode",
                ["demo_full", "demo_planner_critic"],
                index=0,
                key="tab1_exp_mode",
                help="Выберите конвейер агентов"
            )
            
            mode_info = {
                "demo_planner_critic": "Retriever → Planner → Visualizer → Critic → Visualizer (без Stylist)",
                "demo_full": "Retriever → Planner → Stylist → Visualizer → Critic → Visualizer. (Stylist может сделать диаграмму более эстетичной, но склонен к чрезмерному упрощению. Рекомендуем попробовать оба режима и выбрать лучший)"
            }
            st.info(f"**Конвейер:** {mode_info[exp_mode]}")
            
            retrieval_setting = st.selectbox(
                "Retrieval Setting",
                ["auto", "manual", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="Способ извлечения эталонных примеров: auto (автоматический выбор), manual (указанные эталоны), random (случайный выбор), none (без извлечения)"
            )

            task_name = st.selectbox(
                "Output Type",
                ["diagram", "plot"],
                index=0,
                key="tab1_task_name",
                help="Генерировать научную диаграмму или статистический график",
            )
            
            num_candidates = st.number_input(
                "Number of Candidates",
                min_value=1,
                max_value=20,
                value=10,
                key="tab1_num_candidates",
                help="Сколько параллельных кандидатов генерировать"
            )
            
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ["21:9", "16:9", "3:2"],
                key="tab1_aspect_ratio",
                help="Соотношение сторон для генерируемых диаграмм"
            )

            figure_size = st.selectbox(
                "Figure Size",
                ["1-3cm", "4-6cm", "7-9cm", "10-13cm", "14-17cm"],
                index=2,
                key="tab1_figure_size",
                help="Целевой размер фигуры; провайдеры с поддержкой image_size сопоставляют это с 1k, 2k или 4k",
            )
            
            max_critic_rounds = st.number_input(
                "Max Critic Rounds",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="Максимальное количество итераций уточнения критиком"
            )
            
            default_model = get_config_val("defaults", "main_model_name", "MAIN_MODEL_NAME", "gemini-3.1-pro-preview")
            text_model_presets = [default_model] if default_model else ["gemini-3.1-pro-preview"]
            if "gemini-3-flash-preview" not in text_model_presets:
                text_model_presets.append("gemini-3-flash-preview")
            if "gemini-3.1-pro-preview" not in text_model_presets:
                text_model_presets.insert(0, "gemini-3.1-pro-preview")
            text_model_presets.append("Custom")
            text_model_selection = st.selectbox(
                "Имя основной модели",
                text_model_presets,
                index=0,
                key="tab1_model_name",
                help="Модель языка и зрения для понимания и описания диаграмм"
            )
            if text_model_selection == "Custom":
                main_model_name = st.text_input(
                    "Пользовательская основная модель",
                    value="",
                    key="tab1_main_model_name_custom",
                    placeholder="например, openrouter/google/gemini-3.1-pro"
                )
            else:
                main_model_name = text_model_selection

            default_image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", "gemini-3.1-flash-image-preview")
            image_model_presets = [default_image_model] if default_image_model else ["gemini-3.1-flash-image-preview"]
            if "gemini-3-pro-image-preview" not in image_model_presets:
                image_model_presets.append("gemini-3-pro-image-preview")
            if "gemini-3.1-flash-image-preview" not in image_model_presets:
                image_model_presets.insert(0, "gemini-3.1-flash-image-preview")
            image_model_presets.append("Custom")
            image_model_selection = st.selectbox(
                "Имя модели генерации изображений",
                image_model_presets,
                index=0,
                key="tab1_image_model_name",
                help="Модель для генерации изображений диаграмм"
            )
            if image_model_selection == "Custom":
                image_gen_model_name = st.text_input(
                    "Пользовательская модель генерации изображений",
                    value="",
                    key="tab1_image_gen_model_name_custom",
                    placeholder="например, openrouter/openai/gpt-image-1"
                )
            else:
                image_gen_model_name = image_model_selection
        
        st.divider()
        
        # Секция ввода
        st.markdown("## 📝 Ввод")
        
        # Примеры содержимого
        example_method = r"""## Методология: Фреймворк PaperBanana
        
        В этом разделе мы представляем архитектуру PaperBanana — мультиагентной системы, управляемой эталонами, для автоматической академической иллюстрации. Как показано на рисунке \ref{fig:methodology_diagram}, PaperBanana координирует команду из пяти специализированных агентов — Retriever, Planner, Stylist, Visualizer и Critic — для преобразования сырых научных данных в диаграммы и графики уровня публикации. (См. приложение \ref{app_sec:agent_prompts} для промптов)

### Агент Retriever

Задав исходный контекст $S$ и коммуникативное намерение $C$, агент Retriever определяет $N$ наиболее релевантных примеров $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ из фиксированного набора эталонов $\mathcal{R}$ для направления последующих агентов. Как определено в разделе \ref{sec:task_formulation}, каждый пример $E_i \in \mathcal{R}$ является тройкой $(S_i, C_i, I_i)$.
Для использования возможностей рассуждений VLM мы применяем подход генеративного извлечения, при котором VLM выполняет выбор по метаданным кандидатов:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Конкретно, VLM ранжирует кандидатов по совпадению как исследовательского домена (например, Агенты и Рассуждения), так и типа диаграммы (например, конвейер, архитектура), при этом визуальная структура приоритетнее тематического сходства. Благодаря явному обоснованному выбору эталонных иллюстраций $I_i$, чьи соответствующие контексты $(S_i, C_i)$ лучше всего соответствуют текущим требованиям, Retriever обеспечивает конкретную основу как для структурной логики, так и для визуального стиля.

### Агент Planner

Агент Planner служит когнитивным ядром системы. Он принимает исходный контекст $S$, коммуникативное намерение $C$ и извлечённые примеры $\mathcal{E}$ в качестве входных данных. Выполняя обучение по контексту на демонстрациях из $\mathcal{E}$, Planner преобразует неструктурированные или структурированные данные из $S$ в подробное и исчерпывающее текстовое описание $P$ целевой иллюстрации:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Агент Stylist

Для обеспечения соответствия выходных данных эстетическим стандартам современных академических рукописей агент Stylist выступает в роли дизайн-консультанта.
Основная сложность заключается в определении всеобъемлющего «академического стиля», поскольку ручные определения часто бывают неполными.
Для решения этой проблемы Stylist обходит всю коллекцию эталонов $\mathcal{R}$ для автоматического синтеза *Эстетических рекомендаций* $\mathcal{G}$, охватывающих ключевые измерения: палитра цветов, формы и контейнеры, линии и стрелки, компоновка и композиция, а также типографика и иконки (см. приложение \ref{app_sec:auto_summarized_style_guide} для сводки рекомендаций и деталей реализации). Используя эти рекомендации, Stylist уточняет каждое начальное описание $P$ до стилистически оптимизированной версии $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
Это гарантирует, что финальная иллюстрация будет не только точной, но и визуально профессиональной.

### Агент Visualizer

Получив стилистически оптимизированное описание $P^*$, агент Visualizer совместно с агентом Critic рендерит академические иллюстрации и итеративно уточняет их качество. Агент Visualizer использует модель генерации изображений для преобразования текстовых описаний в визуальный вывод. На каждой итерации $t$, задав описание $P_t$, Visualizer генерирует:
$$
I_t = \text{Image-Gen}(P_t)
$$
где начальное описание $P_0$ установлено равным $P^*$.

### Агент Critic

Агент Critic формирует замкнутый цикл уточнения с Visualizer, тщательно анализируя сгенерированное изображение $I_t$ и предоставляя уточнённое описание $P_{t+1}$ для Visualizer. Получив сгенерированное изображение $I_t$ на итерации $t$, Critic проверяет его по отношению к исходному контексту $(S, C)$ для выявления фактологических расхождений, визуальных артефактов или зон улучшения. Затем он предоставляет целевую обратную связь и формирует уточнённое описание $P_{t+1}$, устраняющее выявленные проблемы:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
Это пересмотренное описание затем подаётся обратно в Visualizer для регенерации. Цикл Visualizer-Critic выполняется $T=3$ итерации, при этом финальный вывод составляет $I = I_T$. Этот процесс итеративного уточнения гарантирует, что финальная иллюстрация соответствует высоким стандартам, необходимым для академической публикации.

### Расширение на статистические графики

Фреймворк распространяется на статистические графики путём настройки агентов Visualizer и Critic. Для численной точности Visualizer преобразует описание $P_t$ в исполняемый код Python Matplotlib: $I_t = \text{VLM}_{\text{code}}(P_t)$. Critic оценивает отрисованный график и генерирует уточнённое описание $P_{t+1}$, устраняя неточности или недостатки: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. Применяется тот же процесс итеративного уточнения на $T=3$ раундов. Хотя мы приоритизируем этот подход на основе кода для обеспечения точности, мы также исследуем прямую генерацию изображений в разделе \ref{sec:discussion}. См. приложение \ref{app_sec:plot_agent_prompt} для скорректированных промптов."""

        example_caption = "Рисунок 1: Обзор нашего фреймворка PaperBanana. Задав исходный контекст и коммуникативное намерение, мы сначала применяем Фазу линейного планирования для извлечения релевантных эталонных примеров и синтеза стилистически оптимизированного описания. Затем мы используем Итеративный цикл уточнения (состоящий из агентов Visualizer и Critic) для преобразования описания в визуальный вывод и проведения многораундового уточнения для создания финальной академической иллюстрации."
        
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            # Example selector for method content
            method_example = st.selectbox(
                "Загрузить пример (Метод)",
                ["None", "PaperBanana Framework"],
                key="method_example_selector"
            )
            
            # Set value based on example selection or session state
            if method_example == "PaperBanana Framework":
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")
            
            method_content = st.text_area(
                "Содержание раздела метода / Данные графика (рекомендуется Markdown или JSON)",
                value=method_value,
                height=250,
                placeholder="Вставьте содержание раздела метода сюда...",
                help="Раздел метода или данные графика, описывающие фигуру. Поддерживаются Markdown и JSON."
            )
        
        with col_input2:
            # Example selector for caption
            caption_example = st.selectbox(
                "Загрузить пример (Подпись)",
                ["None", "PaperBanana Framework"],
                key="caption_example_selector"
            )
            
            # Set value based on example selection or session state
            if caption_example == "PaperBanana Framework":
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")
            
            caption = st.text_area(
                "Подпись к фигуре / Визуальное намерение (рекомендуется Markdown)",
                value=caption_value,
                height=250,
                placeholder="Введите подпись к фигуре...",
                help="Подпись или описание генерируемой фигуры. Рекомендуется формат Markdown."
            )
        
        # Кнопка обработки
        if st.button("🚀 Сгенерировать кандидатов", type="primary", use_container_width=True):
            if not method_content or not caption:
                st.error("Укажите содержание метода и подпись!")
            else:
                # Save to session state
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption
                
                with st.spinner(f"Параллельная генерация {num_candidates} кандидатов... Это может занять несколько минут."):
                    # Create input data list
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        figure_size=figure_size,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds,
                        task_name=task_name,
                    )
                    
                    # Process in parallel
                    try:
                        results = asyncio.run(process_parallel_candidates(
                            input_data_list,
                            exp_mode=exp_mode,
                            retrieval_setting=retrieval_setting,
                            main_model_name=main_model_name,
                            image_gen_model_name=image_gen_model_name,
                            task_name=task_name,
                        ))
                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str
                        
                        # Save results to JSON file
                        try:
                            # Create results directory if it doesn't exist
                            results_dir = Path(__file__).parent / "results" / "demo"
                            results_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Generate filename with timestamp
                            json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            
                            # Save to JSON with proper encoding handling (like main.py)
                            with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                                json_string = json.dumps(results, ensure_ascii=False, indent=4)
                                # Clean invalid UTF-8 characters
                                json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
                                f.write(json_string)
                            
                            st.session_state["json_file"] = str(json_filename)
                        st.success(f"✅ Успешно сгенерировано {len(results)} кандидатов!")
                            st.info(f"💾 Результаты сохранены: `{json_filename.name}`")
                    except Exception as e:
                        st.warning(f"⚠️ Сгенерировано {len(results)} кандидатов, но не удалось сохранить JSON: {e}")
                    except Exception as e:
                        st.error(f"Ошибка во время обработки: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display results
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            
            st.divider()
            st.markdown("## 🎨 Сгенерированные кандидаты")
            st.caption(f"Сгенерировано: {timestamp} | Конвейер: {mode_info.get(current_mode, current_mode)}")
            
            # Показываем скачивание JSON-файла, если доступно
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 Результаты сохранены: `{json_file_path.relative_to(Path.cwd())}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="⬇️ Скачать JSON",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            use_container_width=True
                        )
            
            # Отображаем результаты в сетке (3 столбца)
            num_cols = 3
            num_results = len(results)
            
            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)
            
            # Добавляем кнопку скачивания ZIP
            st.divider()
            st.markdown("### 💾 Массовое скачивание")
            
            try:
                import zipfile
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for candidate_id, result in enumerate(results):
                        final_image_key = resolve_final_output(
                            result,
                            exp_mode=current_mode,
                        ).image_key
                        
                        if final_image_key and final_image_key in result:
                            img = base64_to_image(result[final_image_key])
                            if img:
                                img_buffer = BytesIO()
                                img.save(img_buffer, format="PNG")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.png",
                                    img_buffer.getvalue()
                                )
                
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ Скачать ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"paperbanana_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                st.success("ZIP-файл готов к скачиванию!")
            except Exception as e:
                st.error(f"Не удалось создать ZIP: {e}")
    
    # ==================== ВКЛАДКА 2: Уточнение изображения ====================
    with tab2:
        st.markdown("### Уточните и увеличьте разрешение диаграммы до 2K/4K")
        st.caption("Загрузите изображение из кандидатов или любую диаграмму, опишите изменения и сгенерируйте версию в высоком разрешении")
        
        # Боковая панель настроек уточнения
        with st.sidebar:
            st.title("✨ Настройки уточнения")
            
            refine_resolution = st.selectbox(
                "Целевое разрешение",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="Большее разрешение занимает больше времени, но даёт лучшее качество"
            )
            
            refine_aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ["21:9", "16:9", "3:2"],
                index=0,
                key="refine_aspect_ratio",
                help="Соотношение сторон для уточнённого изображения"
            )
        
        st.divider()
        
        # Секция загрузки
        st.markdown("## 📤 Загрузка изображения")
        uploaded_file = st.file_uploader(
            "Выберите файл изображения",
            type=["png", "jpg", "jpeg"],
            help="Загрузите диаграмму для уточнения"
        )
        
        if uploaded_file is not None:
            # Отображаем загруженное изображение
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Оригинальное изображение")
                st.image(uploaded_image, use_container_width=True)
            
            with col2:
                st.markdown("### Инструкции по редактированию")
                edit_prompt = st.text_area(
                    "Опишите нужные изменения",
                    height=200,
                    placeholder="Например, «Измените цветовую схему под стиль академической статьи» или «Сделайте текст крупнее и жирнее» или «Оставьте всё как есть, но увеличьте разрешение»",
                    help="Опишите, что вы хотите изменить, или используйте «Оставьте всё как есть» для простого увеличения разрешения",
                    key="edit_prompt"
                )
                
                if st.button("✨ Уточнить изображение", type="primary", use_container_width=True):
                    if not edit_prompt:
                        st.error("Укажите инструкции по редактированию!")
                    else:
                        with st.spinner(f"Уточнение изображения до разрешения {refine_resolution}... Это может занять минуту."):
                            try:
                                # Конвертируем PIL изображение в байты
                                img_byte_arr = BytesIO()
                                uploaded_image.save(img_byte_arr, format='JPEG')
                                image_bytes = img_byte_arr.getvalue()
                                
                                # Вызываем API nanoviz
                                refined_bytes, message = asyncio.run(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution
                                    )
                                )
                                
                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"Ошибка во время уточнения: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            
            # Отображаем результат уточнения, если доступен
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown("## 🎨 Результат уточнения")
                st.caption(f"Сгенерировано: {st.session_state.get('refine_timestamp', 'N/A')} | Разрешение: {refine_resolution}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### До")
                    st.image(uploaded_image, use_container_width=True)
                
                with col2:
                    st.markdown(f"### После ({refine_resolution})")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, use_container_width=True)
                    
                    # Кнопка скачивания
                    st.download_button(
                        label=f"⬇️ Скачать изображение {refine_resolution}",
                        data=st.session_state["refined_image"],
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()
