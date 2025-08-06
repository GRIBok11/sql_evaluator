import gradio as gr
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from docx import Document


load_dotenv()
groq_api_key = os.getenv("groq_api_key")


llm = ChatGroq(
    temperature=0.3,
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it"
)

evaluation_prompt = PromptTemplate.from_template("""
Вы — эксперт по SQL. Проанализируйте текст ниже, содержащий описание структуры базы данных, формулировку задания и SQL-запрос, представленный студентом. Ваша задача — определить, соответствует ли SQL-запрос требованиям задачи и дать итоговую оценку (Правильный / Неправильный) с кратким пояснением.

Текст:
{text}

Дай ответ в виде правильный скрипт или нет на основании того что вывод соответсвует условию задачи:
""")

analyze_chain = evaluation_prompt | llm | StrOutputParser()


def read_full_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs]).strip()


def evaluate_docx_whole(file, schema_text):
    text = read_full_docx(file)
    if not text.strip() and not schema_text.strip():
        return "❗️**Файл и скрипт пусты — нечего анализировать.**"
    
    combined_text = f"{text.strip()}\n\n### SQL-структура базы данных:\n{schema_text.strip()}"
    result = analyze_chain.invoke({"text": combined_text})
    
    return f"### 📝 Результат анализа\n\n{result.strip()}"



with gr.Blocks(title="AI-оценка SQL-задания 🧾") as demo:
    gr.Markdown("## 📄 Загрузите файл с заданием и вставьте SQL-скрипт структуры БД")
    
    file_input = gr.File(label="📥 Файл с заданием (.docx)", file_types=[".docx"])
    schema_input = gr.Textbox(label="📄 SQL-скрипт создания таблиц", lines=12, placeholder="Вставьте сюда CREATE TABLE ...")
    
    output = gr.Markdown(label="📊 Оценка")
    btn = gr.Button("🔍 Оценить")

    btn.click(fn=evaluate_docx_whole, inputs=[file_input, schema_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
