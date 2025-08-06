import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Загрузка переменных окружения из .env
load_dotenv()
groq_api_key = os.getenv('groq_api_key')

# Проверка ключа
if not groq_api_key:
    raise ValueError("❗ Не найден API-ключ GROQ. Убедись, что в файле .env задан 'groq_api_key'.")

# Инициализация модели
llm = ChatGroq(
    temperature=0.3,
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it"
)

# Описание структуры БД
db_description = """
Структура базы данных:

Таблица `CLIENTS` — содержит информацию о клиентах.
- `ClientID` — уникальный идентификатор клиента.
- `ClientName` — имя клиента.
- `Type` — тип клиента (например, отправитель или получатель).

Таблица `PRODUCTS` — содержит информацию о товарах.
- `ProductID` — уникальный идентификатор товара.
- `ProductName` — название товара (например, "TV", "PC").

Таблица `TTNS` — содержит данные о товарно-транспортных накладных.
- `TTNID` — уникальный идентификатор накладной.
- `TTNNumber` — номер накладной.
- `TTNDate` — дата накладной.
- `SenderID` — внешний ключ на `CLIENTS(ClientID)`, отправитель.
- `ReceiverID` — внешний ключ на `CLIENTS(ClientID)`, получатель.

Таблица `SPECIFICATIONS` — содержит спецификации товаров в рамках накладных.
- `TTNID` — внешний ключ на `TTNS(TTNID)`.
- `ProductID` — внешний ключ на `PRODUCTS(ProductID)`.
- `Count` — количество товара.
- `Price` — цена за единицу товара.
"""

# Шаблон промпта
evaluation_prompt = PromptTemplate.from_template("""
Вы — эксперт по SQL. Ниже приведено описание структуры базы данных, SQL-задача и SQL-запрос, написанный студентом.
Если запрос реализует логику задачи и может дать правильный результат — даже если он отличается от эталона — считайте его "Правильным".

Описание базы данных:
{db_description}

Задание:
{task}

Ответ студента (SQL-запрос):
{student_answer}

Проанализируйте решение студента:
1. Итог запроса будет соответствовать условию задачи.
2. Дайте итоговую краткую оценку: Правильный / Неправильный.

дай ответ в виде правильно\неправильно
""")

# Цепочка анализа
analyze_chain = evaluation_prompt | llm

def analyze_excel(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")
    results = []
    correct_matches = 0
    total_checked = 0

    def extract_verdict(text):
        text = text.lower().replace("ё", "е").strip()
        p = text.find("правильн")
        np = text.find("неправильн")

        if p == -1 and np == -1:
            return "неопределено"
        if np != -1 and (p == -1 or np < p):
            return "неправильно"
        return "правильно"


    for i, row in df.iterrows():
        try:
            task = str(row.iloc[0]).strip()
            answer = str(row.iloc[1]).strip()
            comment = str(row.iloc[2]).strip() if len(row) > 2 else ""

            if not task or not answer:
                continue

            result = analyze_chain.invoke({
                "db_description": db_description,
                "task": task,
                "student_answer": answer
            })

            llm_verdict = extract_verdict(result.content)
            expected_verdict = extract_verdict(comment)

            if llm_verdict == expected_verdict and llm_verdict != "неопределено":
                correct_matches += 1

            total_checked += 1

            combined_output = f"{result.content.strip()}  📝 Правильная оценка: {comment}"
            results.append((i + 1, task, combined_output))

        except Exception as e:
            results.append((i + 1, task, f"⚠️ Ошибка при анализе: {e}"))

    return results, correct_matches, total_checked




if __name__ == "__main__":
    file_path = "тесты.xlsx"
    
    if not os.path.exists(file_path):
        print(f"❗ Файл '{file_path}' не найден.")
    else:
        analyzed, correct, total = analyze_excel(file_path)

        for idx, task_text, feedback in analyzed:
            print(f"\n🧾 Задание {idx}  {feedback}\n{'-'*70}")

        print(f"\n✅ Совпадений: {correct} из {total} ({round(correct / total * 100, 1)}%)")
