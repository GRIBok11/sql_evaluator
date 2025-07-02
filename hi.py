from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0.3,  
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it"
)

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

ТВой анализ:
""")


analyze_chain = evaluation_prompt | llm

my_task = """
Для каждого товара определить среднюю стоимость в одной ТТН в 2017 году. Для каждого продавца выделить все ТТН, выписанные в 2018 году в которых стоимость хотя бы одного товара была больше средней стоимости этого товара за 2017 году."""

my_answer ="""
SELECT q1.ttn, q2.sender FROM (SELECT s.ttnid AS ttn, s.productid AS prod, AVG(s.count * s.price) AS avg_cost FROM ttns t JOIN specifications s ON s.ttnid = t.ttnid WHERE EXTRACT(YEAR FROM t.ttndate) = 2017 GROUP BY s.ttnid, s.productid) q1 RIGHT JOIN (SELECT s.ttnid AS ttn, s.productid AS prod, t.senderid AS sender, s.count * s.price AS cost_ FROM ttns t JOIN specifications s ON s.ttnid = t.ttnid WHERE EXTRACT(YEAR FROM t.ttndate) = 2018 GROUP BY s.ttnid, s.productid, t.senderid, s.count, s.price) q2 ON q2.prod = q1.prod WHERE q1.avg_cost < q2.cost_ ORDER BY q2.sender
"""
response = analyze_chain.invoke({
    "db_description": db_description,
    "task": my_task,
    "student_answer": my_answer
})

print(response.content)