import os

import pytest
from datasets import Dataset
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import FactualCorrectness, Faithfulness

load_dotenv()

THRESHOLD = 0.2
MODEL_NAME = os.getenv("MODEL_NAME")

test_llm = ChatOllama(model=MODEL_NAME)

embeddings = OllamaEmbeddings(model=MODEL_NAME)

data = [
    {
        "response": "Можно выбрать настенный дозатор BERKRAFT Line на 200 мл.",
        "contexts": ["Дозатор для жидкого мыла настенный BERKRAFT Line 200 мл"],
        "reference": "Отличным выбором станет настенный дозатор BERKRAFT Line на 200 мл.",
        "user_input": "Какой можно выбрать дозатор на стену 200 мл.",
    }
]


@pytest.mark.parametrize("example", data)
def test_llm_metrics(example):
    single_dataset = Dataset.from_list([example])
    results_list = evaluate(
        dataset=single_dataset,
        metrics=[FactualCorrectness(), Faithfulness()],
        llm=test_llm,
        show_progress=True,
        raise_exceptions=True,
    )

    for metric in ["faithfulness", "factual_correctness(mode=f1)"]:
        score = results_list[metric][0]
        print(f"Пример: {example['user_input']}")
        print(f"Метрика {metric}: {score}")
        assert score >= THRESHOLD, f"Метрика {metric} слишком низкая: {score}"


