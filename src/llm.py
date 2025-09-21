from typing import List

from ollama import ChatResponse, chat

LIMIT_TXT = 20


class OllamaLLM:
    """
    Обертка для модели Ollama Qwen2-0.5B
    """

    def __init__(self, model_name: str = "qwen2:0.4b"):
        self.model_name = model_name

    def recommend_products(self, query: str, found_items: List[dict]) -> str:
        items_text = "\n".join(
            [
                f"- {item['name']}: {' '.join(item['description'].split()[:LIMIT_TXT])}"
                for item in found_items
            ]
        )
        prompt = (
            f"Пользователь ищет: {query}\n"
            f"Вот найденные товары:\n{items_text}\n"
            f"Сделай краткие рекомендации и предложения для пользователя, какие из них выбрать."
        )

        response: ChatResponse = chat(
            model=self.model_name, messages=[{"role": "user", "content": prompt}]
        )

        return response.message.content.strip()
