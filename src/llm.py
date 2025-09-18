from typing import List

from ollama import chat, ChatResponse


class OllamaLLM:
    """
    Обертка для модели Ollama Qwen2-0.5B
    """
    def __init__(self, model_name: str = "qwen2:0.5b"):
        self.model_name = model_name

    def extract_product_name(self, text: str) -> str:
        response: ChatResponse = chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": f"Выдели только название товара из текста. Верни только одно слово или короткую фразу, никаких кавычек, объяснений или дополнительного текста: {text}"
            }]
        )
        return response.message.content.strip()

    def recommend_products(self, product_name: str, found_items: List[dict]) -> str:
        if not found_items:
            return "По вашему запросу товаров не найдено."

        items_text = "\n".join([f"- {item}" for item in found_items])
        prompt = (
            f"Пользователь ищет: {product_name}\n"
            f"Вот найденные товары:\n{items_text}\n"
            f"Сделай краткие рекомендации и предложения, какие из них выбрать."
        )

        response: ChatResponse = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.message.content.strip()
