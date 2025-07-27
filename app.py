import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Загрузка модели и токенизатора
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Функция для анализа тона
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    labels = ["Negative", "Neutral", "Positive"]
    sentiment = labels[scores.argmax()]
    return sentiment, scores

# Функция для генерации рекомендаций
def generate_recommendations(sentiment):
    if sentiment == "Negative":
        return [
            "Старайтесь использовать более мягкий и доброжелательный тон.",
            "Добавьте позитивные формулировки, чтобы улучшить восприятие."
        ]
    elif sentiment == "Neutral":
        return [
            "Попробуйте добавить больше эмоций, чтобы сделать разговор живее.",
            "Используйте открытые вопросы для вовлечения собеседника."
        ]
    else:  # Positive
        return [
            "Продолжайте поддерживать позитивный тон!",
            "Для большего эффекта добавьте конкретные примеры или комплименты."
        ]

# Streamlit интерфейс
st.title("Агент оценки звонков")
st.write("Введите расшифровку телефонного разговора для анализа тона и рекомендаций.")

# Ввод текста
input_text = st.text_area("Расшифровка разговора", height=200)

if st.button("Анализировать"):
    if input_text.strip():
        # Анализ тона
        sentiment, scores = analyze_sentiment(input_text)
        st.write(f"**Тон общения**: {sentiment}")
        st.write(f"**Уверенность модели**: Позитивный: {scores[2]:.2%}, Нейтральный: {scores[1]:.2%}, Негативный: {scores[0]:.2%}")

        # Генерация рекомендаций
        recommendations = generate_recommendations(sentiment)
        st.write("**Рекомендации для улучшения разговора**:")
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.error("Пожалуйста, введите текст для анализа.")