import nltk
import pandas as pd
nltk.download('punkt')


def remove_incomplete_sentence(text):
    sentences = nltk.sent_tokenize(text)

    if sentences and sentences[-1].endswith(('?', '!', '.', ',', ':', ';')):
        return text

    if sentences:
        sentences.pop()

    cleaned_text = ' '.join(sentences)
    return cleaned_text


def remove_last_responses(text):
    last_user_index = text.rfind("User:")
    if last_user_index != -1:
        cleaned_text = text[:last_user_index]
    else:
        cleaned_text = text
    return cleaned_text.strip()


def remove_prompt(prompt, text):
    DEFAULT_INSTRUCTIONS1 = f"""The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User’s questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins.
"""
    USER_NAME1 = "User"
    BOT_NAME1 = "Falcon"
    DEFAULT_INSTRUCTIONS2 = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""
    USER_NAME2 = "USER"
    BOT_NAME2 = "ASSISTANT"

    DEFAULT_INSTRUCTION3 = f'[INST] {prompt} [/INST]'

    prompt1 = f"{DEFAULT_INSTRUCTIONS1}\n{USER_NAME1}: {prompt}\n{BOT_NAME1}:"
    prompt2 = f"{DEFAULT_INSTRUCTIONS2}\n{USER_NAME2}: {prompt}\n{BOT_NAME2}:"

    text = text.replace(prompt1, "")
    text = text.replace(prompt2, "")
    text = text.replace(DEFAULT_INSTRUCTION3, "")
    text = text.replace(prompt, "")
    return text.strip()


def preprocess_text(prompt, text):
    text = remove_prompt(prompt, text)
    text = remove_last_responses(text)
    text = remove_incomplete_sentence(text)
    return text


def get_word_sentence_counts(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    return len(sentences), len(words)
