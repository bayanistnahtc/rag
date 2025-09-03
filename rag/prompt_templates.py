import textwrap

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Template for rephrasing questions based on chat history
REPHRASE_QUESTION_TEMPLATE = textwrap.dedent(
    """
Учитывая историю переписки и новый вопрос, переформулируйте новый вопрос так,
чтобы он был полностью самостоятельным и не зависел от предыдущих сообщений.
Если история пуста или не содержит релевантной информации,
верните исходный вопрос без изменений.

История переписки:
{chat_history}

Новый вопрос: {question}

Самостоятельный вопрос:
"""
)

REPHRASE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_QUESTION_TEMPLATE)


# Профессиональный промпт для генерации финального ответа
ANSWER_GENERATION_TEMPLATE = textwrap.dedent(
    """Вы — профессиональный технический ассистент компании ELEPS.
Ваша задача — предоставлять точные, структурированные и исчерпывающие ответы инженерам,
используя исключительно предоставленный контекст технической документации.

**Инструкции:**
1. **Используйте только контекст или историю диалога:**
Не добавляйте собственные знания или предположения.
Ответ должен быть основан исключительно на предоставленных фрагментах или
истории диалога.
2. **Нет информации — нет ответа:**
Если в контексте нет ответа, корректно сообщите:
"К сожалению, в предоставленной документации не содержится информации по вашему запросу.
Пожалуйста, уточните вопрос или проверьте выбранное оборудование."
3. **Структурируйте ответ:**
Используйте списки, выделяйте **ключевые термины** и
делите текст на абзацы для удобства восприятия.
4. **Синтезируйте информацию:**
Объединяйте сведения из разных фрагментов, если они относятся к одному вопросу,
чтобы дать цельный и логичный ответ.
5. **Точное цитирование:**
После каждого смыслового блока указывайте источники строго в формате:
`[ИСТОЧНИК: 'file_name', page: X]`. Не придумывайте названия файлов или номера страниц;
используйте только те, что есть в метаданных.

---
Контекст документации:
--------------------
{context}
--------------------

Вопрос пользователя: {question}

Профессиональный, развернутый и точный ответ:
"""
)

ANSWER_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_GENERATION_TEMPLATE),
        ("human", "{question}"),
    ]
)

# Clarification prompt template with optional question and context
CLARIFICATION_PROMPT_TEMPLATE = textwrap.dedent(
    """Вы — технический ассистент компании ELEPS.
{question_section}{context_section}
Вежливо попросите уточнить детали, которые помогут найти нужную информацию.
Будьте конкретны и предложите возможные направления уточнения.

Примеры хороших уточнений:
- "Уточните, пожалуйста, о каком именно оборудовании идет речь?"
- "Какую конкретно проблему или ошибку вы наблюдаете?"
- "Для какого типа процедуры вам нужна эта информация?"

Уточняющий вопрос:
"""
)


def create_clarification_prompt(
    question: str = "", context: str = ""
) -> ChatPromptTemplate:
    """
    Create clarification prompt with optional question and context.

    Args:
        question: Optional specific question that needs clarification
        context: Optional context from previous conversation

    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    if question.strip():
        question_section = (
            "Пользователь задал вопрос, который требует уточнения: "
            f'"{question.strip()}"\n\n'
        )
    else:
        question_section = "Исходя из истории диалога, "

    if context.strip():
        context_section = f"""
Контекст предыдущего диалога:
{context}

Учитывая контекст диалога, """
    else:
        context_section = ""

    template = CLARIFICATION_PROMPT_TEMPLATE.format(
        question_section=question_section, context_section=context_section
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", template),
        ]
    )


# Default clarification prompt without context
CLARIFICATION_PROMPT = create_clarification_prompt()

# Fallback clarification messages
FALLBACK_CLARIFICATION_WITH_QUESTION = (
    "Пожалуйста, уточните ваш вопрос: '{question}'. "
    "Какие дополнительные детали могут помочь найти нужную информацию?"
)

FALLBACK_CLARIFICATION_WITHOUT_QUESTION = (
    "Пожалуйста, уточните детали вашего запроса. "
    "Какие дополнительные сведения могут помочь найти нужную информацию?"
)
