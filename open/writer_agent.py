from typing import Any

from llm_utils import call_chat_completion


def write_chapter(
    client: Any,
    world_setting: dict[str, Any],
    chapter_outline: dict[str, Any],
    memory_snapshot: dict[str, Any],
    *,
    model: str = "deepseek-chat",
    min_chars: int = 2000,
) -> str:
    system_prompt = (
        "你是顶级中文网文作家，擅长快节奏爽文。"
        "必须保证叙事连贯、人物动机清晰、战斗与情绪起伏有层次。"
        "输出纯正文，不要解释，不要标题。"
    )
    user_prompt = f"""
【世界观】
{world_setting}

【本章大纲】
{chapter_outline}

【记忆状态（最近章节）】
{memory_snapshot}

写作要求：
1. 输出不少于 {min_chars} 字中文正文。
2. 风格偏网络小说，爽点密度高，结尾留钩子。
3. 与前文状态一致，不要遗忘角色物品和当前地点。
4. 禁止输出任何“以下是正文”等说明语。
"""
    chapter_text = call_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.95,
        max_tokens=4500,
    )

    if len(chapter_text) < min_chars:
        append_prompt = f"""
你刚才的正文不足 {min_chars} 字，请直接续写同一章内容并收束在悬念点。
只输出新增正文，不要重复之前内容。
"""
        appended = call_chat_completion(
            client,
            model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": chapter_text},
                {"role": "user", "content": append_prompt},
            ],
            temperature=0.9,
            max_tokens=2500,
        )
        chapter_text = f"{chapter_text}\n\n{appended}".strip()

    return chapter_text

