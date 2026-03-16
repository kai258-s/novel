from typing import Any

from pydantic import BaseModel, ValidationError

from llm_utils import call_chat_completion, parse_json_from_text


class ChapterOutline(BaseModel):
    chapter_number: int
    chapter_title: str
    plot_summary: str


def route_plot(
    client: Any,
    world_setting: dict[str, Any],
    total_chapters: int,
    *,
    model: str = "deepseek-chat",
) -> list[dict[str, Any]]:
    if total_chapters <= 0:
        raise ValueError("total_chapters must be greater than 0.")

    system_prompt = (
        "你是顶级网文大纲师。你会根据既有世界观，生成严谨、连贯、有爽点推进的章节大纲。"
        "只返回 JSON 数组，不要任何解释。"
    )
    user_prompt = f"""
世界观设定：
{world_setting}

总章节数：{total_chapters}

请返回长度为 {total_chapters} 的 JSON 数组，每个元素包含：
{{
  "chapter_number": 1,
  "chapter_title": "章名",
  "plot_summary": "本章剧情梗概"
}}

要求：
1. chapter_number 必须从 1 连续到 {total_chapters}。
2. 每章梗概都要有明确剧情推进。
3. 不能出现空字段。
"""
    text = call_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        max_tokens=3000,
    )
    payload = parse_json_from_text(text)
    if not isinstance(payload, list):
        raise ValueError("Plot router output is not a JSON list.")

    outlines: list[dict[str, Any]] = []
    for item in payload:
        try:
            outlines.append(ChapterOutline.model_validate(item).model_dump())
        except ValidationError as exc:
            raise ValueError(f"Invalid chapter outline JSON: {exc}") from exc

    if len(outlines) != total_chapters:
        raise ValueError(
            f"Expected {total_chapters} chapters, but got {len(outlines)} chapters."
        )

    return outlines

