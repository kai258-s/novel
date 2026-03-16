from typing import Any

from pydantic import BaseModel, ValidationError

from llm_utils import call_chat_completion, parse_json_from_text


class WorldSetting(BaseModel):
    novel_name: str
    power_system: str
    main_character: str
    background: str


def build_world(
    client: Any,
    premise: str,
    *,
    model: str = "deepseek-chat",
) -> dict[str, str]:
    system_prompt = (
        "你是资深网络小说策划编辑。"
        "请根据用户一句话设定，生成可用于长篇网文创作的世界观。"
        "只返回 JSON，不要任何解释。"
    )
    user_prompt = f"""
用户设定：{premise}

请输出 JSON，字段必须齐全：
{{
  "novel_name": "小说名",
  "power_system": "力量体系",
  "main_character": "主角初始人设",
  "background": "背景设定"
}}
"""

    text = call_chat_completion(
        client,
        model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=1200,
    )
    payload = parse_json_from_text(text)
    try:
        return WorldSetting.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise ValueError(f"Invalid world setting JSON: {exc}") from exc

