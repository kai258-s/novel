from typing import Any

from pydantic import BaseModel, Field

from llm_utils import call_chat_completion, parse_json_from_text


class StateUpdate(BaseModel):
    current_inventory: list[str] = Field(default_factory=list)
    current_location: str = "未知地点"
    chapter_summary: str = ""


class MemoryState:
    def __init__(self) -> None:
        self.current_inventory: list[str] = []
        self.current_location: str = "未知地点"
        self.recent_context: list[str] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_inventory": self.current_inventory,
            "current_location": self.current_location,
            "recent_context": self.recent_context,
        }

    def update_state(
        self,
        client: Any,
        chapter_text: str,
        *,
        model: str = "deepseek-chat",
    ) -> None:
        system_prompt = (
            "你是小说状态提取器。你需要从正文中抽取主角状态变化。"
            "只返回 JSON，不要任何解释。"
        )
        user_prompt = f"""
请从下述章节正文中提取状态：
1. 主角当前物品/功法列表（current_inventory，字符串数组）
2. 主角当前所在地点（current_location，字符串）
3. 本章 80 字以内摘要（chapter_summary，字符串）

正文：
{chapter_text}

请输出 JSON：
{{
  "current_inventory": ["示例物品A", "示例功法B"],
  "current_location": "地点名",
  "chapter_summary": "摘要"
}}
"""
        text = call_chat_completion(
            client,
            model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        payload = parse_json_from_text(text)
        update = StateUpdate.model_validate(payload)

        if update.current_inventory:
            self.current_inventory = update.current_inventory
        if update.current_location:
            self.current_location = update.current_location
        if update.chapter_summary:
            self.recent_context.append(update.chapter_summary)
            self.recent_context = self.recent_context[-3:]

