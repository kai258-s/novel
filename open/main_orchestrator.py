import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from llm_utils import safe_filename
from plot_router import route_plot
from state_manager import MemoryState
from world_builder import build_world
from writer_agent import write_chapter


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Please set DEEPSEEK_API_KEY in .env.")

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoNovel Agent orchestrator")
    parser.add_argument(
        "--idea",
        type=str,
        required=True,
        help='One-line novel premise, e.g. "赛博朋克修仙"',
    )
    parser.add_argument(
        "--chapters",
        type=int,
        default=10,
        help="Total chapters to generate (default: 10).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        help="Main LLM model name.",
    )
    parser.add_argument(
        "--state-model",
        type=str,
        default=os.getenv("DEEPSEEK_STATE_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        help="Lightweight model for state extraction.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    client = get_client()

    logger.info("Step 1/5: Building world setting from idea...")
    world_setting = build_world(client, args.idea, model=args.model)
    novel_name = safe_filename(world_setting.get("novel_name", "untitled_novel"))
    output_dir = Path("output") / novel_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Novel name: {}", novel_name)

    logger.info("Step 2/5: Routing plot for {} chapters...", args.chapters)
    outlines = route_plot(client, world_setting, args.chapters, model=args.model)

    logger.info("Step 3/5: Initializing in-memory state...")
    state = MemoryState()

    logger.info("Step 4/5: Writing chapters in pipeline...")
    for chapter in outlines:
        chapter_number = chapter["chapter_number"]
        chapter_title = safe_filename(chapter["chapter_title"], fallback=f"chapter_{chapter_number}")
        logger.info("Writing chapter {}: {}", chapter_number, chapter_title)

        chapter_text = write_chapter(
            client,
            world_setting,
            chapter,
            state.snapshot(),
            model=args.model,
        )

        chapter_path = output_dir / f"{chapter_number:03d}_{chapter_title}.txt"
        chapter_path.write_text(chapter_text, encoding="utf-8")
        logger.info("Saved: {}", chapter_path)

        state.update_state(client, chapter_text, model=args.state_model)
        logger.info("State updated. Location: {}", state.current_location)

    logger.info("Step 5/5: Done. All chapters generated in {}", output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Pipeline failed: {}", exc)
        raise

