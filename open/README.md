# AutoNovel Agent (Open Source Minimal Package)

A minimal, ready-to-share version of the AutoNovel project.

## 1. What this project does
This agent generates a web-novel pipeline with LLMs:
- Build world setting from one-line idea
- Generate chapter outlines
- Keep in-memory state across chapters
- Write chapter text and save `.txt` files under `output/{novel_name}/`

## 2. Requirements
- Python 3.10+
- DeepSeek API key

Install dependencies:
```bash
pip install -r requirements.txt
```

## 3. Configuration
1. Copy `.env.example` to `.env`
2. Fill your own API key in `.env`

Example:
```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_STATE_MODEL=deepseek-chat
```

## 4. Run
```bash
python main_orchestrator.py --idea "赛博朋克修仙" --chapters 10
```

Example:
```bash
python main_orchestrator.py --idea "灵气复苏时代，主角靠解析古籍修仙" --chapters 12
```

## 5. Output
Generated chapter files will be saved to:
- `output/{novel_name}/001_xxx.txt`
- `output/{novel_name}/002_xxx.txt`
- ...

## 6. File structure
```text
open/
├── .env.example
├── requirements.txt
├── README.md
├── llm_utils.py
├── main_orchestrator.py
├── world_builder.py
├── plot_router.py
├── state_manager.py
└── writer_agent.py
```

## 7. Notes for open source
- Do NOT commit your real `.env`
- API key should only stay in your local machine
- Generated novel content is not included by default

## 8. Disclaimer
- This project is for learning and research purposes only.
- AI-generated content may contain factual errors, bias, unsafe or inappropriate text. Please review and edit before any public use.
- Users are solely responsible for legal compliance, copyright review, and content moderation in their own jurisdiction.
- Any API usage costs, rate limits, or account risks are borne by the user of their own API key.
