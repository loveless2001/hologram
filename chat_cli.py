"""Command-line chat demo powered by holographic memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from hologram.api import Hologram
from hologram.chatbot import ChatMemory, ChatWindow, SessionLog, resolve_provider


def load_hologram(path: Path, use_clip: bool, model_name: str, pretrained: str) -> Hologram:
    if path.exists():
        return Hologram.load(
            str(path),
            model_name=model_name,
            pretrained=pretrained,
            use_clip=use_clip,
        )
    return Hologram.init(model_name=model_name, pretrained=pretrained, use_clip=use_clip)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Session identifier (e.g. user42)")
    parser.add_argument("--memory", default="memory_store.json", help="Path to hologram memory file")
    parser.add_argument("--log-dir", default="chatlogs", help="Directory for per-session chat logs")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--use-clip", action="store_true", help="Use OpenCLIP encoders instead of hashing fallback")
    parser.add_argument("--model-name", default="ViT-B-32", help="CLIP model name (if --use-clip)")
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="CLIP pretrained weights tag (if --use-clip)",
    )
    parser.add_argument("--chat-model", default="gpt-3.5-turbo", help="LLM model name for OpenAI provider")
    parser.add_argument("--api-key", default=None, help="Optional API key override")
    parser.add_argument(
        "--session-window",
        type=int,
        default=6,
        help="Number of recent turns to keep from the active session",
    )
    parser.add_argument(
        "--cross-session-k",
        type=int,
        default=3,
        help="How many memories to surface from other sessions",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    memory_path = Path(args.memory)
    hologram = load_hologram(memory_path, args.use_clip, args.model_name, args.pretrained)

    provider = resolve_provider(api_key=args.api_key, model=args.chat_model)

    memory = ChatMemory(
        hologram=hologram,
        session_window=args.session_window,
        cross_session_k=args.cross_session_k,
    )
    logs = SessionLog(Path(args.log_dir))
    window = ChatWindow(
        provider=provider,
        memory=memory,
        logs=logs,
        session_id=args.session,
        system_prompt=args.system_prompt,
    )

    try:
        window.run_cli()
    finally:
        hologram.save(str(memory_path))
        print(f"Saved holographic memory to {memory_path}")


if __name__ == "__main__":
    main()
