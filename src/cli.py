import argparse
import logging

import uvicorn

from .service import create_app
from .config import EvaluationConfig, ServiceConfig, load_config, required_key, resolved_path, write_snapshot
from .providers import OpenAIEmbedder, OpenAILLM


def main():
    parser = argparse.ArgumentParser(prog="anchor-mem")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        config, config_path = load_config(args.config, ServiceConfig)
        required_key(config.auth_key_env)
        logging.basicConfig(level=config.log_level.upper())
        snapshot_dir = resolved_path(config_path, config.snapshot_dir)
        write_snapshot(config, snapshot_dir / "resolved-service-config.yaml")
        app = create_app(config, config_path, OpenAILLM(config.llm), OpenAIEmbedder(config.embedding))
        uvicorn.run(app, host=config.host, port=config.port, workers=config.workers)
    else:
        from .evaluation import evaluate

        config, config_path = load_config(args.config, EvaluationConfig)
        evaluate(config, config_path)


if __name__ == "__main__":
    main()
