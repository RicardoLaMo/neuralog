"""
NeuraLog CLI commands for extraction, reasoning, and verification.

This module provides command-line interfaces for NeuraLog functionality.
Can be extended with argparse, Click, or Typer.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from neuralog.core.engine import Engine
from neuralog.core.config import Config
from neuralog.utils.logger import get_logger

logger = get_logger(__name__)


def extract_command(
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    output_file: Optional[str] = None,
    config_path: Optional[str] = None,
    format: str = "json",
) -> int:
    """
    Extract knowledge (entities, relations, triples) from text.

    Args:
        text: Input text to extract from
        file_path: Path to input file
        output_file: Path to save output
        config_path: Path to config file
        format: Output format (json, csv, turtle)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Get input text
        if text is None and file_path is None:
            logger.error("Must provide either --text or --file-path")
            return 1

        if file_path:
            with open(file_path, "r") as f:
                text = f.read()

        # Initialize engine
        config = Config() if config_path is None else Config.from_yaml(config_path)
        engine = Engine(config=config)

        # Extract
        logger.info(f"Extracting from {len(text)} characters of text")
        result = engine.extract(text)

        # Format output
        if format == "json":
            output = _format_result_json(result)
        elif format == "csv":
            output = _format_result_csv(result)
        else:
            logger.warning(f"Unknown format {format}, using json")
            output = _format_result_json(result)

        # Save or print
        if output_file:
            with open(output_file, "w") as f:
                f.write(output)
            logger.info(f"Saved to {output_file}")
        else:
            print(output)

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1


def reason_command(
    input_file: str,
    output_file: Optional[str] = None,
    config_path: Optional[str] = None,
) -> int:
    """
    Apply logical reasoning to knowledge graph.

    Args:
        input_file: Path to input KG file
        output_file: Path to save output
        config_path: Path to config file

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Load knowledge graph
        logger.info(f"Loading knowledge graph from {input_file}")
        config = Config() if config_path is None else Config.from_yaml(config_path)
        engine = Engine(config=config)

        # TODO: Implement KG loading when load_knowledge_graph is complete
        # kg = engine.load_knowledge_graph(input_file)

        # Reason
        # inferred_triples = engine.reason(kg.triples)

        logger.warning("Reasoning not yet implemented")
        return 1

    except Exception as e:
        logger.error(f"Reasoning failed: {e}", exc_info=True)
        return 1


def verify_command(
    input_file: str,
    output_file: Optional[str] = None,
    config_path: Optional[str] = None,
    policy: Optional[str] = None,
) -> int:
    """
    Verify knowledge graph against ontology constraints.

    Args:
        input_file: Path to input KG file
        output_file: Path to save verification report
        config_path: Path to config file
        policy: Verification policy

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Load knowledge graph
        logger.info(f"Loading knowledge graph from {input_file}")
        config = Config() if config_path is None else Config.from_yaml(config_path)
        engine = Engine(config=config)

        # TODO: Implement KG loading when load_knowledge_graph is complete
        # kg = engine.load_knowledge_graph(input_file)

        # Verify
        # result = engine.verify(kg.triples, policy=policy)

        logger.warning("Verification not yet implemented")
        return 1

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return 1


def _format_result_json(result) -> str:
    """Format extraction result as JSON."""
    data = {
        "entities": [
            {
                "text": e.text,
                "label": e.label,
                "confidence": e.confidence,
                "start_char": e.start_char,
                "end_char": e.end_char,
            }
            for e in result.entities
        ],
        "relations": [
            {
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
                "confidence": r.confidence,
            }
            for r in result.relations
        ],
        "triples": [
            {
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.object,
                "confidence": t.confidence,
            }
            for t in result.triples
        ],
        "confidence": str(result.confidence),
    }
    return json.dumps(data, indent=2)


def _format_result_csv(result) -> str:
    """Format extraction result as CSV."""
    lines = []

    # Entities
    lines.append("# ENTITIES")
    lines.append("text,label,confidence,start_char,end_char")
    for e in result.entities:
        lines.append(f'"{e.text}",{e.label},{e.confidence},{e.start_char},{e.end_char}')

    # Relations
    lines.append("\n# RELATIONS")
    lines.append("source,target,relation_type,confidence")
    for r in result.relations:
        lines.append(f'"{r.source}","{r.target}",{r.relation_type},{r.confidence}')

    # Triples
    lines.append("\n# TRIPLES")
    lines.append("subject,predicate,object,confidence")
    for t in result.triples:
        lines.append(f'"{t.subject}","{t.predicate}","{t.object}",{t.confidence}')

    return "\n".join(lines)


# Example CLI entry point using argparse (can be replaced with Click or Typer)
def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuraLog: Neural-Symbolic AI for Information Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from text
  neuralog extract --text "Barack Obama was born in Hawaii."

  # Extract from file
  neuralog extract --file-path input.txt --output-file output.json

  # Extract with custom config
  neuralog extract --file-path input.txt --config-path configs/custom.yaml

  # Verify knowledge graph
  neuralog verify --input-file kg.ttl --output-file report.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract knowledge from text")
    extract_parser.add_argument("--text", help="Input text")
    extract_parser.add_argument("--file-path", help="Path to input file")
    extract_parser.add_argument("--output-file", help="Path to output file")
    extract_parser.add_argument("--config-path", help="Path to config file")
    extract_parser.add_argument(
        "--format", choices=["json", "csv", "turtle"], default="json"
    )

    # Reason command
    reason_parser = subparsers.add_parser("reason", help="Apply reasoning to KG")
    reason_parser.add_argument("input_file", help="Path to input knowledge graph")
    reason_parser.add_argument("--output-file", help="Path to output file")
    reason_parser.add_argument("--config-path", help="Path to config file")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify knowledge graph")
    verify_parser.add_argument("input_file", help="Path to input knowledge graph")
    verify_parser.add_argument("--output-file", help="Path to output report")
    verify_parser.add_argument("--config-path", help="Path to config file")
    verify_parser.add_argument("--policy", help="Verification policy")

    args = parser.parse_args()

    if args.command == "extract":
        exit_code = extract_command(
            text=args.text,
            file_path=args.file_path,
            output_file=args.output_file,
            config_path=args.config_path,
            format=args.format,
        )
    elif args.command == "reason":
        exit_code = reason_command(
            input_file=args.input_file,
            output_file=args.output_file,
            config_path=args.config_path,
        )
    elif args.command == "verify":
        exit_code = verify_command(
            input_file=args.input_file,
            output_file=args.output_file,
            config_path=args.config_path,
            policy=args.policy,
        )
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
