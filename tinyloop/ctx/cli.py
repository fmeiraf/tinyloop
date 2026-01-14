"""
Typer CLI implementation for CTX.

Provides the `tinyloop ctx` command for analyzing conversations.
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .analyzer import CTXAnalyzer, CTXResult

app = typer.Typer(
    name="ctx",
    help="Analyze LLM conversation token usage and detect dumb zone entry.",
    invoke_without_command=True,
)

console = Console()
error_console = Console(stderr=True)


def load_conversation(file_path: Path | None) -> dict:
    """
    Load conversation from file or stdin.

    Args:
        file_path: Path to JSON file, or None to read from stdin

    Returns:
        Parsed conversation dict with messages, optional model, tools
    """
    if file_path:
        if not file_path.exists():
            error_console.print(f"[red]Error:[/red] File not found: {file_path}")
            raise typer.Exit(code=2)
        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            error_console.print(f"[red]Error:[/red] Invalid JSON: {e}")
            raise typer.Exit(code=2)
    else:
        # Read from stdin
        if sys.stdin.isatty():
            error_console.print(
                "[red]Error:[/red] No input provided. "
                "Provide a file path or pipe JSON to stdin."
            )
            raise typer.Exit(code=2)
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            error_console.print(f"[red]Error:[/red] Invalid JSON from stdin: {e}")
            raise typer.Exit(code=2)

    # Normalize input format
    if isinstance(data, list):
        # Direct message array
        return {"messages": data}
    elif isinstance(data, dict):
        if "messages" not in data:
            error_console.print(
                "[red]Error:[/red] Input must contain 'messages' array"
            )
            raise typer.Exit(code=2)
        return data
    else:
        error_console.print("[red]Error:[/red] Invalid input format")
        raise typer.Exit(code=2)


def format_simple_output(result: CTXResult) -> str:
    """Format simple one-line status output."""
    pct = result.percentage_used * 100
    tokens_until = result.threshold_tokens - result.total_tokens

    if result.is_in_dumb_zone:
        over = abs(tokens_until)
        return (
            f"! {result.total_tokens:,} / {result.threshold_tokens:,} tokens "
            f"({pct:.1f}%) -- IN DUMB ZONE ({over:,} tokens over threshold)"
        )
    elif tokens_until < result.threshold_tokens * 0.1:
        # Approaching (within 10%)
        return (
            f"* {result.total_tokens:,} / {result.threshold_tokens:,} tokens "
            f"({pct:.1f}%) -- {tokens_until:,} tokens until dumb zone (warning: approaching)"
        )
    else:
        return (
            f"  {result.total_tokens:,} / {result.threshold_tokens:,} tokens "
            f"({pct:.1f}%) -- {tokens_until:,} tokens until dumb zone"
        )


def render_full_report(result: CTXResult, verbose: bool = False) -> None:
    """Render full TUI report with tables."""
    # Header panel
    header_content = f"""[bold]Model:[/bold] {result.model or 'Unknown'}
[bold]Context Window:[/bold] {result.context_window:,} tokens
[bold]Dumb Zone Threshold:[/bold] {result.threshold * 100:.0f}% ({result.threshold_tokens:,} tokens)
[bold]Tokenizer:[/bold] {result.tokenizer_used}"""

    console.print(
        Panel(
            header_content,
            title="[bold blue]TinyLoop CTX Analysis[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    # Round-by-round table
    round_table = Table(
        title="Round-by-Round Breakdown",
        show_header=True,
        header_style="bold cyan",
    )
    round_table.add_column("Round", justify="right", style="dim")
    round_table.add_column("Role", style="white")
    round_table.add_column("Tokens", justify="right")
    round_table.add_column("Cumulative", justify="right")
    round_table.add_column("% Used", justify="right")
    round_table.add_column("Status", justify="center")

    for r in result.rounds:
        status = "[red]! DUMB ZONE[/red]" if r.is_in_dumb_zone else "[green]  Safe[/green]"
        round_table.add_row(
            str(r.round_number),
            r.role,
            f"{r.tokens:,}",
            f"{r.cumulative_tokens:,}",
            f"{r.percentage_used * 100:.1f}%",
            status,
        )

    console.print(round_table)
    console.print()

    # Token distribution table
    dist_table = Table(
        title="Token Distribution",
        show_header=True,
        header_style="bold cyan",
    )
    dist_table.add_column("Category", style="white")
    dist_table.add_column("Tokens", justify="right")
    dist_table.add_column("Percentage", justify="left")

    total = result.total_tokens
    for category, tokens in result.categories.items():
        if total > 0:
            pct = tokens / total * 100
            bar_width = int(pct / 3)  # Scale to ~30 chars max
            bar = "[cyan]" + "|" * bar_width + "[/cyan]"
        else:
            pct = 0
            bar = ""
        dist_table.add_row(
            category,
            f"{tokens:,}",
            f"{bar} {pct:.1f}%",
        )

    dist_table.add_section()
    dist_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total:,}[/bold]",
        f"[bold]{result.percentage_used * 100:.1f}% of context window[/bold]",
    )

    console.print(dist_table)
    console.print()

    # Verbose breakdown
    if verbose and result.categories_detail:
        detail = result.categories_detail

        if detail.tools_breakdown.total > 0:
            tools_bd = detail.tools_breakdown
            verbose_content = "[bold]TOOLS BREAKDOWN:[/bold]\n"
            if tools_bd.tool_definitions > 0:
                pct = tools_bd.tool_definitions / tools_bd.total * 100
                verbose_content += f"  Tool Definitions: {tools_bd.tool_definitions:,} tokens ({pct:.1f}%)\n"
            if tools_bd.tool_calls > 0:
                pct = tools_bd.tool_calls / tools_bd.total * 100
                verbose_content += f"  Tool Calls:       {tools_bd.tool_calls:,} tokens ({pct:.1f}%)\n"
            if tools_bd.tool_responses > 0:
                pct = tools_bd.tool_responses / tools_bd.total * 100
                verbose_content += f"  Tool Responses:   {tools_bd.tool_responses:,} tokens ({pct:.1f}%)\n"

        if detail.user_breakdown.images > 0:
            user_bd = detail.user_breakdown
            verbose_content += "\n[bold]USER BREAKDOWN:[/bold]\n"
            if user_bd.text_content > 0:
                pct = user_bd.text_content / user_bd.total * 100
                verbose_content += f"  Text Content:    {user_bd.text_content:,} tokens ({pct:.1f}%)\n"
            if user_bd.images > 0:
                pct = user_bd.images / user_bd.total * 100
                verbose_content += f"  Images (est.):   {user_bd.images:,} tokens ({pct:.1f}%)\n"

        if verbose_content.strip():
            console.print(
                Panel(
                    verbose_content.strip(),
                    title="[bold]Detailed Token Breakdown[/bold]",
                    border_style="dim",
                )
            )
            console.print()

    # Summary panel
    if result.is_in_dumb_zone:
        summary_content = f"""[bold red]! DUMB ZONE REACHED at Round {result.dumb_zone_round} ({result.percentage_used * 100:.1f}%)[/bold red]

[dim]Recommendation: Consider summarizing or pruning context
before continuing the conversation.[/dim]"""
        console.print(
            Panel(
                summary_content,
                border_style="red",
            )
        )
    else:
        tokens_until = result.threshold_tokens - result.total_tokens
        summary_content = f"""[bold green]  Safe[/bold green] - {tokens_until:,} tokens until dumb zone

[dim]Current usage: {result.percentage_used * 100:.1f}% of context window[/dim]"""
        console.print(
            Panel(
                summary_content,
                border_style="green",
            )
        )


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    file: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to JSON file. If omitted, reads from stdin.",
            exists=False,
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Dumb zone threshold (0.0-1.0)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.4,
    context_window: Annotated[
        int,
        typer.Option(
            "--context-window",
            "-c",
            help="Context window size in tokens",
            min=1,
        ),
    ] = 168_000,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Model name for tokenizer selection",
        ),
    ] = None,
    simple: Annotated[
        bool,
        typer.Option(
            "--simple",
            "-s",
            help="Output simple one-line status instead of full report",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Use offline estimation only (no API calls)",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed token breakdown (full report only)",
        ),
    ] = False,
) -> None:
    """
    Analyze a conversation for token usage and dumb zone detection.

    The "dumb zone" is the region of the context window (default: after 40%)
    where model performance may degrade.

    Examples:

        # Full report from a file
        tinyloop ctx conversation.json

        # Simple one-line status
        tinyloop ctx -s conversation.json

        # Pipe from another command
        cat conversation.json | tinyloop ctx

        # Custom threshold and context window
        tinyloop ctx -t 0.3 -c 200000 conversation.json
    """
    # If a subcommand was invoked, skip the default behavior
    if ctx.invoked_subcommand is not None:
        return

    # Show help if no file and stdin is a tty (no piped input)
    if file is None and sys.stdin.isatty():
        console.print(ctx.get_help())
        raise typer.Exit(code=0)

    try:
        # Load conversation
        data = load_conversation(file)

        # Get model from input if not specified
        input_model = data.get("model")
        effective_model = model or input_model

        # Get context window from input if specified
        input_context_window = data.get("context_window")
        if input_context_window and context_window == 168_000:
            # Use input context window only if CLI default wasn't changed
            context_window = input_context_window

        # Create analyzer
        analyzer = CTXAnalyzer(
            model=effective_model,
            context_window=context_window,
            threshold=threshold,
            offline=offline,
        )

        # Analyze
        result = analyzer.analyze(
            messages=data["messages"],
            tools=data.get("tools"),
        )

        # Output
        if simple:
            print(format_simple_output(result))
        else:
            render_full_report(result, verbose=verbose)

        # Exit code based on dumb zone status
        raise typer.Exit(code=1 if result.is_in_dumb_zone else 0)

    except typer.Exit:
        raise
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=3)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
