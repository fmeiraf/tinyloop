"""
TinyLoop CLI entry point.

Provides the main `tinyloop` command with subcommands.
"""

import typer

from tinyloop.ctx.cli import app as ctx_app

app = typer.Typer(
    name="tinyloop",
    help="TinyLoop - A super lightweight library for LLM-based applications",
    no_args_is_help=True,
)

# Add ctx subcommand
app.add_typer(ctx_app, name="ctx")


@app.command()
def version() -> None:
    """Show the version of TinyLoop."""
    from tinyloop import __version__

    typer.echo(f"TinyLoop version {__version__}")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
