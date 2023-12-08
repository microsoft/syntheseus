import sys

from syntheseus.cli import eval_single_step, search


def main() -> None:
    supported_commands = {"search": search.main, "eval-single-step": eval_single_step.main}
    supported_command_names = ", ".join(supported_commands.keys())

    if len(sys.argv) == 1:
        print(f"Please choose a command from: {supported_command_names}")
        return

    command = sys.argv[1]
    if command not in supported_commands:
        print(f"Command {command} not supported; choose from: {supported_command_names}")
        return

    # Drop the subcommand name and let the chosen command parse the rest of the arguments.
    del sys.argv[1]
    supported_commands[command]()


if __name__ == "__main__":
    main()
