import os


def remove_file(filepath: str) -> None:
    """
    Removing a file located at filepath
    """
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        print(f"Error file at path {filepath} not found")