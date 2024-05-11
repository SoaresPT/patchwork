import sys
import re
import os
from git import Repo, InvalidGitRepositoryError
import shutil
import pathspec
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import subprocess

file_cache = {}


def validate_git_repo(path):
    try:
        Repo(path)
    except InvalidGitRepositoryError:
        print(f"Error: {path} is not a valid Git repository.")
        sys.exit(1)


def run(cmd):
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = result.stdout.decode("utf-8").strip()
    stderr = result.stderr.decode("utf-8").strip()
    if result.returncode != 0:
        raise Exception(f"Command '{cmd}' failed with return code {result.returncode}.\nStdout: {stdout}.\nStderr: {stderr}")
    return stdout, stderr

def select_files(directory_path):
    files = os.listdir(directory_path)
    print("Select the files you want to refactor:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    selected_files = input("Enter the numbers of the files you want to refactor, separated by commas: ")
    selected_files = selected_files.split(",")
    file_paths = [os.path.join(directory_path, files[int(i) - 1]) for i in selected_files]
    return file_paths


def get_user_prompt():
    print("Enter a prompt to use for the refactoring. The prompt should describe the refactoring task you want to perform.")
    prompt = input("Prompt: ")
    return prompt


def get_file_contents_and_copy(file_paths):
    temp_dir = "./files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_contents = ""
    file_paths_new = []
    for file_path in file_paths:
        try:
            # Convert the file path to an absolute path
            file_path = os.path.abspath(file_path)
            # Copy each file to the temporary directory
            temp_file_path = shutil.copy(file_path, temp_dir)
            with open(temp_file_path, 'r') as file:
                file_paths_new.append(temp_file_path)
                file_contents += f"Path: {temp_file_path}\n```{os.path.basename(temp_file_path).split('.')[1]}\n{file.read()}\n```\n"
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return file_contents, file_paths_new


def get_gitignore_files(directory_path):
    with open(os.path.join(directory_path, '.gitignore'), 'r') as f:
        gitignore = f.read()
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, gitignore.splitlines())
    all_files = [os.path.join(root, name) for root, dirs, files in os.walk(directory_path) for name in files]
    ignored_files = list(spec.match_files(all_files))
    return ignored_files


def convert_files_dict(file_dict):
    result = {"files": {}}
    for file_path in file_dict["files"]:
        with open(file_path, "r") as file:
            result["files"][file_path] = file.read()
    return result


def extract_codeblocks(text):
    # Find all codeblocks, capture both the content and the language
    codeblocks = re.findall(r'```(?P<language>\w+)\n(?P<content>.*?)[\n\r]```[\n\r]?', text, re.DOTALL)

    # Return a list of codeblocks objects, each with the content and language
    return [{"content": codeblock, "language": language} for language, codeblock in codeblocks]


def select_options(options, all_selected=False):
    """
    Allows the user to interactively select from a list of options.

    :param options: List of options available for selection.
    :param all_selected: If True, all options are selected by default.
    :return: A list of selected indices.
    """
    choices = [{"name": option, "value": idx, "checked": all_selected} for idx, option in enumerate(options)]
    selected_indices = inquirer.checkbox(
        message="Select options:",
        choices=choices,
        default=all_selected,
        validate=lambda result: len(result) > 0,
        invalid_message="You must select at least one option.",
    ).execute()
    return selected_indices


def select_user_files(all_files):
    """
    :param all_files: List of all file paths available for selection.
    :return: A list of selected file paths.
    """
    choices = [{"name": file, "value": file} for file in all_files]
    selected_files = inquirer.checkbox(
        message="Select files or directories:",
        choices=choices,
        validate=lambda result: len(result) >= 0,
        invalid_message="You must select at least one file.",
    ).execute()
    return selected_files
