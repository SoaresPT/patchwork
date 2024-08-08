#!/usr/bin/env python3
from pdb import run
from InquirerPy import inquirer
import tempfile
import os
import tempfile
from subprocess import call
import re
from openai import OpenAI
import json
from git import Repo
import constants
from utils import get_gitignore_files, get_user_prompt, select_user_files, validate_git_repo, run
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List
import sys
import threading

from loguru import logger
import tqdm.auto as tqdm

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize the OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY or not GEMINI_API_KEY:
    print("Error: Missing API keys. Please set OPENAI_API_KEY and GEMINI_API_KEY in the environment or in a .env file.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Models and their respective token limits
MODEL_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4192,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4-1106-preview": 127514,
    "gpt-4o": 127514,
    "gpt-4": 16384,
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-flash": 1048576,
}

def setup_logging(level="DEBUG", show_module=False):
    """
    Setups better log format for loguru
    """
    logger.remove(0)  # Remove the default logger
    log_level = level
    log_fmt = u"<green>["
    log_fmt += u"{file:10.10}…:{line:<3} | " if show_module else ""
    log_fmt += u"{time:HH:mm:ss.SSS}]</green> <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(lambda x: tqdm.tqdm.write(x, end=""),
               level=log_level,
               format=log_fmt,
               colorize=True,
               backtrace=True,
               diagnose=True)


setup_logging("DEBUG")

class Agent:
    def __init__(self, name, model="gemini-1.5-pro", temperature=0.0):
        self.name = name
        self.system_message = constants.SYSTEM_MESSAGES[name]
        self.model = model
        self.temperature = temperature
        self.gemini_config = {
            "temperature": temperature,
            "max_output_tokens": MODEL_TOKEN_LIMITS[model],
            "response_mime_type": "application/json",
            "response_schema": constants.AGENTS_SCHEMAS[name]
        }

    def send_messages(self, messages, max_attempts=3, temperature=None):
        if "gemini" in self.model:
            model = genai.GenerativeModel(
                model_name=self.model,  # Use the model specified
                generation_config=self.gemini_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                },
                system_instruction=self.system_message + "\n\nReturn a JSON and always escape newlines, tabs, quotes and other special characters.",
            )
            history = [{"role": m["role"], "parts": m["content"]} for m in messages]
            response = model.generate_content(history, stream=True)  # type: ignore
        else:
            response = client.chat.completions.create(model=self.model,
                                                      messages=[
                                                          {
                                                              "role": "system",
                                                              "content": self.system_message + "\n\nReturn always a JSON."}] + messages,
                                                      temperature=temperature if temperature else self.temperature,
                                                      max_tokens=4096,
                                                      response_format={"type": "json_object"},
                                                      stream=True)

        output = []
        attempts = 0
        for chunk in response:
            token = "".join(
                [part.text for part in chunk.candidates[0].content.parts]).replace("\n", "\\n") if "gemini" in self.model else chunk.choices[0].delta.content
            if token is not None:
                output.append(token)
                print(token, end="", flush=True)
            else:
                attempts += 1
                if attempts >= max_attempts:
                    logger.error(f"Error: Agent {self.name} did not provide a valid response after {max_attempts} attempts.")
                    break
        print("\n")

        output = json.loads(response.text) if "gemini" in self.model else json.loads("".join(output))
        return output


class Coordinator:
    def __init__(self, coordinator_agent: Agent, agents: List[Agent], directory_path: str):
        self.coordinator_agent = coordinator_agent
        self.suggestor_agent = agents[0]
        self.editor_agent = agents[1]
        self.checker_agent = agents[2]
        self.repo = Repo(directory_path)

    def run(self, user_prompt, filepaths, max_attempts=3):
        project_files = self.get_file_contents(filepaths)
        user_input = json.dumps({"files": project_files, "prompt": user_prompt})
        coordinator_output = self.coordinator_agent.send_messages([{"role": "user", "content": user_input}])

        goal_files = self.get_file_contents(coordinator_output['filepaths'])
        suggestor_input = json.dumps({
            "files": goal_files, "goal": user_prompt, "user_prompt": user_prompt})
        suggestor_output = self.suggestor_agent.send_messages([{"role": "user", "content": suggestor_input}])

        tasks = suggestor_output['tasks']
        for task in tasks:
            patches, changed_files = self.get_changes(task)
            self.apply_changes(task, patches, changed_files)

    def replace_file(self, file, content):
        try:
            file_path = os.path.join(self.repo.working_dir, file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            logger.success(f"Replaced content of file {file} successfully.")
        except Exception as e:
            logger.error(f"Error replacing file {file}: {e}")

    def apply_changes(self, task, fixed_patches, full_file_contents, skip_check=False):
        patches = [self.prepare_patch_for_git(patch) for patch in fixed_patches]
        if len(full_file_contents) > 0:
            for file, content in full_file_contents.items():
                self.replace_file(file, content)
        if len(patches) > 0:
            while patches:
                patch = patches.pop(0)
                try:
                    self.apply_patch(patch)
                except Exception as e:
                    action = None

                    def handle_patch_failure():
                        global action
                        choices = ["Retry", "Edit", "Skip"]
                        action = inquirer.select(
                            message="Patch application failed. Choose an action or wait 3 seconds to auto-retry:",
                            choices=choices
                        ).execute()

                    thread = threading.Thread(target=handle_patch_failure, daemon=True)
                    thread.start()
                    thread.join(3)
                    if action == "Retry":
                        fixed_patches, fixed_files = self.get_changes(task, error=e)
                        self.apply_changes(task, fixed_patches, fixed_files, skip_check=True)
                    if action == "Edit":
                        edited_patch = self.edit_patch(patch)
                        patches.insert(0, edited_patch)  # Insert the edited patch at the beginning of the list
                    elif action == "Skip":
                        continue
                else:
                    logger.success(f"Patch applied successfully.")
        if not skip_check:
            self.check_task(task)

    def check_task(self, task):
        content = self.get_file_contents(task["filepaths"])
        checker_input = json.dumps({"files": content, "task": task})
        checker_output = self.checker_agent.send_messages([{"role": "user", "content": checker_input}])
        if checker_output.get('tasks'):
            for new_task in checker_output['tasks']:
                patches, changed_files = self.get_changes(new_task)
                self.apply_changes(new_task, patches, changed_files, skip_check=True)
        else:
            logger.info("No new tasks from checker agent.")

    def edit_patch(self, patch):  # Open the temporary file in the default editor
        editors = ["subl", "subl3", "code", os.environ.get('EDITOR')]
        editor = [editor for editor in editors if os.path.exists(f"/usr/bin/{editor}" or f"/usr/local/bin/{editor}")][0]
        call([editor, "-w", patch])
        return patch

    def get_file_contents(self, file_paths):
        files_dict = {}

        for file_path in file_paths:
            try:
                absolute_path = ""
                if os.path.exists(os.path.join(self.repo.working_dir, file_path)):
                    # Check if the file exists at the root of the git repository
                    absolute_path = os.path.abspath(os.path.join(self.repo.working_dir, file_path))
                elif os.path.exists(file_path):
                    # Check if the file exists in the current directory
                    absolute_path = os.path.abspath(file_path)

                relative_path = os.path.relpath(absolute_path, self.repo.working_dir)

                if not os.path.exists(absolute_path):
                    logger.warning(f"Error: File {absolute_path} does not exist. Will create a new empty file.")
                    with open(absolute_path, 'w') as file:
                        file.write("")
                        files_dict[relative_path] = ""
                    continue
                else:
                    with open(absolute_path, 'r') as file:
                        files_dict[relative_path] = file.read()
                    logger.trace(f"Found file: {relative_path}. File size: {os.path.getsize(absolute_path)} bytes.")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        logger.info(f"Selected {len(file_paths)} files. Total size: {sum([os.path.getsize(file) for file in file_paths])/1024:.2f} kB.")

        return files_dict

    def apply_patch(self, patch):
        if os.path.getsize(patch) == 0:
            logger.warning(f"Patch file {patch} is empty. Skipping application.")
            return

        cmd = f"git apply --recount --verbose --unidiff-zero -C0 --reject --ignore-space-change --ignore-whitespace --whitespace=fix {patch}"  # --inaccurate-eof
        run(cmd)
        logger.success("Patch applied successfully.")

    def handle_task_selection(self, tasks):
        tasks_descriptions = [task["prompt"] for task in tasks]
        selected_indices = inquirer.checkbox(
            message="Please select the refactoring tasks you want to proceed with:",
            choices=[{"name": desc, "value": idx} for idx, desc in enumerate(tasks_descriptions)]
        ).execute()
        return [tasks[i] for i in selected_indices]

    def get_changes(self, task, error=None, temperature=None):
        prompt = f'{task["prompt"]}\n{task["info"]}'
        files = self.get_file_contents(task["filepaths"])

        if error:
            prompt += f"\n\nThere was an error while applying the this patch: {error}. Please create a new patch to retry the failed hunks. Break the patch into more hunks of smaller size, even if contexts overlap."

        message = json.dumps({"files": files, "prompt": prompt})
        editor_messages = [{"role": "user", "content": message}]
        editor_output = self.editor_agent.send_messages(editor_messages, temperature=temperature)

        patches = []
        files = []
        if "patches" in editor_output:
            patches = editor_output['patches']

            # further split patches at the diff hunk boundary
            all_patches = []
            for patch in patches:
                splits = patch.split("diff --git")
                if len(splits) > 1:
                    for split in splits[1:]:
                        all_patches.append("diff --git" + split)
                else:
                    all_patches.append(patch)

            patches = all_patches

        if "files" in editor_output:
            files = editor_output['files']
        logger.info(f"{len(patches)} patches and {len(files)} files for this task")
        return patches, files

    def prepare_patch_for_git(self, raw_patch):
        patch = raw_patch
        normalized_repo_path = os.path.abspath(self.repo.working_dir).rstrip('/').lstrip('/')
        escaped_repo_base_path = re.escape(normalized_repo_path)

        new_patch = re.sub(escaped_repo_base_path + "/", "", patch, flags=re.MULTILINE)

        normalized_repo_path_segments = normalized_repo_path.split('/')
        for i in range(len(normalized_repo_path_segments), 0, -1):
            partial_path = "/".join(normalized_repo_path_segments[:i])
            escaped_partial_path = re.escape(partial_path)
            new_patch = re.sub(rf"{escaped_partial_path}/", "", new_patch, flags=re.MULTILINE)

        if new_patch == patch:
            logger.info(f"Patch filepaths were not modified.")
        else:
            logger.warning(f"Patch filepaths were modified. Stripped the following from the patch file:\n{escaped_repo_base_path}")

        new_patch = re.sub(r"^([^ +-])", r" \1", new_patch, flags=re.MULTILINE)

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(new_patch)
            temp_file_name = temp_file.name

        return temp_file_name

    def get_fixed_patch(self, files_contents, patch, error):
        user_input = files_contents + "\n\n"
        message = f"The following patch failed to apply:\n```patch\n{patch}\n```\n\nThe error was: {error}. Please return a new patch."
        logger.info(message)
        user_input += message

        text, codeblocks = self.editor_agent.send_messages([{"role": "user", "content": user_input}])

        patch = [
            codeblock["content"]
            for codeblock in codeblocks
            if codeblock["language"] == "patch" or codeblock["language"] == "diff"][0]
        logger.info(f"Got a patch for fixing the previous patch:\n```patch\n{patch}\n```")
        return patch


def main():
    if len(sys.argv) < 2:
        print("Usage: script.py <directory_path> [file1 file2 ...]")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.exists(directory_path):
        print(f"Error: {directory_path} does not exist.")
        sys.exit(1)

    validate_git_repo(directory_path)
    ignored_files = get_gitignore_files(directory_path)

    if len(sys.argv) > 2:
        # Specific files are provided as arguments
        file_paths = [
            os.path.join(directory_path, file)
            for file in sys.argv[2:]
            if os.path.isfile(os.path.join(directory_path, file))]
    else:
        # No specific files provided, use all files not in .gitignore
        file_paths = [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, file)) and os.path.join(directory_path, file) not in ignored_files]

    selected_files = select_user_files(file_paths) if len(sys.argv) == 2 else file_paths
    user_prompt = get_user_prompt()

    models = list(MODEL_TOKEN_LIMITS.keys())
    model_choice = inquirer.select(
        message="Choose the model you want to use:",
        choices=models
    ).execute()

    agent_coordinator = Agent(name="agent_coordinator", model=model_choice, temperature=0)
    agent_suggestor = Agent(name="agent_suggestor", model=model_choice, temperature=0)
    agent_editor = Agent(name="agent_editor", model=model_choice, temperature=0.5)
    agent_checker = Agent(name="agent_checker", model=model_choice, temperature=0)

    coordinator = Coordinator(agent_coordinator,
                              agents=[agent_suggestor, agent_editor, agent_checker],
                              directory_path=directory_path)
    coordinator.run(user_prompt, selected_files)

    print("The refactoring was successful. The source code files have been updated with the patches.")


if __name__ == "__main__":
    main()