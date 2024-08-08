# Patchwork

![Demo](demo.mp4)

*Click the image above to see Patchwork in action.*

## What is Patchwork?

Patchwork is your AI-powered development assistant that takes the hassle out of achieving complex development goals. It automates the process, making it smooth and error-free.

## Why Use Patchwork?

Manual development tasks can be tedious and time-consuming, but with Patchwork, it's a breeze! Here's why you'll love it:

1. **AI-Driven Coordination**: Our main agent, `agent_coordinator`, directs everything seamlessly.
2. **Smart Task Generation**: Patchwork uses AI to generate tasks based on your input goal.
3. **Precision Editing**: `agent_editor` creates diff code changes for each task.
4. **Feedback Loop**: For each task, Patchwork applies diff's changes and gives feedback on the success, ensuring everything works smoothly.

## Getting Started

### 1. Install Patchwork

Clone the repository and install the required dependencies.

```sh
git clone https://github.com/yourusername/patchwork.git
cd patchwork
# Create a virtual environment
python -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install the required dependencies
pip install -r requirements.txt
```

### 2. Add Patchwork to PATH
For Unix-like systems, put `patchwork.sh` in your PATH (i.e.: make it executable) from any directory.
```sh
export PATH=$PATH:/path/to/patchwork
```
or, permanently add it to your `.bashrc` or `.zshrc` file.

For Windows, use `patchwork.cmd`:
```cmd
setx PATH "%PATH%;C:\path\to\patchwork"
```
or, permanently add it to your system's environment variables through the System Properties.

Alternatively, you can use a `.env` file.

#### Using a `.env` file
1. Rename `.env.example` to `.env`.
2. Open `.env` and add your API keys:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

### 3. Run Patchwork

Call it from within any Git repository by passing the repo as the first argument.

```sh
cd /path/to/your/git/repo
patchwork.sh .
```
or for Windows:
```cmd
cd /path/to/your/git/repo
patchwork.cmd .
```

### 4. Select the relevant files, and provide a goal

Let Patchwork handle the development process. You'll provide an initial goal, a set of files, and Patchwork will take care of the rest.

## License

Patchwork is licensed under the MIT License. Go nuts if you want, but giving credit is always appreciated.

## Bonus Section: How It Works

Patchwork leverages AI to handle complex development tasks automatically. Here's how it operates:

1. **Identify**: `agent_coordinator` finds the important files based on your goal.
2. **Generate Tasks**: `agent_suggestor` uses AI to create specific tasks for each file.
3. **Edit and Change**: `agent_editor` generates diffs for each task and applies them one by one.
4. **Feedback and Iterate**: It checks the success of each change, making adjustments as needed.

---

With Patchwork, achieving complex development goals is no longer a chore. Let our AI agents take care of the hard work while you relax and enjoy a perfectly developed project. Happy coding! 😄
