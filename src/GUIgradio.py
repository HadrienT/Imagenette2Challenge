import gradio as gr
import os
import subprocess
from typing import Optional
import atexit
import time


def calculate_progress(output):
    epoch, total_epochs, batch, total_batches = output.split("/")
    epoch_progress = int(epoch) / int(total_epochs)
    batch_progress = (int(epoch) * int(total_batches) + int(batch)) / (int(total_epochs) * int(total_batches))
    progress = (epoch_progress + batch_progress) / 2
    return progress


def run_model(model, epochs, batch_size, checkpoint, potential_targets, figures, transformed, progress=gr.Progress()):
    global proc
    transformed_message = "Transformed" if transformed else "Raw"
    # Show parameter values
    message = (f"Model: {model}\n"
               f"Epochs: {epochs}\n"
               f"Batch size: {batch_size}\n"
               f"Checkpoint: {checkpoint}\n"
               f"Potential targets: {potential_targets}\n"
               f"Figures: {figures}\n"
               f"Transformed: {transformed_message}"
               )

    if proc and proc.poll() is None:
        return message + "\n\nSubprocess already running"

    cmd = ["python", "./src/Run.py",
           "--model", model, "--epochs", str(epochs),
           "--batch_size", str(batch_size),
           "--checkpoint", checkpoint,
           "--metric", str(potential_targets),
           ]

    cmd += ["--figures"] if figures else ["--no-figures"]
    cmd += ["--transformed"] if transformed else ["--no-transformed"]
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    # Continually read from the subprocess's output and update the progress bar
    while True:
        output = proc.stdout.readline()
        if output == b'' and proc.poll() is not None:
            break
        if output:
            progress.update(calculate_progress(output.decode().strip()))

        time.sleep(0.1)

    if proc.wait() == 0:
        # subprocess completed successfully
        output = proc.communicate()[0]
        return output.decode('utf-8')
    else:
        # subprocess creation or execution failed
        return "Error creating or running subprocess"


def kill_subprocess() -> None:
    global proc
    if proc and proc.poll() is None:
        proc.terminate()


if __name__ == "__main__":
    proc: Optional[subprocess.Popen[bytes]] = None
    atexit.register(kill_subprocess)
    models = os.listdir('./src/Models/')
    models = [model.split('.py')[0] for model in models if model.endswith('.py') and model.split('.py')[0][0].isalpha()]
    checkpoints = os.listdir('E:/ML/Checkpoints/')
    checkpoints = [checkpoint.split('.pt')[0] for checkpoint in checkpoints if checkpoint.endswith('.pt')]

    iface = gr.Interface(
        fn=run_model,
        inputs=[
            gr.components.Dropdown(models, label="Model"),
            gr.components.Slider(minimum=1, maximum=100, value=1, step=1, label="Epochs"),
            gr.components.Slider(minimum=1, maximum=512, value=32, step=1, label="Batch size"),
            gr.components.Dropdown(checkpoints, label="Checkpoint"),
            gr.components.Slider(minimum=1, maximum=10, value=1, step=1, label="Potential targets"),
            gr.components.Radio(["On", "Off"], value="Off", label="Figures"),
            gr.components.Radio(["Raw", "Transformed"], value="Raw", label="Transformed Dataset"),
        ],
        outputs="text",
        title="GUI Launcher",
        description="Run your machine learning model with specific parameters.",
    )

    iface.launch(queue=True)
