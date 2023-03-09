import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import subprocess
import atexit
import os 

SSD = False
base_path = "E:\\ML\\" if SSD else ".\\"

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("GUI Launcher") # type: ignore
        self.master.geometry("600x300") # type: ignore
        self.pack(pady=20)
        
        # Model parameter
        models = os.listdir('./src/Models/')
        models = [model.split('.py')[0] for model in models if model.endswith('.py') and model.split('.py')[0][0].isalpha()]

        self.model_label = tk.Label(self, text="Model:")
        self.model_label.grid(row=0, column=0, padx=10)
        self.model_var = tk.StringVar(value="")
        self.model_entry = tk.Entry(self, textvariable=self.model_var)
        self.combo1 = ttk.Combobox(self, values=models, state="readonly")
        if len(models) > 0:
            self.combo1.current(0)  # Set the default value
        self.combo1.grid(row=0, column=1)
        self.combo1.bind("<<ComboboxSelected>>", lambda e: self.model_var.set(self.combo1.get()))
        # Set the first value as the default option
        self.model_var.set(self.combo1.get())

        # Epochs parameter
        self.epochs_label = tk.Label(self, text="Epochs:")
        self.epochs_label.grid(row=1, column=0, padx=10)
        self.epochs_var = tk.StringVar(value="1")
        self.epochs_entry = tk.Entry(self, textvariable=self.epochs_var)
        self.epochs_entry.grid(row=1, column=1)

        # Batch size parameter
        self.batch_size_label = tk.Label(self, text="Batch size:")
        self.batch_size_label.grid(row=2, column=0, padx=10)
        self.batch_size_var = tk.StringVar(value="32")
        self.batch_size_entry = tk.Entry(self, textvariable=self.batch_size_var)
        self.batch_size_entry.grid(row=2, column=1)


        # Checkpoint parameter
        checkpoints = os.listdir(base_path + 'Checkpoints\\')
        checkpoints = [checkpoint.split('.pt')[0] for checkpoint in checkpoints if checkpoint.endswith('.pt')]

        self.checkpoint_label = tk.Label(self, text="Checkpoint:")
        self.checkpoint_label.grid(row=3, column=0, padx=10)
        self.checkpoint_var = tk.StringVar(value="")
        
        self.combo2 = ttk.Combobox(self,textvariable=self.checkpoint_var,values=checkpoints,state="normal")
        if len(checkpoints) > 0:
            self.combo2.current(0)  # Set the default value
        self.combo2.grid(row=3, column=1)
        self.combo2.bind("<<ComboboxSelected>>", lambda e: self.checkpoint_var.set(self.combo2.get()))
        # Set the first value as the default option
        self.checkpoint_var.set(self.combo2.get())
        
        # Number of potential classes
        self.possible_target_label = tk.Label(self, text="Potential targets:")
        self.possible_target_label.grid(row=4, column=0, padx=10)
        self.possible_target_var = tk.StringVar(value="1")
        self.possible_target_entry = tk.Entry(self, textvariable=self.possible_target_var)
        self.possible_target_entry.grid(row=4, column=1)

        # Figures parameter
        self.figures_label = tk.Label(self, text="Figures:")
        self.figures_label.grid(row=5, column=0, padx=10)
        self.figures_var = tk.BooleanVar(value=True)
        self.figures_train = tk.Radiobutton(self, text="On", variable=self.figures_var, value=True)
        self.figures_train.grid(row=5, column=1)
        self.figures_infer = tk.Radiobutton(self, text="Off", variable=self.figures_var, value=False)
        self.figures_infer.grid(row=5, column=2)

        # Dataset serialization parameter
        self.dataset_label = tk.Label(self, text="Transformed Dataset:")
        self.dataset_label.grid(row=6, column=0, padx=10)
        self.dataset_var = tk.BooleanVar(value=False)
        self.dataset_raw = tk.Radiobutton(self, text="Raw", variable=self.dataset_var, value=False)
        self.dataset_raw.grid(row=6, column=1)
        self.dataset_transformed = tk.Radiobutton(self, text="Transformed", variable=self.dataset_var, value=True)
        self.dataset_transformed.grid(row=6, column=2)

        # Run button
        self.run_button = tk.Button(self, text="Run", command=self.run)
        self.run_button.grid(row=7, column=1, pady=20)
        
        # Create subprocess
        self.proc = None

        # Register kill_subprocess function to be called when program is closing down
        atexit.register(self.kill_subprocess)
            
    def kill_subprocess(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def run(self):
        # Get parameter values
        model = self.model_var.get()
        epochs = self.epochs_var.get()
        batch_size = self.batch_size_var.get()
        checkpoint = self.checkpoint_var.get()
        potential_targets = self.possible_target_var.get()
        figures = self.figures_var.get()
        transformed = self.dataset_var.get()

        transformed_message = "Transformed" if transformed else "Raw"
        # Show parameter values
        message = f"Model: {model}\nEpochs: {epochs}\nBatch size: {batch_size}\nCheckpoint: {checkpoint}\nPotential targets: {potential_targets} \nFigures: {figures}\nTransformed: {transformed_message}"
        messagebox.showinfo("Parameter values", message)
        # TODO: Call Run.py with the selected parameters
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Error", "A subprocess is already running.")
            return
        cmd = ["python", ".\\src\\Run.py",
                "--model", model,"--epochs", epochs,
                "--batch_size", batch_size,
                "--checkpoint", checkpoint,
                "--metric", potential_targets,
               ]
        
        cmd += ["--figures"] if figures else ["--no-figures"]
        cmd += ["--transformed"] if transformed else ["--no-transformed"]
        self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

        output = self.proc.communicate()[0]
        print(output.decode('utf-8'))
    
def main():
    root = tk.Tk()
    app = App(master=root)
    app.mainloop()
    
        
if __name__ == "__main__":
    main()
