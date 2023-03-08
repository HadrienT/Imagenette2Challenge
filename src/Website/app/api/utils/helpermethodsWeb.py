import multiprocessing
import os
import sys

# Add the parent directory of the current file (i.e., root directory) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..\\..\\..')))
import Infer
def infer()->list:
    # Create a multiprocessing Queue object
    queue = multiprocessing.Queue()
    
    # Start the second process
    p2 = multiprocessing.Process(target=Infer.main, args=(queue,))
    p2.start()

    # Wait for the second process to finish
    p2.join()

    # Get the result from the Queue
    return queue.get()

def empty_folder(dir_path):
    # loop through the files in the directory and remove them
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')    
    