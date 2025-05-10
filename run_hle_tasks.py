import os
import json
import subprocess
import time
from datasets import load_dataset
from tqdm import tqdm

def main():
    # Load the HLE dataset
    print("Loading HLE dataset...")
    ds = load_dataset("cais/hle", split="test")
    print(f"Loaded {len(ds)} tasks from the dataset")
    
    # Create results directory if it doesn't exist
    os.makedirs("results/HLE", exist_ok=True)
    
    # Process each task in the dataset
    for i, example in enumerate(tqdm(ds, desc="Processing tasks")):
        # Skip tasks with images
        if 'image' in example and example['image']:
            print(f"\nSkipping task {i+1}/{len(ds)} because it contains an image")
            print("-" * 80)
            continue
            
        # Extract the question from the dataset
        if 'question' in example:
            task = example['question']
            print(f"Found question: {task[:50]}..." if len(task) > 50 else f"Found question: {task}")
        else:
            # Fallback to input or the entire example if question doesn't exist
            task = example.get('input', example)
            print("Question field not found, using fallback")
        
        # Convert task to JSON string if it's not already a string
        if not isinstance(task, str):
            task = json.dumps(task)
        
        # Prepare the command
        cmd = f'uv run src/main.py --task "{task}" --verbose True --output_path "results/HLE"'
        
        print(f"\nRunning task {i+1}/{len(ds)}:")
        print(f"Command: {cmd}")
        
        task_id = example['id'] if 'id' in example else f"task_{i+1}"
        start_time = time.time()
        
        try:
            # Execute the command with a longer timeout
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True,
                capture_output=True,
                text=True,
                timeout=600,  # Increased to 10 minutes
            )
            print(f"Task completed successfully in {time.time() - start_time:.2f} seconds.")

        except subprocess.TimeoutExpired as e:
            elapsed_time = time.time() - start_time
            print(f"Task timed out after {elapsed_time:.2f} seconds")
            # Save the timeout error to a file with more details
            with open(f"results/HLE/{task_id}_timeout.txt", "w") as f:
                f.write(f"Task timed out after {elapsed_time:.2f} seconds\n")
                f.write(f"Command: {cmd}\n")
                f.write(f"Partial stdout: {e.stdout if hasattr(e, 'stdout') and e.stdout else 'None'}\n")
                f.write(f"Partial stderr: {e.stderr if hasattr(e, 'stderr') and e.stderr else 'None'}\n")
        
        except subprocess.CalledProcessError as e:
            print(f"Error running task: {e}")
            print(f"Error output: {e.stderr}")
            
            # Save the error to a file with more details
            with open(f"results/HLE/{task_id}_error.txt", "w") as f:
                f.write(f"Command: {cmd}\n")
                f.write(f"Return code: {e.returncode}\n")
                f.write(f"Stdout: {e.stdout}\n")
                f.write(f"Stderr: {e.stderr}\n")
        
        print("-" * 80)
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main() 