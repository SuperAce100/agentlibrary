import os
print("imported os")
import json
print("imported json")
import subprocess
print("imported subprocess")
import time
print("imported time")
from datasets import load_dataset
print("imported load_dataset")
from tqdm import tqdm
print("imported tqdm")
from results_eval import evaluate_final_response
print("imported evaluate_final_response")
from concurrent.futures import ThreadPoolExecutor
print("imported ThreadPoolExecutor")
from main import run
print("imported run")

def main():
    # Load the HLE dataset
    print("Loading HLE dataset...")
    ds = load_dataset("cais/hle", split="test")
    print(f"Loaded {len(ds)} tasks from the dataset")

    results_directory = ".data/HLE_results/HLE"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_directory, exist_ok=True)
    print(f"Results directory absolute path: {os.path.abspath(results_directory)}")
    print(f"Directory exists after creation: {os.path.exists(results_directory)}")
    
    # Initialize score tracking
    total_score = 0
    completed_tasks = 0
    
    # Initialize results log file
    results_log_path = os.path.join(results_directory, "results_log.json")
    
    # Load existing results if the file exists
    if os.path.exists(results_log_path):
        try:
            with open(results_log_path, 'r') as f:
                results_log = json.load(f)
        except json.JSONDecodeError:
            # If the file exists but is corrupted, start fresh
            results_log = {"correct_answers": [], "total_correct": 0}
    else:
        results_log = {"correct_answers": [], "total_correct": 0}
    
    # Process each task in the dataset using a thread pool for parallelization
    def process_task(task_info):
        i, example = task_info
        task_id = f"task_{i}"  # Define task_id
        start_time = time.time()  # Define start_time
        
        # Skip tasks with images
        if 'image' in example and example['image']:
            print(f"\nSkipping task {i+1}/{len(ds)} because it contains an image")
            print("-" * 80)
            return None
            
        # Extract the question and answer from the dataset
        if 'question' in example:
            task = example['question']
            print(f"Found question: {task[:50]}..." if len(task) > 50 else f"Found question: {task}")
            
            # Extract answer if available
            answer = example.get('answer', 'No answer provided')
            print(f"Found answer: {answer[:50]}..." if len(answer) > 50 else f"Found answer: {answer}")
        else:
            # Fallback to input or the entire example if question doesn't exist
            task = example.get('input', example)
            print("Question field not found, using fallback")
            answer = 'No answer provided'
        
        if not isinstance(task, str):
            task = json.dumps(task)
        
        try:
            final_response = run(task, 100, verbose=False, trace_path=results_directory)
            score_value = evaluate_final_response(final_response, answer)
            
            # Convert the raw score to a dictionary with the expected structure
            result = {
                'completed': True,
                'score': int(score_value.strip()) if score_value.strip() in ['0', '1'] else 0,
                'task_id': task_id,
                'question': task[:200] + "..." if len(task) > 200 else task  # Include truncated question
            }
            
            return result
        
        except subprocess.TimeoutExpired as e:
            elapsed_time = time.time() - start_time
            print(f"Task timed out after {elapsed_time:.2f} seconds")
            # Save the timeout error to a file with more details
            with open(f"{results_directory}/{task_id}_timeout.txt", "w") as f:
                f.write(f"Task timed out after {elapsed_time:.2f} seconds\n")
                f.write(f"Question: {task}\n")
                f.write(f"Correct answer: {answer}\n")
            
            return {'completed': False, 'score': 0, 'task_id': task_id}
        
        except subprocess.CalledProcessError as e:
            print(f"Error running task: {e}")
            if hasattr(e, 'stderr'):
                print(f"Error output: {e.stderr}")
            
            # Save the error to a file with more details
            with open(f"{results_directory}/{task_id}_error.txt", "w") as f:
                if hasattr(e, 'returncode'):
                    f.write(f"Return code: {e.returncode}\n")
                if hasattr(e, 'stdout'):
                    f.write(f"Stdout: {e.stdout}\n")
                if hasattr(e, 'stderr'):
                    f.write(f"Stderr: {e.stderr}\n")
                f.write(f"Question: {task}\n")
                f.write(f"Correct answer: {answer}\n")
            
            return {'completed': False, 'score': 0, 'task_id': task_id}
        
        print("-" * 80)
        return final_response
    
    max_workers = 8
    print(f"Using {max_workers} worker threads for parallel execution")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect futures
        futures = [executor.submit(process_task, (i, example)) for i, example in enumerate(ds)]
        
        # Process results as they complete
        for future in tqdm(futures, desc="Waiting for tasks to complete"):
            result = future.result()
            if result and result.get('completed'):
                total_score += result['score']
                completed_tasks += 1
                
                # Log correct answers to the results log
                if result['score'] == 1:
                    results_log["correct_answers"].append({
                        "task_id": result['task_id'],
                        "question": result.get('question', 'Question not available')
                    })
                    results_log["total_correct"] += 1
                    
                    # Update the log file after each correct answer
                    with open(results_log_path, 'w') as f:
                        json.dump(results_log, f, indent=2)
                    print(f"File exists after writing: {os.path.exists(results_log_path)}")
    
    # Print final score summary
    if completed_tasks > 0:
        print(f"\nFinal score: {total_score}/{completed_tasks} = {total_score/completed_tasks:.2f}")
    else:
        print("\nNo tasks were successfully completed and evaluated.")
    
    # Final update to the results log
    results_log["completion_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results_log["final_score"] = {
        "total_correct": total_score,
        "total_completed": completed_tasks,
        "accuracy": total_score/completed_tasks if completed_tasks > 0 else 0
    }
    
    with open(results_log_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    
    print(f"\nResults log saved to {results_log_path}")
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main() 