import os
import json
import argparse
import subprocess
from datasets import load_dataset
from pathlib import Path

def load_travel_planner_data(set_type="validation"):
    """Load TravelPlanner dataset"""
    data = load_dataset('osunlp/TravelPlanner', set_type)[set_type]
    return data

def format_task_for_agent(query_data):
    """Format TravelPlanner query into a task for your agent system"""
    # Extract relevant information from the query
    query = query_data["query"]
    
    # Create a structured task prompt
    task = f"""
You are a travel planning assistant. Please create a detailed travel plan based on the following request:

{query}

Your plan should include:
1. Transportation arrangements
2. Daily meals (breakfast, lunch, dinner)
3. Attractions to visit each day
4. Accommodation for each night

Make sure your plan follows all constraints mentioned in the request and adheres to common sense (e.g., reasonable travel times, appropriate meal times, etc.).

The TravelPlanner database is located at agentLibrary/TravelPlanner. You can use this database to find information about:
- Flights (database/flights/)
- Accommodations (database/accommodations/)
- Attractions (database/attractions/)
- Restaurants (database/restaurants/)
- Cities (database/background/)
"""
    return task

def save_results(results, output_path, set_type, model_name, strategy="multi-agent"):
    """Save results in the format expected by TravelPlanner evaluation"""
    os.makedirs(output_path, exist_ok=True)
    
    # Format filename according to TravelPlanner conventions
    if strategy == "multi-agent":
        filename = f"{set_type}_{model_name}_multi-agent.jsonl"
    else:
        filename = f"{set_type}_{model_name}_{strategy}.jsonl"
    
    output_file = os.path.join(output_path, filename)
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_file}")
    return output_file

def run_task_with_system(task, trace_dir, query_id):
    """Run a task using your system's command structure"""
    # Create the trace directory if it doesn't exist
    os.makedirs(trace_dir, exist_ok=True)
    
    # Format the command
    cmd = f'uv run src/main.py --task "{task}" --verbose True --output_path "{trace_dir}/{query_id}"'
    
    # Run the command
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running task: {e}")
        print(f"Stderr: {e.stderr}")
        return f"Error: {e}"

def load_example_submission():
    """Load example submission to get query IDs"""
    example_path = "TravelPlanner/example_submission.jsonl"
    if not os.path.exists(example_path):
        print(f"Warning: Example submission file not found at {example_path}")
        return []
    
    examples = []
    with open(example_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--trace_path", type=str, default="results/traces/travel_planner")
    args = parser.parse_args()
    
    # Load TravelPlanner dataset
    data = load_travel_planner_data(args.set_type)
    
    # Load example submission to get query IDs
    example_submission = load_example_submission()
    
    results = []
    
    # Process each query in the dataset
    for i, query_data in enumerate(data):
        print(f"Processing query {i+1}/{len(data)}")
        
        # Get query ID from example submission if available
        query_id = f"query_{i}"
        if i < len(example_submission):
            query_id = example_submission[i].get("id", query_id)
        
        # Format the task for your agent system
        task = format_task_for_agent(query_data)
        
        # Run your multi-agent system
        trace_dir = os.path.join(args.trace_path, query_id)
        response = run_task_with_system(task, trace_dir, query_id)
        
        # Store the result
        result = {
            "id": query_data["id"],
            "query": query_data["query"],
            "response": response
        }
        results.append(result)
        
        # Save intermediate results after each query
        intermediate_file = save_results(results[:i+1], args.output_dir, args.set_type, args.model_name)
        print(f"Intermediate results saved to {intermediate_file}")
    
    # Save final results
    output_file = save_results(results, args.output_dir, args.set_type, args.model_name)
    
    print(f"All queries processed. Results saved to {output_file}")
    print("Next steps:")
    print("1. Run the parsing script to convert natural language plans to JSON")
    print("2. Run the element extraction script")
    print("3. Run the combination script to prepare for evaluation")
    print("4. Run the evaluation script")

if __name__ == "__main__":
    main()