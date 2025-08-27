"""
Full ML Pipeline Runner for Cinematch AI
Orchestrates the complete machine learning workflow
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"[v0] {description}")
    print(f"[v0] Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[v0] ✅ {description} completed successfully in {duration:.2f} seconds")
        
        # Print output if available
        if result.stdout:
            print("[v0] Script output:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[v0] ❌ Error in {description}:")
        print(f"[v0] Return code: {e.returncode}")
        if e.stdout:
            print("[v0] STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("[v0] STDERR:")
            print(e.stderr)
        return False
    
    except Exception as e:
        print(f"[v0] ❌ Unexpected error in {description}: {str(e)}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("[v0] Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[v0] ✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"[v0] ❌ {package} is missing")
    
    if missing_packages:
        print(f"\n[v0] Missing packages: {', '.join(missing_packages)}")
        print("[v0] Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("[v0] All dependencies are available!")
    return True

def create_pipeline_summary():
    """Create a summary of the pipeline execution"""
    print("\n[v0] Creating pipeline summary...")
    
    summary = {
        'pipeline_execution': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'completed',
            'stages': [
                'Data Preprocessing',
                'Neural Network Training', 
                'Model Evaluation'
            ]
        },
        'outputs_generated': [],
        'next_steps': [
            'Upload your own movie CSV file',
            'Adjust model hyperparameters if needed',
            'Deploy model for real-time recommendations',
            'Integrate with web application'
        ]
    }
    
    # Check which output files were created
    output_files = [
        'processed_movie_data.json',
        'cinematch_model.h5',
        'training_results.json',
        'training_history.png',
        'model_evaluation_results.json',
        'sample_recommendations.json'
    ]
    
    for file in output_files:
        if Path(file).exists():
            summary['outputs_generated'].append(file)
    
    # Save summary
    with open('pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("[v0] Pipeline summary saved to 'pipeline_summary.json'")
    return summary

def main():
    """Main pipeline orchestrator"""
    print("🎬 CINEMATCH AI - ML PIPELINE RUNNER 🎬")
    print("=" * 60)
    print("[v0] Starting complete machine learning pipeline...")
    
    start_time = time.time()
    
    # Step 0: Check dependencies
    if not check_dependencies():
        print("[v0] ❌ Dependency check failed. Please install missing packages.")
        return
    
    # Pipeline stages
    stages = [
        {
            'script': 'data_preprocessing.py',
            'description': 'Data Preprocessing & Feature Engineering',
            'required': True
        },
        {
            'script': 'neural_network_training.py', 
            'description': 'Neural Network Training',
            'required': True
        },
        {
            'script': 'model_evaluation.py',
            'description': 'Model Evaluation & Demonstration',
            'required': False
        }
    ]
    
    # Execute pipeline stages
    successful_stages = 0
    total_stages = len(stages)
    
    for i, stage in enumerate(stages, 1):
        print(f"\n[v0] 📊 STAGE {i}/{total_stages}: {stage['description']}")
        
        success = run_script(stage['script'], stage['description'])
        
        if success:
            successful_stages += 1
        elif stage['required']:
            print(f"[v0] ❌ Required stage failed: {stage['description']}")
            print("[v0] Pipeline execution stopped.")
            return
        else:
            print(f"[v0] ⚠️  Optional stage failed: {stage['description']}")
            print("[v0] Continuing with pipeline...")
    
    # Pipeline completion
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("🎉 PIPELINE EXECUTION COMPLETED! 🎉")
    print(f"{'='*60}")
    print(f"[v0] Total execution time: {total_duration:.2f} seconds")
    print(f"[v0] Successful stages: {successful_stages}/{total_stages}")
    
    # Create summary
    summary = create_pipeline_summary()
    
    print(f"\n[v0] 📁 Generated files:")
    for file in summary['outputs_generated']:
        print(f"[v0]   - {file}")
    
    print(f"\n[v0] 🚀 Next steps:")
    for step in summary['next_steps']:
        print(f"[v0]   - {step}")
    
    print(f"\n[v0] 🎬 Your Cinematch AI model is ready!")
    print("[v0] You can now use it to get personalized movie recommendations!")

if __name__ == "__main__":
    main()
