#!/usr/bin/env python3
"""
Startup script that ensures data is prepared and models are trained before starting the dashboard
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"   ✅ Success")
        if result.stdout:
            print(f"   Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed with error: {e}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        return False

def main():
    print("=========================================")
    print("🏪 RETAIL FORECASTING APP STARTUP")
    print("=========================================")
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")
    
    # Check if models exist, if not, run preprocessing and training
    if not os.path.exists("artifacts/best_model.pkl"):
        print("📦 Models not found. Preparing data and training models...")
        
        # Run data preprocessing
        if not run_command("python src/data_preprocessing.py", "Running data preprocessing..."):
            print("❌ Failed to preprocess data")
            sys.exit(1)
        
        # Run model training
        if not run_command("python simple_train.py", "Training models..."):
            print("❌ Failed to train models")
            sys.exit(1)
        
        print("✅ Data preparation and model training completed!")
    else:
        print("✅ Models already exist. Skipping data preparation and training.")
    
    # Start the Streamlit dashboard
    print("🚀 Starting Streamlit dashboard...")
    os.execvp("streamlit", ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    main()