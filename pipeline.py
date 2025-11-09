#!/usr/bin/env python3
"""
Retail Forecasting Pipeline
==========================

This script orchestrates the complete retail forecasting workflow:
1. Data preprocessing
2. Feature engineering
3. Model training
4. Model evaluation
5. Prediction generation

Usage:
    python pipeline.py
"""

import os
import sys
import subprocess
import time
import argparse

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🚀 {description}")
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
    parser = argparse.ArgumentParser(description='Retail Forecasting Pipeline')
    parser.add_argument('--steps', nargs='+', 
                       choices=['preprocess', 'train', 'predict', 'dashboard', 'api'],
                       default=['preprocess', 'train'],
                       help='Select which steps to run')
    parser.add_argument('--dashboard', action='store_true',
                       help='Start the Streamlit dashboard after training')
    parser.add_argument('--api', action='store_true',
                       help='Start the FastAPI server after training')
    
    args = parser.parse_args()
    
    print("=========================================")
    print("🏪 RETAIL FORECASTING PIPELINE")
    print("=========================================")
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")
    
    # Step 1: Data Preprocessing
    if 'preprocess' in args.steps:
        print("\n🔄 STEP 1: DATA PREPROCESSING")
        print("="*50)
        if not run_command("python src/data_preprocessing.py", "Running data preprocessing..."):
            print("❌ Pipeline failed at data preprocessing step")
            sys.exit(1)
        time.sleep(2)
    
    # Step 2: Model Training
    if 'train' in args.steps:
        print("\n🤖 STEP 2: MODEL TRAINING")
        print("="*50)
        if not run_command("python src/train_model.py", "Training models..."):
            print("❌ Pipeline failed at model training step")
            sys.exit(1)
        time.sleep(2)
    
    # Step 3: Generate Predictions (example)
    if 'predict' in args.steps:
        print("\n🔮 STEP 3: GENERATING PREDICTIONS")
        print("="*50)
        print("   Note: Run predictions through dashboard or API")
        time.sleep(1)
    
    # Step 4: Start Dashboard
    if args.dashboard or 'dashboard' in args.steps:
        print("\n📊 STEP 4: STARTING DASHBOARD")
        print("="*50)
        print("   Starting Streamlit dashboard...")
        print("   Access at: http://localhost:8501")
        run_command("streamlit run app/dashboard.py", "Starting dashboard...")
    
    # Step 5: Start API
    if args.api or 'api' in args.steps:
        print("\n🔌 STEP 5: STARTING API SERVER")
        print("="*50)
        print("   Starting FastAPI server...")
        print("   Access at: http://localhost:8000")
        print("   API docs at: http://localhost:8000/docs")
        run_command("uvicorn app.api:app --reload", "Starting API server...")
    
    print("\n=========================================")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=========================================")

if __name__ == "__main__":
    main()