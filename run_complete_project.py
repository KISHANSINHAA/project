#!/usr/bin/env python3
"""
Complete script to run the retail forecasting project
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def main():
    print("=========================================")
    print("🏪 RETAIL FORECASTING PROJECT")
    print("=========================================")
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")
    
    print("\nChoose an option:")
    print("1. Run complete project (preprocess, train, dashboard)")
    print("2. Run data preprocessing only")
    print("3. Train models only")
    print("4. Run Streamlit dashboard only")
    print("5. Exit")
    
    choice = input("Enter your choice (1/2/3/4/5): ").strip()
    
    if choice == "1":
        print("\n🔄 Running complete project...")
        # Preprocess data
        if run_command("python src/data_preprocessing.py", "Creating sample dataset..."):
            # Train models
            if run_command("python simple_train.py", "Training models..."):
                # Run dashboard
                print("\n📊 Starting Streamlit dashboard...")
                print("Access the dashboard at: http://localhost:8501")
                run_command("streamlit run app/dashboard.py", "Starting Streamlit dashboard...")
        
    elif choice == "2":
        print("\n🔄 Running data preprocessing...")
        run_command("python src/data_preprocessing.py", "Creating sample dataset...")
        
    elif choice == "3":
        print("\n🤖 Training models...")
        run_command("python simple_train.py", "Training models...")
        
    elif choice == "4":
        print("\n📊 Starting Streamlit dashboard...")
        print("Access the dashboard at: http://localhost:8501")
        run_command("streamlit run app/dashboard.py", "Starting Streamlit dashboard...")
        
    elif choice == "5":
        print("👋 Exiting...")
        sys.exit(0)
        
    else:
        print("Invalid choice. Please run again and select a valid option.")

if __name__ == "__main__":
    main()