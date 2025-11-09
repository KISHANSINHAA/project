#!/usr/bin/env python3
"""
Simple script to run the complete retail forecasting project
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
        if result.stdout:
            print(f"Output: {result.stdout}")
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
    print("1. Run data preprocessing")
    print("2. Run Streamlit dashboard")
    print("3. Run both")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        print("\n🔄 Running data preprocessing...")
        run_command("python src/data_preprocessing.py", "Creating sample dataset...")
        
    elif choice == "2":
        print("\n📊 Starting Streamlit dashboard...")
        print("Access the dashboard at: http://localhost:8502")
        run_command("streamlit run app/dashboard.py", "Starting Streamlit dashboard...")
        
    elif choice == "3":
        print("\n🔄 Running data preprocessing...")
        if run_command("python src/data_preprocessing.py", "Creating sample dataset..."):
            print("\n📊 Starting Streamlit dashboard...")
            print("Access the dashboard at: http://localhost:8502")
            run_command("streamlit run app/dashboard.py", "Starting Streamlit dashboard...")
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")

if __name__ == "__main__":
    main()