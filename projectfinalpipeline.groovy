pipeline {
    agent any
    
    tools {
        // Configure your Python tool here if needed
        // python "Python 3.9"
    }
    
    environment {
        // Define environment variables
        PROJECT_NAME = "projectfinalpipeline"
        DOCKER_IMAGE = "retail-forecasting-app"
        DOCKER_TAG = "latest"
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/KISHANSINHAA/project.git'
            }
        }
        
        stage('Setup Environment') {
            steps {
                script {
                    // Create virtual environment and install dependencies
                    sh 'python -m venv venv'
                    sh 'venv/bin/pip install --upgrade pip'
                    sh 'venv/bin/pip install -r requirements.txt'
                }
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                script {
                    sh 'python src/data_preprocessing.py'
                }
            }
        }
        
        stage('Model Training') {
            steps {
                script {
                    sh 'python simple_train.py'
                }
            }
        }
        
        stage('Test Dashboard') {
            steps {
                script {
                    echo 'Testing dashboard functionality...'
                    // Add dashboard test commands here
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .'
                }
            }
        }
        
        stage('Deploy to Streamlit Cloud') {
            steps {
                script {
                    echo 'Deploying to Streamlit Cloud...'
                    // Add Streamlit Cloud deployment steps here
                    // This would typically involve using Streamlit Cloud's GitHub integration
                }
            }
        }
        
        stage('Deploy to Jenkins Server') {
            steps {
                script {
                    echo 'Deploying to Jenkins server at http://localhost:8081...'
                    // Add deployment steps for Jenkins server
                }
            }
        }
    }
    
    post {
        always {
            // Clean up workspace
            cleanWs()
        }
        success {
            echo 'Project Final Pipeline completed successfully!'
        }
        failure {
            echo 'Project Final Pipeline failed!'
        }
    }
}