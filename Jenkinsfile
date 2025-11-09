pipeline {
    agent any
    
    tools {
        // Configure your Python tool here if needed
        // python "Python 3.9"
    }
    
    environment {
        // Define environment variables
        PROJECT_NAME = "retail-forecasting"
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
        
        stage('Test') {
            steps {
                script {
                    echo 'Running tests...'
                    // Add test commands here if you have tests
                    // sh 'python -m pytest tests/'
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
        
        stage('Deploy') {
            steps {
                script {
                    echo 'Deploying application...'
                    // Add deployment steps here
                    // For example, push to container registry
                    // sh 'docker push ${DOCKER_IMAGE}:${DOCKER_TAG}'
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
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}