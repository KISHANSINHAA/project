pipeline {
    agent any

    environment {
        IMAGE_NAME = "retail-sales-forecast"
        DOCKERHUB_USER = "kishansinha2207"
        DOCKER_CREDENTIAL_ID = "3f1d3a28-5856-4e19-a3bb-dee6ec9f8883"
        IMAGE_TAG = "latest"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'master', url: 'https://github.com/KISHANSINHAA/project.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat """
                docker build -t %DOCKERHUB_USER%/%IMAGE_NAME%:%IMAGE_TAG% .
                """
            }
        }

        stage('DockerHub Login') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: "${DOCKER_CREDENTIAL_ID}",
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    bat """
                    echo %DOCKER_PASS% | docker login -u %DOCKER_USER% --password-stdin
                    """
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                bat """
                docker push %DOCKERHUB_USER%/%IMAGE_NAME%:%IMAGE_TAG%
                """
            }
        }

        stage('Deploy Container') {
            steps {
                bat """
                docker stop retail_container 2>nul
                docker rm retail_container 2>nul
                docker run -d -p 8501:8501 --name retail_container %DOCKERHUB_USER%/%IMAGE_NAME%:%IMAGE_TAG%
                """
            }
        }
    }

    post {
        success {
            echo "CI/CD Pipeline executed successfully!"
        }

        failure {
            echo "Pipeline failed. Check Jenkins logs."
        }

        always {
            echo "Pipeline finished."
        }
    }
}
