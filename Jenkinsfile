pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "retail-sales-forecast"
        DOCKER_TAG = "latest"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/KISHANSINHAA/project.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh 'docker build -t $DOCKER_IMAGE:$DOCKER_TAG .'
                }
            }
        }

        stage('Run Container') {
            steps {
                script {
                    sh '''
                    docker stop retail_container || true
                    docker rm retail_container || true
                    docker run -d -p 8501:8501 --name retail_container $DOCKER_IMAGE:$DOCKER_TAG
                    '''
                }
            }
        }

    }

    post {
        success {
            echo 'Deployment Successful!'
        }

        failure {
            echo 'Pipeline Failed!'
        }
    }
}
