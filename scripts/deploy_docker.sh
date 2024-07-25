sudo docker build -t my_ml_service api
sudo docker run -d --rm -p 5152:8080 my_ml_service
sudo docker login
echo Enter your username on docker hub:
read username
# latest is default version
sudo docker tag my_ml_service $username/my_ml_service:latest

# specific version
sudo docker tag my_ml_service $username/my_ml_service:v1.0

sudo docker push $username/my_ml_service:latest
