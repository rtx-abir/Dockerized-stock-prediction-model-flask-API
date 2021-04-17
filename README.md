# flask_api
 A stock prediction model accessible through flask api using a docker file to package it in a container to be able to run in any device
 
 
 # the Docker repository can be found [here](https://hub.docker.com/r/abir2285/stock_prediction_api_container)
 
 ## run instructions 
 
  > Once docker is installed in your local machine run the command "docker pull abir2285/stock_prediction_api_container:forpush" to pull from docker-hub
  > Next use the following command to run the image docker run -p 5000:5000 --name flask_api --memory="2g" -d abir2285/stock_prediction_api_container:forpush
  > specify --memory appropriately if needed.
  > Requests can be made to api endpoint https://localhost:5000/predict using Postman or Python request.py inside the container CMD using the appropriate features
  > Requests.py file already contains 240 test cases, that it randomly chooses one from and runs. 
