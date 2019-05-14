# PoseNet
Simple realization of PoseNet model for camera localization task

## Start inference service:
  There are simple rest-api based on Flask framework.

  Firstly, build docker container:
  `nvidia-docker build -t posenet .`

  Run container with `5000` port for compatibility with test script:
  `nvidia-docker run -p 5000:5000 posenet`

### Perfomance testing:

  Use onboard [test script](inference/test_inference.py):
  (need [requests lib](https://github.com/kennethreitz/requests))
  
  `python inference/test_inference.py /path/to/imagefile.png`
  
  Or simple cURL command:
  
  `curl -X POST -F "image=@/path/to/imagefile.png" http://127.0.0.1:5000/predict`
  
  
