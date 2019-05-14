import sys
import requests


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Please, specify image path as arg')

    url = 'http://127.0.0.1:5000/predict'

    with open(sys.argv[1], 'rb') as img:
        files = {'image': img}
        response = requests.post(url, files=files)
        print(response.json())
