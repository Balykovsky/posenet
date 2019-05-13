import numpy as np

from flask import Flask, request
from commons import get_model, get_tensor

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image'].read()
            batch = get_tensor(image)
            model = get_model()
            output = model(batch)
            pos = np.mean(output[0].cpu().data.numpy().copy(), axis=0)
            qtn = np.mean(output[1].cpu().data.numpy().copy(), axis=0)
            print('position: {}, quaternion: {}'.format(pos, qtn))
