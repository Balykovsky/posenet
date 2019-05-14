import numpy as np

from flask import Flask, request, jsonify, make_response
from commons import get_model, get_tensor


app = Flask(__name__)
model = get_model('weights/weights.pth')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' in request.files:
            try:
                image = request.files['image'].read()
                batch = get_tensor(image)
            except Exception as err:
                print(err.args)
                return make_response(jsonify({'Error': 'Upload valid image file'}), 400)
            output = model(batch)
            pos = np.mean(output[0].cpu().data.numpy().copy(), axis=0)
            qtn = np.mean(output[1].cpu().data.numpy().copy(), axis=0)
            result = {'position': pos.tolist(), 'quaternion': qtn.tolist()}
            return make_response(jsonify(result), 200)
        else:
            return make_response(jsonify({'Error': 'Missed image in request'}), 400)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
