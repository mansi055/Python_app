from flask import Flask
from flask import jsonify
from main_file_3 import main_func

app = Flask(__name__)

@app.route('/<image>')
def index(image):
	output = main_func(image)
	return jsonify({'output':output})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
