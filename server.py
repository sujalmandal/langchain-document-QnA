import os

import main
from flask import Flask, flash, request

app = Flask(__name__)

UPLOAD_DIR_NAME = 'UPLOAD_DIR'
conversation_cache = {}


@app.route("/", methods=['GET'])
def default_route():
    return "<p>DocumentGPT is running!</p>"


@app.route("/get_loaded_files", methods=['GET'])
def get_loaded_files():
    return main.show_all_files()


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return 'please upload a file.'
    file = request.files['file']
    if file.filename.rsplit('.', 1)[1].lower() not in {'pdf'}:
        return "only pdf files allowed!"

    file_path = os.path.join(app.config[UPLOAD_DIR_NAME], file.filename)
    file.save(file_path)
    vec_file = main.read_document(file_path)
    return f"file successfully read, document_id : {vec_file}"


@app.route('/question', methods=['POST'])
def question():
    request_body = request.json
    if request_body["document_id"] is None:
        return 'document_id is required'

    if request_body["question"] is None:
        return 'document_id is required'

    return main.answer(request_body["document_id"], request_body["question"])


if __name__ == "__main__":
    if os.getenv(main.OPENAI_API_KEY_NAME, default=None) is None:
        raise Exception(f'env variable with the name {main.OPENAI_API_KEY_NAME} is required.')
    if os.getenv(UPLOAD_DIR_NAME, default=None) is None:
        raise Exception(f'env variable with the name {UPLOAD_DIR_NAME} is missing.')
    app.secret_key = os.urandom(24)
    app.config[UPLOAD_DIR_NAME] = os.getenv(UPLOAD_DIR_NAME)
    app.run(host='0.0.0.0', port=9090)
