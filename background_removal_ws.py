import imghdr
import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List

import gc
import cv2
import psutil
import yaml
from flask import Flask, request
from werkzeug.utils import secure_filename

from config import path
from common import response

from model import U2NetInterface

# LOG = logger.Logger(config_logging.log_folder)
logger = logging.getLogger('web_service')
log_flask = logging.getLogger('werkzeug')
log_flask.disabled = True

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = path.upload
app.config['OUTPUT_FOLDER'] = path.u2net_output
app.config['MODEL_FOLDER'] = path.models

# start = time.time()
u2net = U2NetInterface(selected_model='u2net', device='gpu')
# end = time.time()
# print('U2Net Interface loaded in ', end - start)


def remove_uploaded_images(do_remove, filepath_list: List[str]):
    if not do_remove:
        filepath_list = filepath_list[1:]
    for fp in filepath_list:
        if os.path.exists(fp):
            os.remove(fp)


def log_results(result, request_id: str):
    pass


def send_response(result: response.Response, response_code: response.StatusCode,):
    print(response.StatusMessage[result.status_code])
    return json.dumps(result, default=lambda o: o.__dict__), {'Content-Type': 'application/json'}


def set_response():
    pass


@app.route('/rest/update/image', methods=['POST'])
def upload_file():
    result = response.Response()

    print('Upload started')
    form = request.form
    partial_start_time = time.time()

    request_id = str(form['request_id']) if 'request_id' in form.keys() else str(datetime.now().timestamp()).replace('.', '')
    request_id = secure_filename(request_id)

    file = request.files['file'] if 'file' in request.files.keys() else None

    if not file or not request_id:
        return send_response(result, response.BAD_REQUEST)

    wip_path = os.path.join(path.upload, datetime.today().strftime('%Y%m%d'))
    os.makedirs(wip_path, exist_ok=True)
    filepath_tmp = os.path.join(wip_path, str(request_id))
    file.save(filepath_tmp)

    # print(str(request_id) + ' - TMP file saved: ' + str(time.time() - partial_start_time))
    partial_start_time = time.time()

    invalid_path = os.path.join(wip_path, 'invalid', str(request_id) + '.jpg')
    os.makedirs(os.path.join(wip_path, 'invalid'), exist_ok=True)
    if not cv2.haveImageReader(filepath_tmp):
        shutil.move(filepath_tmp, invalid_path)
        return send_response(result, response.UNSUPPORTED_MEDIA_TYPE)

    # print(str(request_id) + ' - Format check: ' + str(time.time() - partial_start_time))
    partial_start_time = time.time()

    ext = imghdr.what(filepath_tmp)
    os.rename(filepath_tmp, filepath_tmp + '.' + ext)
    filepath = filepath_tmp + '.' + ext

    # print(str(request_id) + ' - File saved in the final folder: ' + str(time.time() - partial_start_time))
    partial_start_time = time.time()

    result.data = filepath

    try:
        print('Backgound removal started')
        u2net.predict(filepath)
        print(str(request_id) + ' - Backgound removal Completed: ' + str(time.time() - partial_start_time))
        # partial_start_time = time.time()

        # set_response(result, request_id)

    except Exception as e:
        logger.error('Unable to complete request ' + str(request_id) + ' on given image', exc_info=e)
        set_status(result, response.GENERIC_ERROR)

    result.execution_time = time.time() - partial_start_time

    gc.collect()
    log_profiling_info()

    return send_response(result, result.status_code)


def set_end_time(result, start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    result.execution_time = round(execution_time, 2)


def set_status(result, code):
    result.status_code = code
    result.status_message = response.StatusMessage[code]


def log_profiling_info():
    # if profiling.profiling_measure:
    pid = os.getpid()
    cpu_percent = str(psutil.Process(pid).cpu_percent()) + '%'
    memory = str(psutil.Process(pid).memory_info())
    memory_percent = str(round(psutil.Process(pid).memory_percent(), 2)) + '%'
    virtual_memory = psutil.virtual_memory()
    usage = {'cpu': cpu_percent, 'memory_percent': memory_percent, 'memory': memory,
             'virtual_memory_total': str(virtual_memory.total >> 30) + ' GB',
             'virtual_memory_available': str(virtual_memory.available >> 20) + ' MB'}
    print(json.dumps(usage, indent=4))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10004)
