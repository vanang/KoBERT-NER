# internal basic import
import yaml
import os
import logging
from logging import Formatter
from logging.handlers import TimedRotatingFileHandler

# NER prediction
from predict_api import predict_ner

# Flask and web service
from flask import Flask, render_template, make_response, request
from flask_cors import CORS

app = Flask('NER Prediction', static_url_path='')
CORS(app)

# get config
with open("app_config/app_config.yaml", 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

app.debug = config['app']['debug']

# set logging
if not app.debug:
    log_dir = os.path.join(os.getcwd(), config['logging']['location'])
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    file_handler = \
        TimedRotatingFileHandler(os.path.join(config['logging']['location'], config['logging']['filename']),
                                 when=config['logging']['rotate_cycle'],
                                 backupCount=config['logging']['backup_count'])
    file_handler.setFormatter(Formatter(config['logging']['format']))
    logging.getLogger('werkzeug').addHandler(file_handler)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(config['logging']['level'])


@app.route('/ner', methods=['POST'])
def perform_ner():
    payloads = request.get_json()
    try:
        context = payloads['context']
        preds = predict_ner(context=context)
        response = make_response({
            'prediction': preds,
            'status': 'success',
            'message': 'prediction succeeded'
        })
    except Exception as e:
        app.logger.exception('Exception: ', e)
        response = make_response({
            'prediction': '',
            'status': 'fail',
            'message': 'prediction failed due to ' + e
        })

    return response


if __name__ == "__main__":
    app.run(host=config['app']['host'], port=config['app']['port'])
