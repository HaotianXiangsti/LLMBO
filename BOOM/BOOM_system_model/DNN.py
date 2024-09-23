import configparser

import os


class DNN():
    def __init__(self, dnn_type):


        config = configparser.ConfigParser()

        config_file_path = os.path.join('/content/drive/MyDrive/BOOM/config', f'{dnn_type}.ini')

        config.read(config_file_path)

        self.layer_comp_cost_info_forward = [float(num_str) for num_str in
                                             config.get('layer_comp_cost_info_forward', 'params').split(',')]

        self.layer_comp_cost_info_backward = [float(num_str) for num_str in
                                              config.get('layer_comp_cost_info_backward', 'params').split(',')]
        self.layer_output = [float(num_str) for num_str in
                             config.get('layer_output', 'params').split(',')]
        self.layer_gradient = [float(num_str) for num_str in
                             config.get('layer_gradient', 'params').split(',')]
        self.paras = [float(num_str) for num_str in
                      config.get('layer_paras', 'params').split(',')]
        self.N = int(config.get('N', 'value'))
        self.dataset_size = int(config.get('dataset_size', 'value'))
        self.batch_size = int(config.get('batch_size', 'value'))
        self.info = config.get('network_info', 'info')

