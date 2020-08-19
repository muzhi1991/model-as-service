#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import os
from multiprocessing.pool import Pool

import numpy as np
from termcolor import colored
from zmq.utils import jsonapi

from . import BaseServer
from . import BaseSink
from . import BaseWorker
from . import ServerCmd
from . import SinkJob
from .helper import *

__all__ = ['__version__', 'TFServer']
__version__ = '1.10.0'

_tf_ver_ = check_tf_version()


class TFServer(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        ## add new status
        self.status_static.update({
            'tensorflow_version': _tf_ver_,
            'tf_server_version': __version__,
        })

    def create_sink(self, addr_front2sink, config):
        return TFSink(self.args, addr_front2sink, config)

    def create_worker(self, idx, addr_backend_list, addr_sink, device_id, graph_path, config):
        return TFWorker(idx, self.args, addr_backend_list, addr_sink, device_id, graph_path, config)

    def load_graph_config(self):
        self.logger.info('freeze, optimize and export graph, could take a while...')
        with Pool(processes=1) as pool:
            # optimize the graph, must be done in another process
            from .graph import optimize_graph
            graph_path, bert_config = pool.apply(optimize_graph, (self.args,))
        # from .graph import optimize_graph
        # self.graph_path = optimize_graph(self.args, self.logger)
        return graph_path, bert_config


class TFSink(BaseSink):
    def __init__(self, args, front_sink_addr, config):
        super().__init__(args, front_sink_addr, config)

    def create_sink_job(self, args, config):
        return TFSinkJob(args, config)


class TFSinkJob(SinkJob):
    def __init__(self, cmd_args, model_config):
        super().__init__(cmd_args, model_config)
        # self.max_seq_len_unset = self.cmd_args.max_seq_len is None
        # self.max_position_embeddings = self.model_config.max_position_embeddings
        # self.fixed_embed_length = self.cmd_args.fixed_embed_length
        # ## todo ???优雅一点
        # self.with_extras = self.cmd_args.show_tokens_to_client

    # def reset_array_element_shape(self, d_shape):
    #     """
    #     根据第一条数据的shape，是否需要修改每个返回结果的shape，
    #     :param d_shape:第一条数据不包括batch_size的shape，(seq长度，embedding长度)
    #     :return:
    #     """
    #     # bert中未设置一致的seq长度，
    #     if self.max_seq_len_unset and len(d_shape) > 1:
    #         # if not set max_seq_len, then we have no choice but set result ndarray to
    #         # [B, max_position_embeddings, dim] and truncate it at the end
    #         d_shape[0] = self.model_config.max_position_embeddings
    #     return d_shape
    #
    # def post_process(self, final_ndarray):
    #     if self.max_seq_len_unset and not self.fixed_embed_length:
    #         # https://zhuanlan.zhihu.com/p/59767914
    #         x = np.ascontiguousarray(final_ndarray[:, 0:self.max_effective_len])
    #     else:
    #         x = self.final_ndarray
    #     return x


class TFWorker(BaseWorker):
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, graph_config):
        super().__init__(id, args, worker_address_list, sink_address, device_id, graph_path, graph_config)
        args = self.args
        self.model_dir = args.model_dir
        self.show_tokens_to_client = args.show_tokens_to_client
        self.max_seq_len = args.max_seq_len
        self.do_lower_case = args.do_lower_case
        self.mask_cls_sep = args.mask_cls_sep
        self.no_special_token = args.no_special_token
        self.max_position_embeddings = self.config.max_position_embeddings

    # def load_graph_config(self):
    #     self.logger.info('freeze, optimize and export graph, could take a while...')
    #     with Pool(processes=1) as pool:
    #         # optimize the graph, must be done in another process
    #         from .graph import optimize_graph
    #         graph_path, bert_config = pool.apply(optimize_graph, (self.args,))
    #     # from .graph import optimize_graph
    #     # self.graph_path = optimize_graph(self.args, self.logger)
    #     return graph_path, bert_config

    # def init(self):
    #     from .bert.tokenization import FullTokenizer
    #     from .bert.extract_features import convert_lst_to_features
    #     self.convert_lst_to_features = convert_lst_to_features
    #     self.tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'),
    #                                    do_lower_case=self.do_lower_case)

    def get_estimator(self, tf):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # input_names = ['input_ids', 'input_mask', 'input_type_ids']
            input_names = list(self.get_tf_feature_inputs(tf).keys())
            output_names = [self.get_tf_feature_output(tf)]

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=[k + ':0' for k in output_names])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'outputs': output[0]
            })

        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        # session-wise XLA doesn't seem to work on tf 1.10
        # if args.xla:
        #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

    #
    # def to_features(self, client_id, raw_msg, extra_params, logger, sink):
    #     msg = jsonapi.loads(raw_msg)
    #
    #     # check if msg is a list of list, if yes consider the input is already tokenized
    #     is_tokenized = all(isinstance(el, list) for el in msg)
    #     tmp_f = list(self.convert_lst_to_features(msg, self.max_seq_len,
    #                                               self.max_position_embeddings,
    #                                               self.tokenizer, logger,
    #                                               is_tokenized, self.mask_cls_sep, self.no_special_token))
    #     if self.show_tokens_to_client:
    #         sink.send_multipart([client_id, jsonapi.dumps([f.tokens for f in tmp_f]),
    #                              b'', ServerCmd.extra_data_single])
    #     return {
    #         'client_id': client_id,
    #         'input_ids': [f.input_ids for f in tmp_f],
    #         'input_mask': [f.input_mask for f in tmp_f],
    #         'input_type_ids': [f.input_type_ids for f in tmp_f]
    #     }

    def infer_loop(self, feature_generator, sink_embed, logger):
        tf = import_tf(self.device_id, self.verbose, use_fp16=self.use_fp16)
        estimator = self.get_estimator(tf)

        def input_fn():
            return (tf.data.Dataset.from_generator(
                feature_generator,
                output_types=self.get_feature_types(tf),
                output_shapes=self.get_feature_shapes(tf)).prefetch(self.prefetch_size))

        for r in estimator.predict(input_fn,
                                   yield_single_examples=False):
            send_ndarray(sink_embed, r['client_id'], r['outputs'], ServerCmd.data_embed)
            logger.info('job done\tsize: %s\tclient: %s' % (r['outputs'].shape, r['client_id']))

    def get_feature_types(self, tf):
        inner_types = {
            'client_id': tf.string
        }
        model_types = {k: v['type'] for k, v in self.get_tf_feature_inputs(tf).items()}
        return {**inner_types, **model_types}

    def get_feature_shapes(self, tf):
        inner_features = {
            'client_id': ()
        }
        model_features = {k: v['shape'] for k, v in self.get_tf_feature_inputs(tf).items()}
        return {**inner_features, **model_features}

    def get_tf_feature_inputs(self, tf):
        raise Exception("subclass should implement")

    def get_tf_feature_output(self, tf):
        raise Exception("subclass should implement")
