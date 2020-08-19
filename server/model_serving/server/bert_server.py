#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>
import multiprocessing
import os
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from multiprocessing import Process
from multiprocessing.pool import Pool

import numpy as np
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from .helper import *
from .http import BertHTTPProxy
from .zmq_decor import multi_socket
from .tf_server import TFServer
from .tf_server import TFSink
from .tf_server import TFSinkJob
from .tf_server import TFWorker
from .tf_server import ServerCmd

__all__ = ['__version__', 'BertServer']
__version__ = '1.10.0'

_tf_ver_ = check_tf_version()


# class ServerCmd:
#     terminate = b'TERMINATION'
#     show_config = b'SHOW_CONFIG'
#     show_status = b'SHOW_STATUS'
#     new_job = b'REGISTER'
#     data_token = b'TOKENS'
#     data_embed = b'EMBEDDINGS'
#
#     @staticmethod
#     def is_valid(cmd):
#         return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())


class BertServer(TFServer):
    def __init__(self, args):
        super().__init__(args)
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        ## add new status
        self.status_static.update({
            'bert_server_version': __version__
        })

    def create_sink(self, addr_front2sink, config):
        return BertSink(self.args, addr_front2sink, config)

    def create_worker(self, idx, addr_backend_list, addr_sink, device_id, graph_path, config):
        return BertWorker(idx, self.args, addr_backend_list, addr_sink, device_id, graph_path, config)


class BertSink(TFSink):
    def __init__(self, args, front_sink_addr, bert_config):
        super().__init__(args, front_sink_addr, bert_config)
        # self.show_tokens_to_client = args.show_tokens_to_client
        # self.max_seq_len = args.max_seq_len
        # self.max_position_embeddings = bert_config.max_position_embeddings
        # self.fixed_embed_length = args.fixed_embed_length

    def create_sink_job(self, args, config):
        return BertSinkJob(args, config)


class BertSinkJob(TFSinkJob):
    def __init__(self, cmd_args, model_config):
        super().__init__(cmd_args, model_config)
        # max_seq_len, max_position_embeddings, with_tokens, fixed_embed_length
        # self._pending_embeds = []
        # self.tokens = []
        # self.tokens_ids = []
        # self.checksum = 0
        # self.final_ndarray = None
        # self.progress_tokens = 0 # replace process_extra
        # self.progress_embeds = 0 # replace process+
        ## todo ???优雅一点
        self.with_extras = cmd_args.show_tokens_to_client  # with_extras

        self.max_seq_len_unset = cmd_args.max_seq_len is None
        self.max_position_embeddings = model_config.max_position_embeddings
        # self.max_effective_len = 0
        self.fixed_embed_length = cmd_args.fixed_embed_length

    def reset_array_element_shape(self, d_shape):
        """
        根据第一条数据的shape，是否需要修改每个返回结果的shape，
        :param d_shape:第一条数据不包括batch_size的shape，(seq长度，embedding长度)
        :return:
        """
        # bert中未设置一致的seq长度，
        if self.max_seq_len_unset and len(d_shape) > 1:
            # if not set max_seq_len, then we have no choice but set result ndarray to
            # [B, max_position_embeddings, dim] and truncate it at the end
            d_shape[0] = self.model_config.max_position_embeddings
        return d_shape

    def post_process(self, final_ndarray):
        if self.max_seq_len_unset and not self.fixed_embed_length:
            # https://zhuanlan.zhihu.com/p/59767914
            x = np.ascontiguousarray(final_ndarray[:, 0:self.max_effective_len])
        else:
            x = self.final_ndarray
        return x


class BertWorker(TFWorker):
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, graph_config):
        super().__init__(id, args, worker_address_list, sink_address, device_id, graph_path, graph_config)
        # self.worker_id = id
        # self.device_id = device_id
        # self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
        # self.max_seq_len = args.max_seq_len
        # self.do_lower_case = args.do_lower_case
        # self.mask_cls_sep = args.mask_cls_sep
        # self.daemon = True
        # self.exit_flag = multiprocessing.Event()
        # self.worker_address = worker_address_list
        # self.num_concurrent_socket = len(self.worker_address)
        # self.sink_address = sink_address
        # self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        # self.gpu_memory_fraction = args.gpu_memory_fraction
        # self.model_dir = args.model_dir
        # self.verbose = args.verbose
        # self.graph_path = graph_path
        # self.bert_config = graph_config
        # self.use_fp16 = args.fp16
        # self.show_tokens_to_client = args.show_tokens_to_client
        # self.no_special_token = args.no_special_token
        # self.is_ready = multiprocessing.Event()

    def init(self):
        from .bert.tokenization import FullTokenizer
        from .bert.extract_features import convert_lst_to_features
        self.convert_lst_to_features = convert_lst_to_features
        self.tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'),
                                       do_lower_case=self.do_lower_case)

    def to_features(self, client_id, raw_msg, extra_params, logger, sink):
        msg = jsonapi.loads(raw_msg)

        # check if msg is a list of list, if yes consider the input is already tokenized
        is_tokenized = all(isinstance(el, list) for el in msg)
        tmp_f = list(self.convert_lst_to_features(msg, self.max_seq_len,
                                                  self.max_position_embeddings,
                                                  self.tokenizer, logger,
                                                  is_tokenized, self.mask_cls_sep, self.no_special_token))
        if self.show_tokens_to_client:
            sink.send_multipart([client_id, jsonapi.dumps([f.tokens for f in tmp_f]),
                                 b'', ServerCmd.extra_data_single])
        return {
            'client_id': client_id,
            'input_ids': [f.input_ids for f in tmp_f],
            'input_mask': [f.input_mask for f in tmp_f],
            'input_type_ids': [f.input_type_ids for f in tmp_f]
        }

    def get_tf_feature_inputs(self, tf):
        return {'input_ids': {'shape': (None, None), 'type': tf.int32},
                'input_mask': {'shape': (None, None), 'type': tf.int32},
                'input_type_ids': {'shape': (None, None), 'type': tf.int32}}

    def get_tf_feature_output(self, tf):
        return "final_encodes"

    #
    # def close(self):
    #     self.logger.info('shutting down...')
    #     self.exit_flag.set()
    #     self.is_ready.clear()
    #     self.terminate()
    #     self.join()
    #     self.logger.info('terminated!')
    #
    # def get_estimator(self, tf):
    #     from tensorflow.python.estimator.estimator import Estimator
    #     from tensorflow.python.estimator.run_config import RunConfig
    #     from tensorflow.python.estimator.model_fn import EstimatorSpec
    #
    #     def model_fn(features, labels, mode, params):
    #         with tf.gfile.GFile(self.graph_path, 'rb') as f:
    #             graph_def = tf.GraphDef()
    #             graph_def.ParseFromString(f.read())
    #
    #         input_names = ['input_ids', 'input_mask', 'input_type_ids']
    #
    #         output = tf.import_graph_def(graph_def,
    #                                      input_map={k + ':0': features[k] for k in input_names},
    #                                      return_elements=['final_encodes:0'])
    #
    #         return EstimatorSpec(mode=mode, predictions={
    #             'client_id': features['client_id'],
    #             'encodes': output[0]
    #         })
    #
    #     config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
    #     config.gpu_options.allow_growth = True
    #     config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
    #     config.log_device_placement = False
    #     # session-wise XLA doesn't seem to work on tf 1.10
    #     # if args.xla:
    #     #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    #
    #     return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))
    #
    # def run(self):
    #     self._run()
    #
    # @zmqd.socket(zmq.PUSH)
    # @zmqd.socket(zmq.PUSH)
    # @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    # def _run(self, sink_embed, sink_token, *receivers):
    #     # Windows does not support logger in MP environment, thus get a new logger
    #     # inside the process for better compatibility
    #     logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
    #
    #     logger.info('use device %s, load graph from %s' %
    #                 ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))
    #
    #     tf = import_tf(self.device_id, self.verbose, use_fp16=self.use_fp16)
    #     estimator = self.get_estimator(tf)
    #
    #     for sock, addr in zip(receivers, self.worker_address):
    #         sock.connect(addr)
    #
    #     sink_embed.connect(self.sink_address)
    #     sink_token.connect(self.sink_address)
    #     for r in estimator.predict(self.input_fn_builder(receivers, tf, sink_token), yield_single_examples=False):
    #         send_ndarray(sink_embed, r['client_id'], r['encodes'], ServerCmd.data_embed)
    #         logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))
    #
    # def input_fn_builder(self, socks, tf, sink):
    #     from .bert.extract_features import convert_lst_to_features
    #     from .bert.tokenization import FullTokenizer
    #
    #     def gen():
    #         # Windows does not support logger in MP environment, thus get a new logger
    #         # inside the process for better compatibility
    #         logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
    #         tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'),
    #                                   do_lower_case=self.do_lower_case)
    #
    #         poller = zmq.Poller()
    #         for sock in socks:
    #             poller.register(sock, zmq.POLLIN)
    #
    #         logger.info('ready and listening!')
    #         self.is_ready.set()
    #
    #         while not self.exit_flag.is_set():
    #             events = dict(poller.poll())
    #             for sock_idx, sock in enumerate(socks):
    #                 if sock in events:
    #                     client_id, raw_msg = sock.recv_multipart()
    #                     msg = jsonapi.loads(raw_msg)
    #                     logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg), client_id))
    #                     # check if msg is a list of list, if yes consider the input is already tokenized
    #                     is_tokenized = all(isinstance(el, list) for el in msg)
    #                     tmp_f = list(convert_lst_to_features(msg, self.max_seq_len,
    #                                                          self.bert_config.max_position_embeddings,
    #                                                          tokenizer, logger,
    #                                                          is_tokenized, self.mask_cls_sep, self.no_special_token))
    #                     if self.show_tokens_to_client:
    #                         sink.send_multipart([client_id, jsonapi.dumps([f.tokens for f in tmp_f]),
    #                                              b'', ServerCmd.data_token])
    #                     yield {
    #                         'client_id': client_id,
    #                         'input_ids': [f.input_ids for f in tmp_f],
    #                         'input_mask': [f.input_mask for f in tmp_f],
    #                         'input_type_ids': [f.input_type_ids for f in tmp_f]
    #                     }
    #
    #     def input_fn():
    #         return (tf.data.Dataset.from_generator(
    #             gen,
    #             output_types={'input_ids': tf.int32,
    #                           'input_mask': tf.int32,
    #                           'input_type_ids': tf.int32,
    #                           'client_id': tf.string},
    #             output_shapes={
    #                 'client_id': (),
    #                 'input_ids': (None, None),
    #                 'input_mask': (None, None),
    #                 'input_type_ids': (None, None)}).prefetch(self.prefetch_size))
    #
    #     return input_fn

#
# class ServerStatistic:
#     def __init__(self):
#         self._hist_client = CappedHistogram(500)
#         self._hist_msg_len = defaultdict(int)
#         self._client_last_active_time = CappedHistogram(500)
#         self._num_data_req = 0
#         self._num_sys_req = 0
#         self._num_total_seq = 0
#         self._last_req_time = time.perf_counter()
#         self._last_two_req_interval = []
#         self._num_last_two_req = 200
#
#     def update(self, request):
#         client, msg, req_id, msg_len = request
#         self._hist_client[client] += 1
#         if ServerCmd.is_valid(msg):
#             self._num_sys_req += 1
#             # do not count for system request, as they are mainly for heartbeats
#         else:
#             self._hist_msg_len[int(msg_len)] += 1
#             self._num_total_seq += int(msg_len)
#             self._num_data_req += 1
#             tmp = time.perf_counter()
#             self._client_last_active_time[client] = tmp
#             if len(self._last_two_req_interval) < self._num_last_two_req:
#                 self._last_two_req_interval.append(tmp - self._last_req_time)
#             else:
#                 self._last_two_req_interval.pop(0)
#             self._last_req_time = tmp
#
#     @property
#     def value(self):
#         def get_min_max_avg(name, stat, avg=None):
#             if len(stat) > 0:
#                 avg = sum(stat) / len(stat) if avg is None else avg
#                 min_, max_ = min(stat), max(stat)
#                 return {
#                     'avg_%s' % name: avg,
#                     'min_%s' % name: min_,
#                     'max_%s' % name: max_,
#                     'num_min_%s' % name: sum(v == min_ for v in stat),
#                     'num_max_%s' % name: sum(v == max_ for v in stat),
#                 }
#             else:
#                 return {}
#
#         def get_num_active_client(interval=180):
#             # we count a client active when its last request is within 3 min.
#             now = time.perf_counter()
#             return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)
#
#         avg_msg_len = None
#         if len(self._hist_msg_len) > 0:
#             avg_msg_len = sum(k * v for k, v in self._hist_msg_len.items()) / sum(self._hist_msg_len.values())
#
#         parts = [{
#             'num_data_request': self._num_data_req,
#             'num_total_seq': self._num_total_seq,
#             'num_sys_request': self._num_sys_req,
#             'num_total_request': self._num_data_req + self._num_sys_req,
#             'num_total_client': self._hist_client.total_size(),
#             'num_active_client': get_num_active_client()},
#             self._hist_client.get_stat_map('request_per_client'),
#             get_min_max_avg('size_per_request', self._hist_msg_len.keys(), avg=avg_msg_len),
#             get_min_max_avg('last_two_interval', self._last_two_req_interval),
#             get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
#         ]
#
#         return {k: v for d in parts for k, v in d.items()}
