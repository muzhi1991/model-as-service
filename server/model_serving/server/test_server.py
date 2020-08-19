#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

from termcolor import colored
from zmq.utils import jsonapi

from . import BaseServer
from . import BaseSink
from . import BaseWorker
from . import ServerCmd
from . import SinkJob
from .helper import *

__all__ = ['__version__', 'TestServer']
__version__ = '1.10.0'

_tf_ver_ = check_tf_version()


class TestServer(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        ## add new status
        self.status_static.update({
            'tensorflow_version': _tf_ver_,
            'tf_server_version': __version__,
        })

    def create_sink(self, addr_front2sink):
        return TestSink(self.args, addr_front2sink)

    def create_worker(self, idx, addr_backend_list, addr_sink, device_id):
        return TestWorker(idx, self.args, addr_backend_list, addr_sink, device_id)


class TestSink(BaseSink):
    def __init__(self, args, front_sink_addr):
        super().__init__(args, front_sink_addr)

    def load_graph_config(self):
        return "graph path", {}

    def create_sink_job(self):
        return TestSinkJob(self.args, self.config)


class TestSinkJob(SinkJob):
    def __init__(self, cmd_args, model_config):
        super().__init__(cmd_args, model_config)

        self.hello = self.cmd_args.hello
        ## todo ???优雅一点
        self.with_extras = True

    def reset_array_element_shape(self, d_shape):
        """
        根据第一条数据的shape，是否需要修改每个返回结果的shape，
        :param d_shape:第一条数据不包括batch_size的shape，(seq长度，embedding长度)
        :return:
        """
        return d_shape

    def post_process(self, final_ndarray):
        return final_ndarray

    @property
    def is_done(self):
        return self.checksum > 0 and self.checksum == self.progress_extras
        # if self.with_extras:
        #     return self.checksum > 0 and self.checksum == self.progress_extras and self.checksum == self.progress_embeds
        # else:
        #     return self.checksum > 0 and self.checksum == self.progress_embeds


class TestWorker(BaseWorker):
    def __init__(self, id, args, worker_address_list, sink_address, device_id):
        super().__init__(id, args, worker_address_list, sink_address, device_id)
        config = self.config
        args = self.args

    def load_graph_config(self):

        return "graph path", {}

    def init(self):
        print("init")

    def to_features(self, client_id, raw_msg, extra_params, logger, sink):

        return {
            'client_id': client_id,
            'raw_msg': raw_msg,
            'extra_params': extra_params,
        }

    def infer_loop(self, feature_generator, sink_embed, logger):
        def process(g):
            for e in g:
                print("processing", e)
                yield e

        for r in process(feature_generator):
            # send_ndarray(sink_embed, r['client_id'], r['encodes'], ServerCmd.data_embed)
            sink_embed.send_multipart([r['client_id'], jsonapi.dumps(r), b'', ServerCmd.extra_data_single])
            logger.info('job done\tclient: %s' % (r['client_id']))
