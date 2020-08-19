#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# using BertClient in sync way

import sys
import time
import functools

from model_serving.client.bert_client import BertClient


def to_features(a,b,c):
    print(a,b,c)
    return a+1

def foo():
    for i in range(10):
        yield 1, 2
ff=functools.partial(to_features, c=100)
for d in map(functools.partial(to_features, c=100),foo):
    print(d)


#
# if __name__ == '__main__':
#     # bc = BertClient(port=int(sys.argv[1]), port_out=int(sys.argv[2]), show_server_config=True)
#     # encode a list of strings
#     bc = BertClient()
#
#     with open('README.md', encoding="utf-8") as fp:
#         data = [v for v in fp if v.strip()][:512]
#         num_tokens = sum(len([vv for vv in v.split() if vv.strip()]) for v in data)
#
#     show_tokens = len(sys.argv) > 3 and bool(sys.argv[3])
#     # bc.encode(data)  # warm-up GPU
#     # for j in range(10):
#     #     tmp = data * (2 ** j)
#     #     c_num_tokens = num_tokens * (2 ** j)
#     #     start_t = time.time()
#     #     bc.encode(tmp, show_tokens=show_tokens)
#     #     time_t = time.time() - start_t
#     #     print('encoding %10d sentences\t%.2fs\t%4d samples/s\t%6d tokens/s' %
#     #           (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))
