#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import sys
import warnings
from collections import namedtuple

from . import BaseClient
from . import ConcurrentBaseClient

__all__ = ['BertClient', 'ConcurrentBertClient']

__version__ = '1.10.0'

if sys.version_info >= (3, 0):
    from ._py3_var import *
else:
    from ._py2_var import *

_Response = namedtuple('_Response', ['id', 'content'])
Response = namedtuple('Response', ['id', 'embedding', 'tokens'])


class BertClient(BaseClient):
    def __init__(self, ip='localhost', port=5555, port_out=5556,
                 output_fmt='ndarray', show_server_config=False,
                 identity=None, check_version=True, check_length=True,
                 check_token_info=True, ignore_all_checks=False,
                 timeout=-1):
        """ A client object connected to a BertServer

        Create a BertClient that connects to a BertServer.
        Note, server must be ready at the moment you are calling this function.
        If you are not sure whether the server is ready, then please set `ignore_all_checks=True`

        You can also use it as a context manager:

        .. highlight:: python
        .. code-block:: python

            with BertClient() as bc:
                bc.encode(...)

            # bc is automatically closed out of the context

        :type timeout: int
        :type check_version: bool
        :type check_length: bool
        :type check_token_info: bool
        :type ignore_all_checks: bool
        :type identity: str
        :type show_server_config: bool
        :type output_fmt: str
        :type port_out: int
        :type port: int
        :type ip: str
        :param ip: the ip address of the server
        :param port: port for pushing data from client to server, must be consistent with the server side config
        :param port_out: port for publishing results from server to client, must be consistent with the server side config
        :param output_fmt: the output format of the sentence encodes, either in numpy array or python List[List[float]] (ndarray/list)
        :param show_server_config: whether to show server configs when first connected
        :param identity: the UUID of this client
        :param check_version: check if server has the same version as client, raise AttributeError if not the same
        :param check_length: check if server `max_seq_len` is less than the sentence length before sent
        :param check_token_info: check if server can return tokenization
        :param ignore_all_checks: ignore all checks, set it to True if you are not sure whether the server is ready when constructing BertClient()
        :param timeout: set the timeout (milliseconds) for receive operation on the client, -1 means no timeout and wait until result returns
        """
        self.length_limit = 0
        self.token_info_available = False
        # self.check_version = check_version
        self.check_length = check_length
        self.check_token_info = check_token_info
        super().__init__(ip, port, port_out,
                         output_fmt,
                         show_server_config,
                         identity,
                         check_version,
                         ignore_all_checks,
                         timeout)

        # self.length_limit = 0
        # self.token_info_available = False
        # self.check_length = check_length
        # self.check_token_info = check_token_info

    def check_status(self, s_status):
        if self.check_version and s_status['bert_server_version'] != self.client_status['bert_client_version']:
            raise AttributeError('version mismatch! server version is %s but client version is %s!\n'
                                 'consider disable version-check by "BertClient(check_version=False)"' % (
                                     s_status['bert_server_version'], self.client_status['bert_client_version']))

        if self.check_length:
            if s_status['max_seq_len'] is not None:
                self.length_limit = int(s_status['max_seq_len'])
            else:
                self.length_limit = None
        if self.check_token_info:
            self.token_info_available = bool(s_status['show_tokens_to_client'])

    @property
    def client_status(self):
        return {**super().client_status, 'bert_client_version': __version__}

    def encode(self, texts, blocking=True, is_tokenized=False, show_tokens=False):
        """ Encode a list of strings to a list of vectors

        `texts` should be a list of strings, each of which represents a sentence.
        If `is_tokenized` is set to True, then `texts` should be list[list[str]],
        outer list represents sentence and inner list represent tokens in the sentence.
        Note that if `blocking` is set to False, then you need to fetch the result manually afterwards.

        .. highlight:: python
        .. code-block:: python

            with BertClient() as bc:
                # encode untokenized sentences
                bc.encode(['First do it',
                          'then do it right',
                          'then do it better'])

                # encode tokenized sentences
                bc.encode([['First', 'do', 'it'],
                           ['then', 'do', 'it', 'right'],
                           ['then', 'do', 'it', 'better']], is_tokenized=True)

        :type is_tokenized: bool
        :type show_tokens: bool
        :type blocking: bool
        :type timeout: bool
        :type texts: list[str] or list[list[str]]
        :param is_tokenized: whether the input texts is already tokenized
        :param show_tokens: whether to include tokenization result from the server. If true, the return of the function will be a tuple
        :param texts: list of sentence to be encoded. Larger list for better efficiency.
        :param blocking: wait until the encoded result is returned from the server. If false, will immediately return.
        :param timeout: throw a timeout error when the encoding takes longer than the predefined timeout.
        :return: encoded sentence/token-level embeddings, rows correspond to sentences
        :rtype: numpy.ndarray or list[list[float]]

        """
        if is_tokenized:
            self._check_input_lst_lst_str(texts)
        else:
            self._check_input_lst_str(texts)

        if self.length_limit is None:
            warnings.warn('server does not put a restriction on "max_seq_len", '
                          'it will determine "max_seq_len" dynamically according to the sequences in the batch. '
                          'you can restrict the sequence length on the client side for better efficiency')
        elif self.length_limit and not self._check_length(texts, self.length_limit, is_tokenized):
            warnings.warn('some of your sentences have more tokens than "max_seq_len=%d" set on the server, '
                          'as consequence you may get less-accurate or truncated embeddings.\n'
                          'here is what you can do:\n'
                          '- disable the length-check by create a new "BertClient(check_length=False)" '
                          'when you do not want to display this warning\n'
                          '- or, start a new server with a larger "max_seq_len"' % self.length_limit)

        embedding_outputs, token_extra_infos = super().infer(texts, blocking)
        # super(BaseClient, self).infer(texts, blocking)

        if self.token_info_available and show_tokens:
            return embedding_outputs, token_extra_infos
        elif not self.token_info_available and show_tokens:
            warnings.warn('"show_tokens=True", but the server does not support showing tokenization info to clients.\n'
                          'here is what you can do:\n'
                          '- start a new server with "bert-serving-start -show_tokens_to_client ..."\n'
                          '- or, use "encode(show_tokens=False)"')
        return embedding_outputs

    @staticmethod
    def _check_length(texts, len_limit, tokenized):
        if tokenized:
            # texts is already tokenized as list of str
            return all(len(t) <= len_limit for t in texts)
        else:
            # do a simple whitespace tokenizer
            return all(len(t.split()) <= len_limit for t in texts)

    @staticmethod
    def _check_input_lst_str(texts):
        if not isinstance(texts, list):
            raise TypeError('"%s" must be %s, but received %s' % (texts, type([]), type(texts)))
        if not len(texts):
            raise ValueError(
                '"%s" must be a non-empty list, but received %s with %d elements' % (texts, type(texts), len(texts)))
        for idx, s in enumerate(texts):
            if not isinstance(s, _str):
                raise TypeError('all elements in the list must be %s, but element %d is %s' % (type(''), idx, type(s)))
            if not s.strip():
                raise ValueError(
                    'all elements in the list must be non-empty string, but element %d is %s' % (idx, repr(s)))
            if _py2:
                texts[idx] = _unicode(texts[idx])

    @staticmethod
    def _check_input_lst_lst_str(texts):
        if not isinstance(texts, list):
            raise TypeError('"texts" must be %s, but received %s' % (type([]), type(texts)))
        if not len(texts):
            raise ValueError(
                '"texts" must be a non-empty list, but received %s with %d elements' % (type(texts), len(texts)))
        for s in texts:
            BertClient._check_input_lst_str(s)


class ConcurrentBertClient(ConcurrentBaseClient):

    def __init__(self, max_concurrency=10, **kwargs):
        """ A thread-safe client object connected to a BertServer

        Create a BertClient that connects to a BertServer.
        Note, server must be ready at the moment you are calling this function.
        If you are not sure whether the server is ready, then please set `check_version=False` and `check_length=False`

        :type max_concurrency: int
        :param max_concurrency: the maximum number of concurrent connections allowed

        """

        self.available_bc = [BertClient(**kwargs) for _ in range(max_concurrency)]
        self.max_concurrency = max_concurrency

    @ConcurrentBaseClient._concurrent
    def encode(self, **kwargs):
        pass

    def fetch(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentBaseClient" is not implemented yet')

    def fetch_all(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentBaseClient" is not implemented yet')

    def infer_async(self, **kwargs):
        raise NotImplementedError('Async encoding of "ConcurrentBaseClient" is not implemented yet')
