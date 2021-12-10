#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import sys
import inspect
import logging

logger = logging.getLogger(__name__)

try:
    from .swig_ascendfaiss import *
except ImportError as E:
    # we import * so that the symbol X can be accessed as faiss.X
    logger.error("Loading ascendfaiss error")
    print(E)


def replace_method(the_class, name, replacement, ignore_missing=False):
    try:
        orig_method = getattr(the_class, name)
    except AttributeError:
        if ignore_missing:
            return
        raise
    if orig_method.__name__ == 'replacement_' + name:
        # replacement was done in parent class
        return
    setattr(the_class, name + '_c', orig_method)
    setattr(the_class, name, replacement)


def handle_Index(the_class):
    def replacement_add(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        n, d = x.shape
        assert d == self.d
        assert ids.shape == (n,), 'not same nb of vectors as ids'
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_assign(self, x, k):
        n, d = x.shape
        assert d == self.d
        labels = np.empty((n, k), dtype=np.int64)
        self.assign_c(n, swig_ptr(x), swig_ptr(labels), k)
        return labels

    def replacement_train(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.train_c(n, swig_ptr(x))

    def replacement_search(self, x, k):
        n, d = x.shape
        assert d == self.d
        distances = np.empty((n, k), dtype=np.float32)
        labels = np.empty((n, k), dtype=np.int64)
        self.search_c(n, swig_ptr(x),
                      k, swig_ptr(distances),
                      swig_ptr(labels))
        return distances, labels

    def replacement_search_and_reconstruct(self, x, k):
        n, d = x.shape
        assert d == self.d
        distances = np.empty((n, k), dtype=np.float32)
        labels = np.empty((n, k), dtype=np.int64)
        recons = np.empty((n, k, d), dtype=np.float32)
        self.search_and_reconstruct_c(n, swig_ptr(x),
                                      k, swig_ptr(distances),
                                      swig_ptr(labels),
                                      swig_ptr(recons))
        return distances, labels, recons

    def replacement_remove_ids(self, x):
        if isinstance(x, IDSelector):
            sel = x
        else:
            assert x.ndim == 1
            sel = IDSelectorBatch(x.size, swig_ptr(x))
        return self.remove_ids_c(sel)

    def replacement_reconstruct(self, key):
        x = np.empty(self.d, dtype=np.float32)
        self.reconstruct_c(key, swig_ptr(x))
        return x

    def replacement_reconstruct_n(self, n0, ni):
        x = np.empty((ni, self.d), dtype=np.float32)
        self.reconstruct_n_c(n0, ni, swig_ptr(x))
        return x

    def replacement_update_vectors(self, keys, x):
        n = keys.size
        assert keys.shape == (n,)
        assert x.shape == (n, self.d)
        self.update_vectors_c(n, swig_ptr(keys), swig_ptr(x))

    def replacement_range_search(self, x, thresh):
        n, d = x.shape
        assert d == self.d
        res = RangeSearchResult(n)
        self.range_search_c(n, swig_ptr(x), thresh, res)
        # get pointers and copy them
        lims = rev_swig_ptr(res.lims, n + 1).copy()
        nd = int(lims[-1])
        D = rev_swig_ptr(res.distances, nd).copy()
        Idx = rev_swig_ptr(res.labels, nd).copy()
        return lims, D, Idx

    def replacement_sa_encode(self, x):
        n, d = x.shape
        assert d == self.d
        codes = np.empty((n, self.sa_code_size()), dtype='uint8')
        self.sa_encode_c(n, swig_ptr(x), swig_ptr(codes))
        return codes

    def replacement_sa_decode(self, codes):
        n, cs = codes.shape
        assert cs == self.sa_code_size()
        x = np.empty((n, self.d), dtype='float32')
        self.sa_decode_c(n, swig_ptr(codes), swig_ptr(x))
        return x

    replace_method(the_class, 'add', replacement_add)
    replace_method(the_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(the_class, 'assign', replacement_assign)
    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'search', replacement_search)
    replace_method(the_class, 'remove_ids', replacement_remove_ids)
    replace_method(the_class, 'reconstruct', replacement_reconstruct)
    replace_method(the_class, 'reconstruct_n', replacement_reconstruct_n)
    replace_method(the_class, 'range_search', replacement_range_search)
    replace_method(the_class, 'update_vectors', replacement_update_vectors,
                   ignore_missing=True)
    replace_method(the_class, 'search_and_reconstruct',
                   replacement_search_and_reconstruct, ignore_missing=True)
    replace_method(the_class, 'sa_encode', replacement_sa_encode)
    replace_method(the_class, 'sa_decode', replacement_sa_decode)


def handle_Index_Int8(the_class):
    def replacement_add(self, x):
        print("x type: {}".format(x.dtype))
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        n, d = x.shape
        assert d == self.d
        assert ids.shape == (n,), 'not same nb of vectors as ids'
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_assign(self, x, k):
        n, d = x.shape
        assert d == self.d
        labels = np.empty((n, k), dtype=np.int64)
        self.assign_c(n, swig_ptr(x), swig_ptr(labels), k)
        return labels

    def replacement_train(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.train_c(n, swig_ptr(x))

    def replacement_updateCentroids(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.updateCentroids_c(n, swig_ptr(x))

    def replacement_search(self, x, k):
        n, d = x.shape
        assert d == self.d
        distances = np.empty((n, k), dtype=np.float32)
        labels = np.empty((n, k), dtype=np.int64)
        self.search_c(n, swig_ptr(x),
                      k, swig_ptr(distances),
                      swig_ptr(labels))
        return distances, labels

    def replacement_remove_ids(self, x):
        if isinstance(x, IDSelector):
            sel = x
        else:
            assert x.ndim == 1
            sel = IDSelectorBatch(x.size, swig_ptr(x))
        return self.remove_ids_c(sel)

    replace_method(the_class, 'add', replacement_add)
    replace_method(the_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(the_class, 'assign', replacement_assign)
    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'updateCentroids', replacement_updateCentroids)
    replace_method(the_class, 'search', replacement_search)
    replace_method(the_class, 'remove_ids', replacement_remove_ids)


this_module = sys.modules[__name__]

for symbol in dir(this_module):
    obj = getattr(this_module, symbol)
    if inspect.isclass(obj):
        the_class = obj
        if issubclass(the_class, Index):
            handle_Index(the_class)
        if issubclass(the_class, AscendIndexInt8):
            handle_Index_Int8(the_class)


###########################################
# Add Python references to objects
# we do this at the Python class wrapper level.
###########################################


def add_ref_in_constructor(the_class, parameter_no):
    # adds a reference to parameter parameter_no in self
    # so that that parameter does not get deallocated before self
    original_init = the_class.__init__

    def replacement_init(self, *args):
        original_init(self, *args)
        self.referenced_objects = [args[parameter_no]]

    def replacement_init_multiple(self, *args):
        original_init(self, *args)
        pset = parameter_no[len(args)]
        self.referenced_objects = [args[no] for no in pset]

    if type(parameter_no) == dict:
        # a list of parameters to keep, depending on the number of arguments
        the_class.__init__ = replacement_init_multiple
    else:
        the_class.__init__ = replacement_init


def replace_destructor(the_class):
    # adds a reference to parameter parameter_no in self
    # so that that parameter does not get deallocated before self
    original_del = the_class.__del__

    def replacement_del(self):
        if original_del:
            original_del(self)

    the_class.__del__ = replacement_del


def add_ref_in_method(the_class, method_name, parameter_no):
    original_method = getattr(the_class, method_name)

    def replacement_method(self, *args):
        ref = args[parameter_no]
        if not hasattr(self, 'referenced_objects'):
            self.referenced_objects = [ref]
        else:
            self.referenced_objects.append(ref)
        return original_method(self, *args)

    setattr(the_class, method_name, replacement_method)


def add_ref_in_function(function_name, parameter_no):
    # assumes the function returns an object
    original_function = getattr(this_module, function_name)

    def replacement_function(*args):
        result = original_function(*args)
        ref = args[parameter_no]
        result.referenced_objects = [ref]
        return result

    setattr(this_module, function_name, replacement_function)


# handle all the AscendResources refs
add_ref_in_constructor(AscendIndexIVFPQ, 0)

replace_destructor(AscendIndexIVFPQ)


def index_cpu_to_ascend_py(devices, index, co=None):
    # builds C++ vectors for ascend indices
    vdev = IntVector()
    for i in devices:
        vdev.push_back(i)
    index = index_cpu_to_ascend(vdev, index, co)
    return index
