# Copyright (c) Karl Schulz, Leon Sixt
#
# All rights reserved.
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.



import pickle
import os
import glob
from collections import OrderedDict


_assert_dir = "asserts"


def set_assert_dir(path):
    global _assert_dir
    _assert_dir = path

def get_assert_file(framework_name, key):
    outpath = os.path.join(_assert_dir, key)
    out_fname = os.path.join(outpath, framework_name + '.pickle')
    return out_fname


def assert_cache(framework_name, key, obj, assertion_fn,
            message_fn=None):
    outpath = os.path.join(_assert_dir, key)
    os.makedirs(outpath, exist_ok=True)
    out_fname = get_assert_file(framework_name, key)
    message_fn = message_fn or (lambda a,b: None)

    with open(out_fname, 'wb') as f:
        pickle.dump(obj, f)
        
    for fname in glob.glob(outpath + "/*.pickle"):
        framework = os.path.basename(fname).split('.')[0]
        if os.path.abspath(fname) == os.path.abspath(out_fname):
            continue
        with open(fname, 'rb') as f:
            other_obj = pickle.load(f)
            assert assertion_fn(obj, other_obj), message_fn(obj, other_obj)
    

    
    
def get_asserted_values(key):
    outpath = os.path.join(_assert_dir, key)
    out_dict = OrderedDict()
    for fname in glob.glob(outpath + "/*.pickle"):
        framework = os.path.basename(fname).split('.')[0]
        with open(fname, 'rb') as f:
            out_dict[framework] = pickle.load(f)
    return out_dict
            