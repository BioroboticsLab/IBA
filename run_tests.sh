#! /usr/bin/env bash
#
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

# save all failing bash commands
failures=0
trap 'failures=$((failures+1))' ERR

# run unit tests
# check pep8
py.test IBA
# run pytorch and tensorflow separatly
py.test --cov-append test/test_pytorch.py
py.test --cov-append test/test_tensorflow_v1.py

# build documentation
cd doc
make html
make coverage

# check everything is documented
coverage_lines=$(cat _build/coverage/python.txt | wc -l)

if [ "$coverage_lines" != "2" ]; then
    >&2 echo ""
    >&2 echo "Found undocumented methods / classes. Please add a docstring!"
    >&2 echo ""
    >&2 cat _build/coverage/python.txt
    exit 1
fi


if ((failures == 0)); then
    echo "Success"
else
    echo "$failures failures"
    exit 1
fi
