#! /usr/bin/env bash

# save all failing bash commands
failures=0
trap 'failures=$((failures+1))' ERR

# run unit tests
# check pep8
py.test IBA;
# run pytorch and tensorflow separatly
py.test test/test_pytorch.py;
py.test test/test_tensorflow_v1.py

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
