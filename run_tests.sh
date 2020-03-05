#! /usr/bin/env bash

set -e

# run unit tests
py.test


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



