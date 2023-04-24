#!/usr/bin/env bash

if [ ${PWD##*/} == "scripts" ]; then
    cd ..
else
    :
fi

BASE_DIR="."

export FLASK_APP=app/index.py
flask run

exit 0