#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_octave_conv tests && \
    nosetests --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_octave_conv tests
