#!/bin/bash

KERAS_BACKEND=tensorflow python auto-encoder.py train $1 $2 0
