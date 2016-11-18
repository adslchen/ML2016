#!/bin/bash


KERAS_BACKEND=tensorflow python semi-supervised.py train $1 $2 0 20 3 4 
