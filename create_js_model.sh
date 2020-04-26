#!/bin/sh
source env/bin/activate
python ./src/save_model.py
tensorflowjs_converter --input_format keras saved_model/generator.h5 tensorflowjs/