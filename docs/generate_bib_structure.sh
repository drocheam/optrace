#!/bin/bash

tree -I "__pycache__" ./optrace > ./docs/source/reference/structure_library.txt
tree -I "__pycache__" ./examples > ./docs/source/reference/structure_examples.txt
