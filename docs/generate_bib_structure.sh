#!/bin/bash

tree -I "__pycache__" ./optrace > ./docs/source/library/structure_library.txt
tree -I "__pycache__" ./examples > ./docs/source/library/structure_examples.txt
