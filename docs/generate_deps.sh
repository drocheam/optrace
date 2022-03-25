#!/bin/bash

# call from within the docs folder

pydeps '../src/Backend' --cluster --no-show -o ./source/images/Backend_Tree.svg
pydeps '../src/Frontend' --cluster --no-show -o ./source/images/Frontend_Tree.svg
