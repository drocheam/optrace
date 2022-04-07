#!/bin/bash

# call from within the docs folder

pydeps '../optrace/Backend' --cluster --no-show -o ./source/images/Backend_Tree.svg
pydeps '../optrace/Frontend' --cluster --no-show -o ./source/images/Frontend_Tree.svg
