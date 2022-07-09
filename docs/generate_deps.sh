#!/bin/bash

# call from within the docs folder

pydeps '../optrace/' --cluster --max-bacon=2 --no-show -o './source/images/Dep_Tree.svg'
