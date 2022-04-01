#!/bin/bash

out=" "
pkgs=$(pip list)

pyver=$(python --version)

out+=$(echo "Python Version: $pyver" && echo " " && echo " ")
out+=$(echo "Library Versions:" && echo " ")

out2=" "
out2+=$(echo "$pkgs" | grep "colorio " && echo " ")
out2+=$(echo "$pkgs" | grep "numexpr " && echo " ")
out2+=$(echo "$pkgs" | grep "numpy " && echo " ")
out2+=$(echo "$pkgs" | grep "matplotlib " && echo " ")
out2+=$(echo "$pkgs" | grep "traits " && echo " ")
out2+=$(echo "$pkgs" | grep "pynput " && echo " ")
out2+=$(echo "$pkgs" | grep "pyface " && echo " ")
out2+=$(echo "$pkgs" | grep "PyQt5 " && echo " ")
out2+=$(echo "$pkgs" | grep "Pillow " && echo " ")
out2+=$(echo "$pkgs" | grep "traitsui " && echo " ")
out2+=$(echo "$pkgs" | grep "mayavi " && echo " ")
out2+=$(echo "$pkgs" | grep "vtk " && echo " ")
out2+=$(echo "$pkgs" | grep "scipy " && echo " ")

out2=$(echo "$out2" | sort)

echo "$out$out2" > ./source/bib_versions.txt
