#!/bin/bash

# Documentation generator script
########

pip show pdoc3 &>/dev/null  &
if [ $? = 0 ]; then
    pdoc --html ./ --force
else
    echo "Please install pdoc3 and re run the script --> pip3 install pdoc3 (https://pdoc3.github.io/pdoc/)."
fi
