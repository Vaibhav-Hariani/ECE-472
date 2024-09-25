#!/bin/bash
isort .
black .
for var in "$@"
    do
        for file in $var/*.py
        do
            vim -c "hardcopy > Submissions/$(basename $file).ps" -c wq $file
        done
    done