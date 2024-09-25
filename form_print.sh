#!/bin/bash
isort .
black .
for var in "$@"
    do
        for file in $var/*.py
        do 
            output="Submissions/$(basename $file)"
            vim -c "hardcopy > $output.ps" -c wq $file
            # ps2pdf "$output.ps" -o "$output.pdf"
            # rm "$output.ps"
        done
    done