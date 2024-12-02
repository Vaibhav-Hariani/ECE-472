#!/bin/bash
isort .
black .
for var in "$@"
    do
            output="Submissions/$(basename $var)"
            vim -c "hardcopy > $output.ps" -c wq $var
            echo "$output.ps $output.pdf"
            ps2pdf $output.ps $output.pdf
    done
        rm Submissions/*.ps