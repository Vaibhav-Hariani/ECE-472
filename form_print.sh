#!/bin/bash
isort .
black .
for var in "$@"
    do
        mkdir -p "Submissions/$var"
        for file in $var/*.py
        do 
            output="Submissions/$var/$(basename $file)"
            vim -c "hardcopy > $output.ps" -c wq $file
            # echo "$output.ps $output.pdf"
            ps2pdf $output.ps $output.pdf
        done
    done
    rm Submissions/*/*.ps
#  pdfunite Submissions/*.py.pdf Submissions/Combined_Submission.pdf
