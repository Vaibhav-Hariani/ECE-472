#!/bin/bash
isort .
black .
for var in "$@"
    do
        for file in $var/*.py
        do 
            output="Submissions/$(basename $file)"
            vim -c "hardcopy > $output.ps" -c wq $file
            echo "$output.ps $output.pdf"
            ps2pdf $output.ps $output.pdf
            rm Submissions/*.ps
        done
    done
#  pdfunite Submissions/*.py.pdf Submissions/Combined_Submission.pdf
