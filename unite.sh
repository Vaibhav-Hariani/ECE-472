#!/bin/bash
cd Submissions
pdfunite *-Cover.pdf Proj_?/*.pdf Commons/*.pdf Submission-$(date +%F).pdf