#!/bin/bash

mkdir -p ../data/in/omniweb

vars=(4 5 13 21 25 27 28)

vars_string=""
for var in "${vars[@]}"; do
    vars_string+="&vars=${var}"
done

for year in {2014..2024}; do
    start_date="${year}010100"
    end_date="${year}123123"
    curl -d "activity=retrieve&res=min&spacecraft=omni_min&start_date=${start_date}&end_date=${end_date}${vars_string}" https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi > ../data/in/omniweb/${year}.txt
done
