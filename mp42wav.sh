#!/bin/bash


input_directory="raw/"

output_directory="raw_wav/"

# Ensure the output directory exists

mkdir -p "$output_directory"


for mp4_file in "$input_directory"/*.mp4; do

    filename=$(basename -- "$mp4_file")

    filename_without_extension="${filename%.*}"


    output_wav="$output_directory/$filename_without_extension.wav"

    ffmpeg -i "$mp4_file" "$output_wav"
done
