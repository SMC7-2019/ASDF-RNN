#!/bin/bash

folderIn="../data/input"
folderOut="../data/output"

audioFolder="$folderOut/audio"

clipDuration=8
samplingRate=22050

counter=0
offset=4

mkdir -p $audioFolder
    
echo "     Block size: $clipDuration seconds"

fileList=(${folderIn}/*.mp4)

for file in "${fileList[@]}"
do
    
    echo
    echo "Source:" ${file}

    #get video duration
    fileDuration=$(ffprobe -i "${file}" -show_entries format=duration -v quiet -of csv="p=0")
    clips=$(echo "$fileDuration / $clipDuration" | bc)
    ffmpeg -v error -i "${file}" -ac 1 -ar $samplingRate $folderOut/audio_temp.wav
    
    echo "Lenght: $fileDuration seconds"
    echo " Clips:" $((clips+1)) 
    
    #split the video: audio clips and audio+video clips
    for clip in $(seq 0 $((clips-1))); do

        printf -v pfCounter "%08d" $counter
        echo -n -e "\r  Clip:" ${pfCounter}

        ffmpeg -v error -y -i $folderOut/audio_temp.wav -ac 1 -ar $samplingRate -ss $((2 + $clip * $clipDuration)) -t $clipDuration $audioFolder/aug_2_audio_${pfCounter}.wav
        ffmpeg -v error -y -i $folderOut/audio_temp.wav -ac 1 -ar $samplingRate -ss $((4 + $clip * $clipDuration)) -t $clipDuration $audioFolder/aug_4_audio_${pfCounter}.wav
        ffmpeg -v error -y -i $folderOut/audio_temp.wav -ac 1 -ar $samplingRate -ss $((6 + $clip * $clipDuration)) -t $clipDuration $audioFolder/aug_6_audio_${pfCounter}.wav
        
        counter=$((counter+1))
    done    

    rm $folderOut/audio_temp.wav


    echo
done