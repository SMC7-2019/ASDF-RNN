#!/bin/bash

folderIn="data/input"
folderOut="data/output"

framesFolder="$folderOut/frames"
audioFolder="$folderOut/audio"
videoFolder="$folderOut/video"
jsonFolder="$folderOut/json"

clipDuration=8
samplingRate=22050
frameRate=30
framesPerClip=$((clipDuration*frameRate))

counter=0

mkdir -p $framesFolder
mkdir -p $audioFolder
mkdir -p $videoFolder
mkdir -p $jsonFolder
    
echo "  Sampling rate: ${samplingRate}Hz"
echo "     Block size: $clipDuration seconds"
echo "Frames per clip: $framesPerClip"

fileList=(${folderIn}/*.mp4)

for file in "${fileList[@]}"
do
    
    echo
    echo "Source:" ${file}

    #get video duration
    fileDuration=$(ffprobe -i "${file}" -show_entries format=duration -v quiet -of csv="p=0")
    clips=$(echo "$fileDuration / $clipDuration" | bc)
    
    echo "Lenght: $fileDuration seconds"
    echo "Frames:" $(echo "$fileDuration * $frameRate"  | bc)
    echo " Clips:" $((clips+1)) 

    #extract audio from master file 
    #it's MUCH faster extracting audio clips from this wav file instead of the original video
    #echo "Extracting audio track..."
    #ffmpeg -v error -i "${file}" -ac 1 -ar $samplingRate $folderOut/audio_temp.wav

    #frame extraction
    echo "Extracting frames..." 
    ffmpeg -v error -i "$file" -vf "scale=600:400:force_original_aspect_ratio=decrease,pad=600:400:(ow-iw)/2:(oh-ih)/2,setsar=1" -r $frameRate -start_number $((counter*framesPerClip)) $framesFolder/frame_%08d.png
    
    #split the video: audio clips and audio+video clips
    for clip in $(seq 0 $((clips-1))); do

        from=$((counter*framesPerClip))

        printf -v pfCounter "%08d" $counter
        echo -n -e "\r  Clip:" ${pfCounter}

        #ffmpeg -v error -y -i $folderOut/audio_temp.wav -ac 1 -ar $samplingRate -ss $(($clip * $clipDuration)) -t $clipDuration $audioFolder/audio_${pfCounter}.wav
        
        #ffmpeg -v error -y -r $frameRate -f image2 -s 600x400 -start_number $from -i ${framesFolder}/frame_%08d.png -i $audioFolder/audio_${pfCounter}.wav -vframes $framesPerClip -vcodec mpeg2video -crf 10  -pix_fmt yuv420p $videoFolder/video_${pfCounter}.mpg

        counter=$((counter+1))
    done    

    #rm $folderOut/audio_temp.wav


    echo
done