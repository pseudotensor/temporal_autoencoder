#!/bin/bash

oldpath=`pwd`

# go to videos directory
cd videos


######
# First make side-by-side
# Left is model, right is CLSTM
for fil in `ls model_*.mov`
do
    num="${fil//[!0-9]/}"

    sizevx=$((100+20+100)) # must be consistent with main.py and side-by-side above
    sizevy=100
    FPSbase=4

    # make text frames
    rm -rf bth_${num}.mov
    bash $oldpath/myslide.sh 20 $FPSbase .5 "Mini-Batch $num" $sizevx $sizevy bth_${num}.mov
    rm -rf btf_${num}.mov
    bash $oldpath/myslide.sh 20 $FPSbase 5 "Mini-Batch $num" $sizevx $sizevy btf_${num}.mov
    # make separator frames
    numframes=50
    moviesec=`echo "$numframes / $FPSbase" | bc -l`
    echo "moviesec=$moviesec"
    rm -rf sep_${num}.mov
    bash $oldpath/myslide.sh 100 $FPSbase $moviesec "|" 20 100 sep_${num}.mov

    newwidth=$((100*2+20))


    # http://stackoverflow.com/questions/17623676/text-on-video-ffmpeg
    # overlap text onto video
    rm -rf modelt_${num}.mov
    avconv -i model_${num}.mov -vf drawtext="fontfile=/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-L.ttf: \
text='Model': fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5: \
x=(w-text_w)/2: y=(h*0.3-text_h)/2" -codec:a copy modelt_${num}.mov

    rm -rf clstmt_${num}.mov
    avconv -i clstm_${num}.mov -vf drawtext="fontfile=/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-L.ttf: \
text='Clstm': fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5: \
x=(w-text_w)/2: y=(h*0.3-text_h)/2" -codec:a copy clstmt_${num}.mov

#    https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos
    #http://unix.stackexchange.com/questions/233832/merge-two-video-clips-into-one-placing-them-next-to-each-other

    rm -rf out_${num}.mov
    avconv -i modelt_${num}.mov -i sep_${num}.mov -i clstmt_${num}.mov -filter_complex "[0:v]pad=220:ih[int];[int][1:v]overlay=W/2:0[int2];[int2][2:v]overlay=120:0[vid]" -map [vid] out_${num}.mov

    #avconv -i model_${num}.mov -i sep_${num}.mov -i clstm_${num}.mov -filter_complex "nullsrc=size=220x100 [base]; [0:v] setpts=PTS-STARTPTS, scale=100x100 [left]; [1:v] setpts=PTS-STARTPTS, scale=20x100 [mid]; [2:v] setpts=PTS-STARTPTS, scale=100x100 [right];[base][left] overlay=shortest=1 [tmp1]; [tmp1][mid] overlay=shortest=1:x=100 [tmp2]; [tmp2][right] overlay=shortest=1:x=120" out_${num}.mov
#    avconv -i model_${num}.mov -i sep_${num}.mov -i clstm_${num}.mov -filter_complex "nullsrc=size=220x100 [base]; [0:v] setpts=PTS-STARTPTS, scale=100:100 [left]; [1:v] setpts=PTS-STARTPTS, scale=20:100 [mid]; [2:v] setpts=PTS-STARTPTS, scale=100:100 [right];[base][left] overlay [tmp1]; [tmp1][mid] overlay [tmp2]; [tmp2][right] overlay" out_${num}.mov

#    exit

done


######
# Second append existing sequence of video outputs in order by number

domerge=1
if [ $domerge -eq 1 ]
then
    # merge
    # can change FPS
    #FPS=30
    rm -rf out_all.mov
    rm -rf out_all2.mov

    #mencoder -mf fps=${FPS} -ovc lavc -o out_all.mov $outlist
    outlist=`ls out_*.mov bth_*.mov | sort -n -t _ -k 2`
    outlistf=`ls out_*.mov btf_*.mov | sort -n -t _ -k 2`

    mencoder -ovc lavc -o out_all.mov $outlist
    rm -rf out_all.mp4
    avconv -i out_all.mov out_all.mp4
    rm -rf out_all.mov

    # MERGE FOR FAST
    #mencoder -mf fps=${FPS} -ovc lavc -o out_all.mov $outlist

    mencoder -ovc lavc -o out_all2.mov $outlistf
    rm -rf out_all2.mp4
    avconv -i out_all2.mov out_all2.mp4
    rm -rf out_all2.mov

    # other ideas
    #http://askubuntu.com/questions/671673/merge-multiple-mp4-files-into-a-single-video-via-the-terminal
    #reference
    # mencoder -really-quiet -ovc lavc -lavcopts vcodec=mjpeg -mf fps=${FPS} -vf scale=${videoX}:${videoY} -o $output_video_file_name video_*.avi
    
fi

dofast=1
if [ $dofast -eq 1 ]
then
    # redo FPS using:
    rm -rf test test_track1.h264
    MP4Box -add out_all.mp4#video -raw 1 -new test
    MP4Box -add test_track1.h264:fps=30 -new out_all_fast.mp4
    rm -rf test test_track1.h264

    # redo FPS using:
    rm -rf test test_track1.h264
    MP4Box -add out_all2.mp4#video -raw 1 -new test
    MP4Box -add test_track1.h264:fps=30 -new out_all2_fast.mp4
    rm -rf test test_track1.h264
fi

# then as user do:
# smplayer videos/out_all_fast.mp4
echo ""
echo ""
echo "Now as user can do:"
echo "smplayer videos/out_all.mp4"
echo "smplayer videos/out_all_fast.mp4"



# return to old path
cd $oldpath

