#!/bin/bash

# http://unix.stackexchange.com/questions/31027/text-to-movie-from-commandline

fontSize=$1; FPSbase=$2 sec=$3; text="$4"; sizevx=$5 ; sizevy=$6 ; filename=$7
FPS=`echo "1 / $3" | bc -l`
#FPS=`echo "($FPS+0.5)/1" | bc`
echo "FPS=$FPS and FPSbase=$FPSbase"
numfiles=`echo "$FPSbase/$FPS" | bc -l`
numfiles=`echo "($numfiles+0.5)/1" | bc`
# ensure at least 2 files, some players don't show if only 1 (e.g. mplayer)
numfiles=`echo $(($numfiles>2?$numfiles:2))`
echo "numfiles=$numfiles"
convert -background black -fill white -pointsize $fontSize -gravity center -size ${sizevx}x${sizevy} caption:"$text" "text.jpg"

# duplicate
for i in `seq 0 $numfiles`
do
    cp text.jpg text$i.jpg
done

rm -rf $filename
avconv -y -r $FPSbase -i text%d.jpg $filename

doclean=1
if [ $doclean -eq 1 ]
then
    rm -rf text.jpg
    for i in `seq 0 $numfiles`
    do
        rm -rf text$i.jpg
    done
fi

#smplayer test.mp4
