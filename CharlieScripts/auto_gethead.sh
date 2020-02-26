#!/bin/bash
# auto_gethead
tels=("n" "t" "s" "a")
filt=("FILTNAM" "FILTER" "FILTER" "FILTER")
dirs=("/data/Lick/Nickel/workspace/" "/data/THACHER/workspace" "/data/LCO/Swope/workstch" "/data/ANDICAMCCD/workspace")
base="005aae"
for i in `seq 0 3`; do
    telescope=${tels[$i]}
    filter=${filt[$i]}
    dir=${dirs[$i]}
    echo "Working on telescope: $telescope..."
    for directory in `ls -d $dir/ut??????`; do
        echo $directory
        basedir=`basename $directory`
        gethead -a OBJECT RA_CNTR DEC_CNTR EXPTIME MJD-OBS $filter M3SIGMA $directory/1/${telescope}${base}*.sw.dcmp 2> /dev/null > tmp
        if [ ! -s tmp ]; then
            # empty so delete
            echo "${telescope}_${basedir} is empty..."
            rm tmp
        else
            echo "Writing ${telescope}_${basedir}..."
            cat tmp | awk '{if (NF==8) print}' > $HOME/GW190425_Data/${telescope}_${basedir}.dat
            rm tmp
        fi
    done
done