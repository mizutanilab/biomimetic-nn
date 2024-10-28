while [ -n "`ps -e | grep python3`" ]; do
        #echo "waiting..."
        sleep 1000
done
date
#sleep 8000
#export EPOCHS=125
export EPOCHS=1250
export WND2D=0.4

export RUN_ID=1
export RNDSEED=25692
python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=30186
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=24796
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=14623
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=15679
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=14494
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=29857
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=17140
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=17945
#python3 -u DDIM_AFHQcat_240901d_git.py

RUN_ID=`expr $RUN_ID + 1`
RNDSEED=21186
#python3 -u DDIM_AFHQcat_240901d_git.py

echo "END"



