PORT=8085

if [ "$1" = "" ]; then
    echo "tboard.sh port"
    exit
fi
PORT = $1
tensorboard --port=$PORT --logdir=./tboard/