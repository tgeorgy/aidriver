#cd agent
#th train.lua > out.log &

#cd ../environment/local-runner
cd environment/local-runner

for iter in `seq 1 1000`;
do
    echo Iteration $iter
    cd ../local-runner
    sh local-runner-console.sh &

    sleep 3

    cd ../python2-cgdk
    python Runner.py
    
    mv ../local-runner/game.log ../local-runner/logs/game.${iter}.log
done   
