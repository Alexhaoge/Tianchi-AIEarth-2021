mkdir result
mkdir output
#python main.py -e 200 -l 0.01 -b 512 --loss rmse --val-loss score --refit
python main_lxyshuffle.py --infer
python zip.py
