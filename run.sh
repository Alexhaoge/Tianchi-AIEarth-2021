mkdir result
mkdir output
python main.py -e 200 -l 0.01 -b 1024 --loss rmse --val-loss score --refit
# python main.py --infer
python zip.py
