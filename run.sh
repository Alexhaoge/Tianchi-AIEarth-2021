mkdir result
mkdir output
python main.py -e 100 --loss rmse --val-loss score -l 0.01
# python main.py --infer
python zip.py