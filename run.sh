mkdir result
mkdir output
python main.py -e 100 --loss rmse --val-loss score
# python main.py --infer
python zip.py