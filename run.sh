mkdir result
mkdir output
python main.py -e 200 --loss rmse --val-loss score --small-dataset 500
# python main.py --infer
python zip.py