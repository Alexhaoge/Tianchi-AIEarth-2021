mkdir result
mkdir output
python main.py -e 100 --loss rmse --val-loss score -l 0.05 --small-dataset=500 -p 30
# python main.py --infer
python zip.py