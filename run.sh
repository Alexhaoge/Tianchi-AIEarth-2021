mkdir result
mkdir output
python main.py -e 100 --loss rmse --val-loss score --no-stop
# python main.py --infer
python zip.py