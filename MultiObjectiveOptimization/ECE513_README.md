To Run in this directory do:

python multi_task/train_multi_task.py --param_file=./sarcos.json

To change between mgda and mgda-ub change the use_approximation in sarcos.json:
"use_approximation": true => mgda-ub
"use_approximation": false => mgda