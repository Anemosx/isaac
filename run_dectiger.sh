#!/bin/bash

python3 src/main.py --config=qtran --env-config=dec_tiger
python3 src/main.py --config=ow_qmix --env-config=dec_tiger
python3 src/main.py --config=cw_qmix --env-config=dec_tiger
python3 src/main.py --config=qmix_smac --env-config=dec_tiger
python3 src/main.py --config=iql --env-config=dec_tiger

python3 src/main.py --config=qtran --env-config=dec_tiger_det
python3 src/main.py --config=ow_qmix --env-config=dec_tiger_det
python3 src/main.py --config=cw_qmix --env-config=dec_tiger_det
python3 src/main.py --config=qmix_smac --env-config=dec_tiger_det
python3 src/main.py --config=iql --env-config=dec_tiger_det

python3 src/main.py --config=qtran --env-config=dec_tiger_full
python3 src/main.py --config=ow_qmix --env-config=dec_tiger_full
python3 src/main.py --config=cw_qmix --env-config=dec_tiger_full
python3 src/main.py --config=qmix_smac --env-config=dec_tiger_full
python3 src/main.py --config=iql --env-config=dec_tiger_full
