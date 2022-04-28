python scripts/generate_data_seg.py --object-set Shape2Motion/faucet/train data/Shape2Motion/faucet_train_1K --num-scenes 1000 --pos-rot 0 --global-scaling 0.6 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/faucet/val data/Shape2Motion/faucet_val_50 --num-scenes 50 --pos-rot 0 --global-scaling 0.6 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/generate_data_seg_test.py --object-set Shape2Motion/faucet/test data/Shape2Motion/faucet_test_standard --pos-rot 0 --global-scaling 0.6 --dense-photo --mp

python scripts/generate_data_seg.py --object-set Shape2Motion/oven/train data/Shape2Motion/oven_train_1K --num-scenes 1000 --pos-rot 1 --global-scaling 0.5 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/oven/test data/Shape2Motion/oven_test_50 --num-scenes 50 --pos-rot 1 --global-scaling 0.5 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/scripts/generate_data_seg_test.py --object-set Shape2Motion/oven/test data/Shape2Motion/oven_test_standard --pos-rot 1 --global-scaling 0.5 --dense-photo --mp

python scripts/generate_data_seg.py --object-set Shape2Motion/cabinet/train data/Shape2Motion/cabinet_train_1K --num-scenes 1000 --pos-rot 1 --global-scaling 0.8 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/cabinet/test data/Shape2Motion/cabinet_test_50 --num-scenes 50 --pos-rot 1 --global-scaling 0.8 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/scripts/generate_data_seg_test.py --object-set Shape2Motion/cabinet/test data/Shape2Motion/cabinet_test_standard --pos-rot 1 --global-scaling 0.8 --dense-photo --mp

python scripts/generate_data_seg.py --object-set Shape2Motion/laptop/train data/Shape2Motion/laptop_train_1K --num-scenes 1000 --pos-rot 0 --global-scaling 0.8 --num-proc 50 --sample-method mix --dense-photo
python scripts/generate_data_seg.py --object-set Shape2Motion/laptop/test data/Shape2Motion/laptop_test_50 --num-scenes 50 --pos-rot 0 --global-scaling 0.8 --num-proc 25 --sample-method uniform --dense-photo --canonical
python scripts/scripts/generate_data_seg_test.py --object-set Shape2Motion/laptop/test data/Shape2Motion/laptop_test_standard --pos-rot 0 --global-scaling 0.8 --dense-photo --mp
