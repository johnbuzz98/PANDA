dataname='cifar10 fashion'
label_list='0 1 2 3 4 5 6 7 8 9'
panda_list='ewc es ses'
batch_size=32
epochs=20
lr=1e-2

for data in $dataname
do
    for label in $label_list
    do
        for pandatype in $panda_list
        do
        echo "data: $data, label: $label, panda: $panda, batch_size: $batch_size, epochs: $epochs, lr: $lr"
        EXP_NAME="data_$data-label_$label-panda_$panda-bs_$batch_size-epochs_$epochs-lr_$lr"
        if [ -d "$EXP_NAME" ]
        then
            echo "$EXP_NAME is exist"
        else
            python main.py \
                --exp-name $EXP_NAME \
                --dataname $data \
                --label $label \
                --pandatype $pandatype \
                --batch-size $batch_size \
                --epochs $epochs \
                --lr $lr
        fi
        done
    done
done
