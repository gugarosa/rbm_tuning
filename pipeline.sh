# Common variables definition
DATA="mnist"
BATCH_SIZE=128
EPOCHS=5
DEVICE="cpu"
N_RUNS=1

# Architecture variables
MODEL="rbm"
N_INPUT=784
N_HIDDEN=128
N_CLASS=10
STEPS=1
LR=0.1
MOMENTUM=0
DECAY=0
T=1

# Optimization variables
MH="pso"
N_AGENTS=3
N_ITER=1
BOUNDS=0.01

# Iterates through all possible seeds
for SEED in $(seq 1 $N_RUNS); do
    # Trains an architecture
    python rbm_training.py ${DATA} ${MODEL} -n_input ${N_INPUT} -n_hidden ${N_HIDDEN} -n_classes ${N_CLASS} -steps ${STEPS} -lr ${LR} -momentum ${MOMENTUM} -decay ${DECAY} -temperature ${T} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -device ${DEVICE} -seed ${SEED}

    # Optimizes an architecture
    python rbm_optimization.py ${DATA} ${MODEL} ${MH} -batch_size ${BATCH_SIZE} -bounds ${BOUNDS} -n_agents ${N_AGENTS} -n_iter ${N_ITER} -seed ${SEED}

    # Stores files in an outputs folder
    mv ${MODEL}.pth outputs/${MODEL}_${SEED}.pth
    mv ${MH}_${MODEL}.history outputs/${MH}_${MODEL}_${SEED}.history
    mv ${MH}_${MODEL}.optimized outputs/${MH}_${MODEL}_${SEED}.optimized
done