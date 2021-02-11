import train as trainer
import sys

def num(string):
    return int(string) if string else None

trainer.train(CUDA_VISIBLE_DEVICES=sys.argv[1], warmup_length=num(sys.argv[2]), checkpoint=sys.argv[3], log_frequency=num(sys.argv[4]))
