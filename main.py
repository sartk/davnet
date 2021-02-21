import train as trainer
import sys

def num(string):
    return int(string) if string else None

trainer.train(message=sys.argv[1], CUDA_VISIBLE_DEVICES=sys.argv[2], warmup_length=num(sys.argv[3]), checkpoint=sys.argv[4], log_frequency=num(sys.argv[5]))
