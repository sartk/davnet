ceport train as trainer
import sys
trainer.train(CUDA_VISIBLE_DEVICES=sys.argv[1], warmup_length=int(sys.argv[3]), checkpoint=sys.argv[2])
