import train as trainer
import sys
trainer.train(CUDA_VISIBLE_DEVICES=sys.argv[1], all_source_epoch=int(sys.argv[3]), checkpoint=sys.argv[2])
