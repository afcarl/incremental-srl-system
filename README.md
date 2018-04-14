# incremental-srl-system

## Install
```
git clone https://github.com/hiroki13/incremental-srl-system.git
```

## Getting Started
### Prerequisites:
* python 2
* numpy
* theano

### Example Command
- Command Prompt Mode

Type the following command.
```
python main.py
```

Then the following outputs will be displayed.

```
SYSTEM START

	Loading Embeddings...

	Vocab Size: 130000

Model configuration
	SHIFT: EmbCorpus -> EmbInit -> BiRNNLayer-1:(100x128):gru -> Dense(256x1,sigmoid)
	  - Params: 1197479
	LABEL: EmbCorpus -> EmbInit -> EmbMark -> BiRNNLayer-2:(105x128):gru -> Dense(256x54,softmax)
	  - Params: 1510630

Building a predict func...

Input a tokenized sentence.
>>>  
```

When you type a token, the corresponding result will be displayed.

```
>>> She
She

>>>  likes
She likes
PRD:likes  She/A0 likes/_

>>>  cats
She likes cats
PRD:likes  She/A0 likes/_ cats/A1
```

- Server/Client Mode
```
python server.py
```

```
python client.py
```

### Retrain models
- (1) Train a model for predicate identification
```
python main.py --task pi --mode train --train_data path/to/data --dev_data path/to/data --output_fn hoge.pi --save
```

- (2) Train a model for label prediction
```
python main.py --task lp --mode train --train_data path/to/data --dev_data path/to/data --output_fn hoge --save
```

- (3) Use the trained models
```
python main.py --load_word param/word.lp.hoge.txt --load_label param/label.lp.hoge.txt --load_pi_args param/args.pi.hoge.pkl.gz --load_lp_args param/args.lp.hoge.pkl.gz --load_pi_param param/param.pi.hoge.pkl.gz --load_lp_param param/param.lp.hoge.pkl.gz 
```

