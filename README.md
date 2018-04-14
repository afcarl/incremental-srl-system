# incremental-srl-system

## Install
```
git clone https://github.com/hiroki13/incremental-srl-system.git
```
</br>

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
</br>

## Retraining models
### Input File Format
- The CoNLL-09 format
- Trial Data is available at https://ufal.mff.cuni.cz/conll2009-st/trial-data.html
```
1       Ms.     ms.     ms.     NNP     NNP     _       _       2       2       TITLE   TITLE   _       _       _
2       Haag    haag    haag    NNP     NNP     _       _       3       3       SBJ     SBJ     _       _       A0
3       plays   play    play    VBZ     VBZ     _       _       0       0       ROOT    ROOT    Y       play.02 _
4       Elianti elianti elianti NNP     NNP     _       _       3       3       OBJ     OBJ     _       _       A1
5       .       .       .       .       .       _       _       3       3       P       P       _       _       _
```

### Example Command
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

