import sys
import argparse

import numpy as np
import theano

sys.setrecursionlimit(100000000)
theano.config.floatX = 'float32'
np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep SRL tagger.')

    parser.add_argument('--mode', default='train', help='train/predict')
    parser.add_argument('--task', default='isrl', help='isrl/lp/pi')
    parser.add_argument('--action', default='pi_lp', help='pi_lp/lp')
    parser.add_argument('--online', action='store_true', default=False, help='online mode')

    ##################
    # Input Datasets #
    ##################
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to dev data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--gold_data', help='path to gold data')
    parser.add_argument('--pred_data', help='path to pred data')

    ###################
    # Dataset Options #
    ###################
    parser.add_argument('--data_size', type=int, default=1000000, help='data size to be used')

    ######################
    # Preprocess Options #
    ######################
    parser.add_argument('--cut_word', type=int, default=0)
    parser.add_argument('--cut_label', type=int, default=0)
    parser.add_argument('--unuse_word_corpus', action='store_true', default=False)

    ##################
    # Output Options #
    ##################
    parser.add_argument('--save', action='store_true', default=False, help='parameters to be saved or not')
    parser.add_argument('--output_fn', type=str, default=None, help='output file name')
    parser.add_argument('--output_type', type=str, default='txt', help='txt/xlsx')

    ###################
    # Samples/Batches #
    ###################
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')

    ###################
    # NN Architecture #
    ###################
    parser.add_argument('--emb_dim', type=int, default=32, help='dimension of embeddings')
    parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of hidden layer')
    parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--rnn_unit', default='gru', help='gru/lstm')

    ####################
    # Training Options #
    ####################
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--init_emb', default=None, help='Initial embeddings to be loaded')
    parser.add_argument('--init_emb_fix', action='store_true', default=False, help='init embeddings to be fixed or not')

    ########################
    # Optimization Options #
    ########################
    parser.add_argument('--opt_type', default='adam', help='sgd/adagrad/adadelta/adam')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='gradient clipping')
    parser.add_argument('--reg', type=float, default=0.0001, help='L2 Reg rate')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout Rate')

    ###################
    # Loading Options #
    ###################
    parser.add_argument('--load_param', default=None, help='path to params')
    parser.add_argument('--load_label', default=None, help='path to label ids')
    parser.add_argument('--load_word', default=None, help='path to word ids')
    parser.add_argument('--load_pi_param', default=None, help='path to params')
    parser.add_argument('--load_lp_param', default=None, help='path to params')

    argv = parser.parse_args()

    from srl.utils import write

    write('\nSYSTEM START')

    if argv.task == 'isrl':
        from srl.isrl.predictors import ISRLPredictor
        from srl.isrl.preprocessors import ISRLPreprocessor
        from srl.isrl.model_api import ISRLSystemAPI

        ISRLPredictor(argv=argv,
                      preprocessor=ISRLPreprocessor,
                      model_api=ISRLSystemAPI).run()

    elif argv.task == 'lp':
        from srl.trainers import LPTrainer
        from srl.predictors import LPPredictor
        from srl.preprocessors import LPPreprocessor
        from srl.model_api import LPModelAPI

        if argv.mode == 'train':
            write('\nMODE: Training')
            LPTrainer(argv=argv,
                      preprocessor=LPPreprocessor,
                      model_api=LPModelAPI).run()
        else:
            write('\nMODE: Predicting')
            LPPredictor(argv=argv,
                        preprocessor=LPPreprocessor,
                        model_api=LPModelAPI).run()

    elif argv.task == 'pi':
        from srl.trainers import PITrainer
        from srl.predictors import PIPredictor
        from srl.preprocessors import PIPreprocessor
        from srl.model_api import PIModelAPI

        if argv.mode == 'train':
            write('\nMODE: Training')
            PITrainer(argv=argv,
                      preprocessor=PIPreprocessor,
                      model_api=PIModelAPI).run()
        else:
            write('\nMODE: Predicting')
            PIPredictor(argv=argv,
                        preprocessor=PIPreprocessor,
                        model_api=PIModelAPI).run()
