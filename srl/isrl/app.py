from ..utils import write


class App(object):
    def __init__(self, argv):
        self.argv = argv

    def run(self):
        argv = self.argv
        write('\nSYSTEM START')

        if argv.mode == 'train':
            from trainers import MulSeqTrainer
            from preprocessors import ISRLPreprocessor
            from model_api import ISRLSystemAPI
            write('\nMODE: Training')
            trainer = MulSeqTrainer(argv=argv,
                                    preprocessor=ISRLPreprocessor,
                                    model_api=ISRLSystemAPI)
            trainer.run()
        else:
            from predictors import MulSeqPredictor
            from preprocessors import ISRLPreprocessor
            from model_api import ISRLSystemAPI
            write('\nMODE: Predicting')
            predictor = MulSeqPredictor(argv=argv,
                                        preprocessor=ISRLPreprocessor,
                                        model_api=ISRLSystemAPI)
            if argv.online:
                predictor.run_online()
            else:
                predictor.run()
