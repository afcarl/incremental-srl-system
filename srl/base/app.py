from . import write


class App(object):
    def __init__(self, argv):
        self.argv = argv

    def run(self):
        argv = self.argv
        write('\nSYSTEM START')

        from preprocessors import Preprocessor
        from model_api import BaseModelAPI

        if argv.mode == 'train':
            from trainers import Trainer
            write('\nMODE: Training')
            trainer = Trainer(argv=argv,
                              preprocessor=Preprocessor,
                              model_api=BaseModelAPI)
            trainer.run()
        else:
            from predictors import Predictor
            write('\nMODE: Predicting')
            predictor = Predictor(argv=argv,
                                  preprocessor=Preprocessor,
                                  model_api=BaseModelAPI)
            if argv.online:
                predictor.run_online()
            else:
                predictor.run()
