import socket
import argparse
import pickle
import theano

theano.config.floatX = 'float32'

# Server setting
host = "localhost"
port = 11000
bsize = 1024

# Open tcp
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)


def get_argv():
    parser = argparse.ArgumentParser(description='Deep SRL tagger.')

    parser.add_argument('--mode', default='predict', help='train/predict')
    parser.add_argument('--task', default='isrl', help='srl/isrl')
    parser.add_argument('--action', default='shift_and_label', help='shift/label/shift_and_label')
    parser.add_argument('--online', action='store_true', default=False, help='online mode')
    parser.add_argument('--test', action='store_true', default=False, help='unit test')
    parser.add_argument('--unuse_word_corpus', action='store_true', default=False)

    parser.add_argument('--emb_dim', type=int, default=32, help='dimension of embeddings')
    parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of hidden layer')
    parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--rnn_unit', default='lstm', help='gru/lstm')

    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--init_emb', default=None, help='Init embeddings to be loaded')
    parser.add_argument('--init_emb_fix', action='store_true', default=False, help='init embeddings to be fixed or not')

    parser.add_argument('--opt_type', default='adam', help='sgd/adagrad/adadelta/adam')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='gradient clipping')
    parser.add_argument('--reg', type=float, default=0.0001, help='L2 Reg rate')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout Rate')

    parser.add_argument('--load_shift_model_param', default=None, help='path to params')
    parser.add_argument('--load_label_model_param', default=None, help='path to params')
    parser.add_argument('--load_word', default=None, help='path to word ids')
    parser.add_argument('--load_label', default=None, help='path to label ids')

    argv = parser.parse_args()
    print
    print argv
    print

    return argv


if __name__ == '__main__':
    from srl.isrl.predictors import ISRLPredictor
    from srl.isrl.preprocessors import ISRLPreprocessor
    from srl.isrl.model_api import ISRLSystemAPI

    # Main loop
    while True:
        predictor = ISRLPredictor(argv=get_argv(),
                                  preprocessor=ISRLPreprocessor,
                                  model_api=ISRLSystemAPI)
        predictor.run_server_mode()
        vocab_label = predictor.model_api.vocab_label

        # Wait socket connection
        print "Waiting for connections..."
        client_socket, client_address = server_socket.accept()
        print "Connected from:", client_address

        # Receive inputs and return outputs
        sent = []
        time_step = 0
        while True:

            # Receive inputs
            data = client_socket.recv(bsize)
            if not data:
                break
            inputs = data.rstrip().split()

            if len(inputs) < 2:
                client_socket.send("Please input a command & word")
                continue

            command = inputs[0]
            words = inputs[1:]

            if command == "srl":
                text = ""
                for word in words:
                    stack_a, stack_p, shift_proba, label_proba, label_pred = predictor.predict_server_mode(word,
                                                                                                           time_step)
                    sent.append(word)
                    for w, p in zip(sent, stack_p):
                        text += '%s/%s ' % (w, p)
                    text += '\n'

                    for i, (p, labels) in enumerate(zip(stack_p, label_pred)):
                        if p == 0:
                            continue

                        text += 'PRD:%s\t' % sent[i]
                        for w_index in xrange(len(sent)):
                            form = sent[w_index]
                            label = vocab_label.get_word(labels[w_index])
                            text += '%s/%s ' % (form, label)
                        text += '\n'

                    time_step += 1
                output = pickle.dumps(stack_p)
                #                client_socket.send(text)
                client_socket.send(output)

        client_socket.close()
        break

server_socket.close()
