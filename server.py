import socket
import argparse
import gzip
import cPickle
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

    parser.add_argument('--init_emb', default=None, help='Init embeddings to be loaded')
    parser.add_argument('--load_word', default='param/word.txt', help='path to word ids')
    parser.add_argument('--load_label', default='param/label.txt', help='path to label ids')
    parser.add_argument('--load_pi_args', default='param/args.pi.pkl.gz', help='path to arg params')
    parser.add_argument('--load_lp_args', default='param/args.lp.pkl.gz', help='path to arg params')
    parser.add_argument('--load_pi_param', default='param/param.pi.pkl.gz', help='path to params')
    parser.add_argument('--load_lp_param', default='param/param.lp.pkl.gz', help='path to params')

    argv = parser.parse_args()

    with gzip.open(argv.load_pi_args, 'rb') as gf:
        pi_args = cPickle.load(gf)

    with gzip.open(argv.load_lp_args, 'rb') as gf:
        lp_args = cPickle.load(gf)

    return argv, pi_args, lp_args


if __name__ == '__main__':
    from srl.isrl.predictors import ISRLPredictor
    from srl.isrl.preprocessors import ISRLPreprocessor
    from srl.isrl.model_api import ISRLSystemAPI

    # Main loop
    while True:
        argv, pi_args, lp_args = get_argv()
        predictor = ISRLPredictor(argv=argv,
                                  preprocessor=ISRLPreprocessor,
                                  model_api=ISRLSystemAPI)
        predictor.run_server_mode(pi_args, lp_args)
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

            text = ""
            for word in inputs:
                outputs = predictor.predict_server_mode(word, time_step)
                stack_a, stack_p, shift_proba, label_proba, label_pred = outputs

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
            output = pickle.dumps(text)
            client_socket.send(output)

        client_socket.close()
        break

server_socket.close()
