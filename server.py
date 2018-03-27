import socket
import argparse
import json
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

    parser.add_argument('--init_emb', default='data/senna.130k.50d.txt', help='Init embeddings to be loaded')
    parser.add_argument('--load_word', default='param/word.txt', help='path to word ids')
    parser.add_argument('--load_label', default='param/label.txt', help='path to label ids')
    parser.add_argument('--load_pi_args', default='param/args.pi.pkl.gz', help='path to arg params')
    parser.add_argument('--load_lp_args', default='param/args.lp.pkl.gz', help='path to arg params')
    parser.add_argument('--load_pi_param', default='param/param.pi.pkl.gz', help='path to params')
    parser.add_argument('--load_lp_param', default='param/param.lp.pkl.gz', help='path to params')

    return parser.parse_args()


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

            json_dict = {'sent': [],
                         'prds': [],
                         'labels': []}

            for word in inputs:
                outputs = predictor.predict_server_mode(word, time_step)
                stack_a, stack_p, shift_proba, label_proba, label_pred = outputs

                sent.append(word)
                prds = []
                labels = []

                for i, (p, labels_i) in enumerate(zip(stack_p, label_pred)):
                    if p == 0:
                        continue
                    prds.append(i)
                    labels.append([vocab_label.get_word(labels_i[w_index]) for w_index in xrange(len(sent))])

                json_dict['sent'] = sent
                json_dict['prds'] = prds
                json_dict['labels'] = labels
                time_step += 1

            output = json.dumps(json_dict)
            client_socket.send(output)

        client_socket.close()
        break

server_socket.close()
