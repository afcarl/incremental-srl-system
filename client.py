import socket
import json

# Server setting
host = "localhost"
port = 11000
bsize = 1024

# Open tcp
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

print '\nInput a tokenized sentence.'
while True:
    word = raw_input('>>>  ')

    client.send(word)
    response = client.recv(4096)
    if len(response) == 0:
        print "Server Error"
        break

    json_dict = json.loads(response)
    words = json_dict['sent']
    prds = [words[p_index] for p_index in json_dict['prds']]
    print " ".join(words)
    for i, labels in enumerate(json_dict['labels']):
        text = ["%s/%s" % (w, l) for w, l in zip(words, labels)]
        print "PRD: %s %s" % (prds[i], " ".join(text))
    print
