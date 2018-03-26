import socket
import pickle

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
    re = pickle.loads(response)

    if len(response) == 0:
        print "Server Error"
        break
    print response
