import socket
if __name__ == "__main__":
    HOST = "localhost"
    PORT = 3333
    try:
        client_Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("\n[*] Socket successfully created")
    except socket.error as err:
        print("\n[*] Socket creation failed with error : ",err)
    print("\n[*] Connecting to server @ ip = {} and port = {}".format(HOST,PORT))
    client_Socket.connect((HOST, PORT))
    # read the image as binary file
    Image = open("images/img3.jpg", "rb")
    t = 1
    print("\n[*] Sending image as bytes...")
    CHUNK_SIZE = 4 * 1024   
    while t>0:
        print(t)
        data = Image.read(CHUNK_SIZE)
        t = client_Socket.send(data)
    client_Socket.send(b"0")
    print("\n[*] Finish sending the image...")
    print("\n------------------------------------------------\n")
    print("----------------Receving data-------------------\n")
    print("\n------------------------------------------------\n")
    t = b''
    res = b''
    while not(t == b'0'):
        t = client_Socket.recv(CHUNK_SIZE)
        res += t
        print("\n[*] Length of Data received: "+ str(len(t)))
    g = open("from_server.jpg", "wb")

    g.write(res)
    g.close()
    client_Socket.close()
    #BrokenPipeError: [Errno 32] Broken pipe
