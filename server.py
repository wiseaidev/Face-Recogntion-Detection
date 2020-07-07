import socket 
import sys
import time
if __name__ == "__main__":
    HOST = "localhost"
    PORT = 3333
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("\n[*] Socket successfully created")
    except socket.error as err:
        print("\n[*] Socket creation failed with error : ",err)
    try:
        server_socket.bind((HOST, PORT))
    except socket.error as msg:
        print('Bind failed. Error  : {} '.format(msg))
        sys.exit()
    server_socket.listen(1)
    print("\n[*] Socket is now listening")
    # each iteration read/send 4 k-bytes
    CHUNK_SIZE = 4 * 1024  
    while True:
        Client_Socket, addr = server_socket.accept()
        print('\n[*] Connected from ip: {} and port : {} '.format(addr[0],addr[1]))
        SND_Bytes= b""
        res = b''
        # receive the image as a stream of bytes
        # if the server receives b'0' ==> end of image
        while not (SND_Bytes == b'0'):
            SND_Bytes = Client_Socket.recv(CHUNK_SIZE)
            res += SND_Bytes
        # w: write -- b: binary
        g = open("image_from_client.jpg", "wb")
        # write down the stream of bytes as image
        g.write(res)
        # close the file
        g.close()
        print("Image received and recreated")
        # r: read -- b: binary
        f = open("image_from_client.jpg", "rb")
        # print(len(f))
        print("Sending image data...")
        # sleep untill the client invoke the bloking recv call
        time.sleep(0.5)
        NB_Bytes = 1
        while NB_Bytes>0:
            data = f.read(CHUNK_SIZE)
            NB_Bytes = Client_Socket.send(data)
            print("Length of Data :"+str(NB_Bytes))
        # send the bytes "0" to indicate the end of the image
        Client_Socket.send(b'0')
        Client_Socket.close()
        print("\n----------------Connection Closed-------------------\n")