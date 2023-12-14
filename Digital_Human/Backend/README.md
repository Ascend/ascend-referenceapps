# Server-Client框架设计
服务端（Server）设计用于处理多个客户端（Clients）。每个客户端都由唯一的ID标识，该ID用于存储有关客户端的信息，并为每个客户端分配一个唯一的ADH Pipeline。

当客户端连接到服务器时，服务器将客户端的信息存储在字典（self.clients）中，其中键是客户端的ID，值是客户端信息的字典。

同时，服务器还为客户端分配一个唯一的ADH Pipeline。这是通过创建AIPipeline类的新实例并将其存储在另一个字典（self.client_pipelines）中完成的，其中键值是客户端的ID。

## 模型流水线
模型流水线在ADH_Pipeline类中实现。这个类负责按顺序运行多个AI模型，其中一个模型的输出被输入到下一个模型中。

管道中的每个模型都是Model类的实例，该类负责加载AI模型，处理输入数据，在处理过的数据上运行模型，并处理输出数据。

## 客户端、流水线和模型之间的交互
当服务器从客户端接收到输入数据时，它会检索客户端的ADH Pipeline并运行对应的流水线。流水线处理输入数据，在处理过的数据上运行序列中的每个模型，并处理最后模型的输出数据。然后服务器将输出数据发送回客户端（Web前端）。

<br>

# Client-Server Program

This is a simple client-server program written in Python using asyncio for asynchronous networking.

## Files

- `server.py`: This is the server script. It listens for incoming connections and handles messages from clients.
- `client.py`: This is the client script. It connects to the server and sends messages.

## How to Use

1. Replace 'ip_address' with the IP address of the server machine (or IP of docker container) in client.py

1. Start the server: Run `python server.py` in your terminal. The server will start and listen for incoming connections.

2. Start the client: In a new terminal window, run `python client.py`. The client will connect to the server.

3. Send messages: In the client terminal, you can type messages and press enter to send them to the server. The server will echo back the messages.

## Commands

- To exit the client, type 'exit' and press enter. This will close the client connection.
- To shut down the server, send the message 'shutdown' from the client. This will stop the server from accepting new connections and shut it down.

## Error Handling

The client script has built-in error handling for connection errors. If the server is not running when you start the client, it will print an error message: "Could not connect to the server. Is it running?"

## Note

The client script will not send empty messages. If you try to send an empty message, it will print a warning: "Cannot send an empty message", and prompt you to enter a new message.
<br>