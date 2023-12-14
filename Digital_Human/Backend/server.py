# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import asyncio
import json
from pipeline import ADH_Pipeline  # Import the AIPipeline class

sys.path.append("/home/devkit/ccw/AI_DIGITAL_HUMAN")
sys.path.append("/home/devkit/ccw/AI_DIGITAL_HUMAN/AI_Module/wenetspeech/asr_client_server")

class ServerShutdown(Exception):
    pass

class Server:
    def __init__(self, port=8765, ip_address='0.0.0.0'):
        self.server = None
        self.port = port            # Replace 'PORT' with the port number of the server machine (or port of docker container)
        self.ip_address = ip_address  # Replace 'IP_ADDRESS' with the IP address of the server machine (or IP of docker container)
        self.clients = {}            # Maintain user information
        self.client_pipelines = {}   # Plan1: Each client has its own pipeline

    async def start_server(self):
        self.server = await asyncio.start_server(
            self.handle_client, self.ip_address, self.port)

        addr = self.server.sockets[0].getsockname()
        print(f'[INFO] Serving on {addr}')

        async with self.server:
            await self.server.serve_forever()

    async def handle_client(self, reader, writer):
        while True:
            data = await reader.read(1000)
            if not data:
                self.handle_disconnection(writer)
                break

            message = data.decode()
            self.process_message(message, writer)

    def handle_disconnection(self, writer):
        client_id = writer.get_extra_info('peername')
        print(f"Client {client_id} disconnected")
        if client_id in self.clients:
            del self.clients[client_id]  # Remove the client from the clients dictionary
            del self.client_pipelines[client_id]  # Remove the client's pipeline

    def process_message(self, message, writer):
        try:
            # Convert the string back to a dictionary
            message_dict = json.loads(message)
        except json.JSONDecodeError:
            print("[ERROR] Failed to decode JSON")
            return

        client_id = message_dict['id']
        message_dict['ip_address'] = writer.get_extra_info('peername')
        self.clients[client_id] = message_dict

        # Assign a new pipeline to this client
        self.client_pipelines[client_id] = ADH_Pipeline()

        print(f"[INFO] Received message from {writer.get_extra_info('peername')}")

        if message_dict.get('message') == 'start':
            self.client_pipelines[client_id].run_pipeline()

        # Send a response back to the client
        response = "Message received"
        writer.write(response.encode())

        # Handle shutdown signal
        self.handle_shutdown(client_id)

        # Print all connected clients
        self.print_connected_clients()

    def print_connected_clients(self):
        print("[INFO] Connected clients:")
        for i, (id, client_info) in enumerate(self.clients.items(), start=1):
            print("-------------------------")
            print(f"Client {i}:")
            print(f"ID: {id}")
            print(f"Models assigned: {self.client_pipelines[id].module_names}")
            print(f"Last Message: {client_info['message']}")
            print(f"Shut Down: {client_info['shut_down']}")
            print(f"Channel: {client_info['channel']}")
        print("-------------------------")

    def handle_shutdown(self, client_id):
        if self.clients[client_id]['shut_down']:
            self.server.close()
            print("[INFO] Server has stopped accepting new connections")
            raise ServerShutdown("[INFO] Server is shutting down")


def main():
    server = Server()
    try:
        asyncio.run(server.start_server())
    except ServerShutdown as e:
        print(f'{e}')
    finally:
        print("[INFO] Server has completely shut down")

if __name__ == "__main__":
    main()