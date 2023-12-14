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

import asyncio
import uuid
import json
import time

class Client:
    def __init__(self, ip_address, port, id=uuid.uuid4(), channel=1):
        self.ip_address = ip_address
        self.port = port
        self.id = id
        self.channel = channel

    async def send_message(self):
        writer = None
        print(f"[INFO] Client ID: {self.id}")
        try:
            reader, writer = await asyncio.open_connection(self.ip_address, self.port)

            while True:
                message = input("Enter the message (type 'exit' to exit): ")
                if message.lower() == 'exit':
                    print("[INFO] Exiting the client")
                    break

                shut_down = input("Do you want to shut down the server? (yes/no): ") == 'yes'

                # Create a dictionary with the ID, isEnd flag, and shut_down flag
                message_dict = {
                    'message': message,
                    'id': str(self.id),
                    'shut_down': shut_down,
                    'channel': self.channel,
                }

                # Convert the dictionary to a string and encode it to bytes
                message_bytes = json.dumps(message_dict).encode()
                self.print_message_info(message_dict)
                
                start_time = time.time()  # Start the timer
                writer.write(message_bytes)
                data = await reader.read(1000)
                end_time = time.time()  # End the timer

                if not data:
                    print("[ERROR] Connection lost. Did not receive a response from the server.")
                    break

                received_message = data.decode()
                print(f'[INFO] Received: {received_message}')

                elapsed_time = end_time - start_time
                print(f"[INFO] Time taken for message transfer: {elapsed_time * 1000} ms")

                # If shut_down is True, exit the client
                if shut_down:
                    print("[INFO] Shutting down the client")
                    break

        except ConnectionRefusedError:
            print("[ERROR] Could not connect to the server. Is it running?")
        except TimeoutError:
            print("[ERROR] Connection timed out")
        except Exception as e:
            print(f'[ERROR] An error occurred: {e}')

        finally:
            print('[INFO] Closing the connection')
            if writer:
                writer.close()
                await writer.wait_closed()
    
    def print_message_info(self, message_dict):
        print("Message Information:")
        print(f"Message: {message_dict['message']}")
        print(f"ID: {message_dict['id']}")
        print(f"Shut Down: {'Yes' if message_dict['shut_down'] else 'No'}")
        print(f"Channel: {message_dict['channel']}")
    

def main():
    # Replace 'ip_address' with the IP address of the server machine (or IP of docker container)
    ip_address = '71.14.88.12'
    port = 8765
    client = Client(ip_address)
    asyncio.run(client.send_message())

if __name__ == '__main__':
    main()