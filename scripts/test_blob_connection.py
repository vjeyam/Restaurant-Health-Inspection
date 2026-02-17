import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_BLOB_CONTAINER")

bsc = BlobServiceClient.from_connection_string(conn_str)
cc = bsc.get_container_client(container_name)

try:
    cc.create_container()
except Exception:
    pass

cc.upload_blob("healthcheck/test.txt", b"connection working", overwrite=True)

print("Success! Blob uploaded.")
