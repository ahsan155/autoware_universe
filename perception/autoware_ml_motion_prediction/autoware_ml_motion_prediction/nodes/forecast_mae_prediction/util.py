import uuid

def uuid_to_str(uuid_msg):
    # uuid_msg.uuid is a numpy array of 16 bytes
    return str(uuid.UUID(bytes=bytes(uuid_msg.uuid)))