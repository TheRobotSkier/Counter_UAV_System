import jack

client = jack.Client("Recorder")
inports = [client.inports.register(f"input_{i}") for i in range(4)]

@client.set_process_callback
def process(frames):
    # access input buffers here
    pass

client.activate()
