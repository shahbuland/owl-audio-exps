import boto3
import threading
from dotenv import load_dotenv
import os

load_dotenv()

import random, tarfile, io, queue, torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionClosedError, SSLError, ReadTimeoutError, ResponseStreamingError
from http.client import IncompleteRead


class RandomizedQueue(queue.Queue):
    def _init(self, maxsize):
        self.queue = []

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        i = random.randrange(len(self.queue))
        self.queue[i], self.queue[-1] = self.queue[-1], self.queue[i]
        return self.queue.pop()


class S3CoDLatentAudioDataset(IterableDataset):
    def __init__(self, window_length=120, file_share_max=20,
                 rank=0, world_size=1,
                 bucket_name="cod-latent-depth-4x4",
                 prefix="labelled", verbose=True,
                 buf=1000):
        super().__init__()
        self.window = window_length
        self.file_share_max = file_share_max
        self.rank, self.world_size = rank, world_size
        self.bucket_name, self.prefix = bucket_name, prefix
        self.verbose = verbose

        # in‑memory buffer of windows
        self.data_queue = RandomizedQueue(maxsize=buf)

    def get_s3_client(self):
        worker = get_worker_info()
        cfg = Config(
            retries={'max_attempts': 10, 'mode': 'adaptive'},
            connect_timeout=60, read_timeout=300, max_pool_connections=worker.num_workers * 4,
            signature_version='s3v4', tcp_keepalive=True, parameter_validation=False,
        )
        return boto3.client(
            "s3",
            endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION'],
            config=cfg
        )

    def __iter__(self):

        client = self.get_s3_client()
        objs = client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix).get("Contents", [])
        self.keys = [o["Key"] for o in objs if o["Key"].endswith(".tar")]

        worker = get_worker_info()
        if worker:
            random.seed(self.rank * worker.num_workers + worker.id)

        def producer():
            while True:
                if self.data_queue.qsize() < self.data_queue.maxsize:
                    key = random.choice(self.keys)
                    # retryable get_object with backoff
                    try:
                        resp = client.get_object(Bucket=self.bucket_name, Key=key)
                        body = resp['Body'].read()
                    except (ReadTimeoutError, SSLError, ClientError,
                            ConnectionClosedError, ResponseStreamingError, IncompleteRead) as e:
                        if self.verbose:
                            print(f"[Rank {self.rank}] S3 read error: {e}")
                        continue
                    if not body:
                        continue
                    with tarfile.open(fileobj=io.BytesIO(body)) as tar:
                        bases = {
                            m.name.rsplit(".", 2)[0]
                            for m in tar if m.name.endswith(".latent.pt")
                        }
                        for base in bases:
                            def load(suf):
                                f = tar.extractfile(f"{base}.{suf}.pt")
                                return (torch.load(io.BytesIO(f.read()))
                                        if f else None)

                            latent = load("latent")
                            audio = load("audiolatent")
                            mouse = load("mouse")
                            button = load("buttons")

                            mouse = torch.zeros(latent.size(0), 2) if mouse is None else mouse
                            button = torch.zeros(latent.size(0), 11) if button is None else button

                            if latent is None or audio is None:
                                continue

                            # clamp & nan→0 once
                            latent = torch.nan_to_num(latent.clamp(-8,8)).float()
                            L = min(latent.size(0), mouse.size(0), button.size(0), audio.size(0))

                            for _ in range(self.file_share_max):
                                if L < self.window:
                                    break
                                i = random.randint(0, L - self.window)
                                item = (
                                    latent[i:i+self.window],
                                    mouse [i:i+self.window],
                                    button[i:i+self.window],
                                    audio [i:i+self.window],
                                )
                                self.data_queue.put(item)

        threading.Thread(target=producer, daemon=True).start()
        while True:
            if self.data_queue.qsize() == 1:
                print("queue empty on rank", self.rank)
            yield self.data_queue.get()


def collate_fn(batch):
    # batch is list of quadruples
    latents, mouses, buttons, audios = zip(*batch)

    latents = torch.stack(latents).bfloat16()    # [b,n,c,h,w]
    mouses = torch.stack(mouses).bfloat16()      # [b,n,2]
    buttons = torch.stack(buttons).bfloat16()    # [b,n,n_buttons]
    audios = torch.stack(audios).bfloat16()      # [b,n,d]

    return latents, audios, mouses, buttons

def get_loader(batch_size, **data_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    ds = S3CoDLatentAudioDataset(rank=rank, world_size=world_size, **data_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        prefetch_factor=4,  # prefetch to mitigate slow batch loads
    )

if __name__ == "__main__":
    import time
    loader = get_loader(16,
                       window_length=16,
                       bucket_name="cod-data-latent-360x640to4x4",
                       prefix="feats/unlabelled",
                       file_share_max=50)

    import time
    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    first_time = end - start

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    second_time = end - start

    x,w,y,z = batch
    print(f"Time to load first batch: {first_time:.2f}s")
    print(f"Time to load second batch: {second_time:.2f}s")
    print(f"Video shape: {x.shape}")
    print(f"Audio shape: {w.shape}")
    print(x.std())
    print(f"Mouse shape: {y.shape}")
    print(f"Button shape: {z.shape}")
