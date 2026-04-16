import json
import logging
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient  # Async MongoDB
import redis.asyncio as redis
from redis.exceptions import ResponseError
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bulk-writer")

# Initialize clients
redis_client = redis.from_url(settings.REDIS_URL)
mongo_client = AsyncIOMotorClient(settings.MONGO_URL)
collection = mongo_client[settings.MONGO_DB].scans

STREAM_NAME = "scan_stream"
GROUP_NAME = "bulk-writers"
CONSUMER_NAME = "bulk-writer-1"
BATCH_SIZE = 50
BLOCK_TIME_MS = 5000


async def init_stream():
    try:
        # Note the 'await' here
        await redis_client.xgroup_create(STREAM_NAME, GROUP_NAME, id="0", mkstream=True)
        logger.info("Consumer group created")
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group already exists")
        else:
            raise


async def process_batch():
    # 1. Read from Redis (Blocking wait)
    entries = await redis_client.xreadgroup(
        groupname=GROUP_NAME,
        consumername=CONSUMER_NAME,
        streams={STREAM_NAME: ">"},
        count=BATCH_SIZE,
        block=BLOCK_TIME_MS,
    )

    if not entries:
        return

    docs = []
    message_ids = []

    for _, messages in entries:
        for msg_id, msg in messages:
            try:
                # msg[b"data"] is bytes in Redis
                data = json.loads(msg[b"data"])
                docs.append(data)
                message_ids.append(msg_id)
            except Exception as e:
                logger.error(f"Invalid JSON in {msg_id}: {e}")
                await redis_client.xack(STREAM_NAME, GROUP_NAME, msg_id)

    if not docs:
        return

    # 2. Async Write to Mongo
    try:
        await collection.insert_many(docs, ordered=False)
        await redis_client.xack(STREAM_NAME, GROUP_NAME, *message_ids)
        logger.info(f"Successfully processed {len(docs)} items")
    except Exception as e:
        # If this fails, we DON'T ACK.
        # Items stay in PEL (Pending Entires List) for retry logic.
        logger.error(f"Batch write failed: {e}")


async def main():
    await init_stream()
    logger.info("Starting bulk writer...")

    while True:
        try:
            await process_batch()
            # No sleep needed here; xreadgroup handles the wait
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(1)  # Backoff on error


if __name__ == "__main__":
    asyncio.run(main())
