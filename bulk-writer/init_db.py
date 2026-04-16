import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from settings import settings

logger = logging.getLogger("db-init")
logging.basicConfig(level=logging.INFO)

async def init_db():
    client = AsyncIOMotorClient(settings.MONGO_URL)
    db = client[settings.MONGO_DB]
    collection = db.scans

    logger.info("Initializing MongoDB...")

    # 1. Create Indexes
    # Creating a unique index on 'scan_id' or 'hash' prevents duplicates
    # and makes lookups lightning fast.
    try:
        index_name = await collection.create_index(
            [("task_id", 1)], 
            unique=True, 
            background=True
        )
        logger.info(f"Created index: {index_name}")
        
        # Index on timestamp for efficient cleanup/sorting
        await collection.create_index([("created_at", -1)])
        logger.info("Created timestamp index")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")

    # 2. Optional: Seed Data
    # Only insert if the collection is empty
    count = await collection.count_documents({})
    if count == 0:
        logger.info("Seeding initial data...")
        await collection.insert_one({"info": "Database initialized", "version": "1.0"})

    logger.info("Database initialization complete.")
    client.close()

if __name__ == "__main__":
    asyncio.run(init_db())