import redis
import os
import random

RedisHost = "redis-17942.c39301.eu-central-1-mz.ec2.cloud.rlrcp.com"
RedisPort = 17942
RedisPassword = "a92aC3Y0u5n803wvO1QCElOpw5tKmW9p"

r = redis.Redis(
    host=RedisHost,
    port=RedisPort,
    password=RedisPassword,
    db=0,
    ssl=True,
    decode_responses=True
)

def make_key(user_id, name):
    return f"user:{user_id}:{name}"

def seed_user(user_id):
    r.set(make_key(user_id, "camera"), "on")
    r.set(make_key(user_id, "cleaning_control"), "start")
    r.set(make_key(user_id, "robot:state"), "idle")

    # Пример телеметрии
    telemetry = {
        "workTime": str(random.randint(0, 100)),
        "temperature": f"{random.uniform(20, 30):.2f}",
        "sensor1": str(random.randint(0, 100)),
        "sensor2": str(random.randint(0, 100)),
        "sensor3": str(random.randint(0, 100)),
        "sensor4": str(random.randint(0, 100)),
        "water": str(random.randint(0, 100)),
        "battery": str(random.randint(0, 100)),
        "moved": str(random.randint(0, 100))
    }

    r.hset(make_key(user_id, "telemetry"), mapping=telemetry)
    print(f"[✓] User {user_id} seeded.")

if __name__ == "__main__":
    seed_user(9)
