# Path: scripts/check_discord.py
"""One-shot diagnostic: connect, wait 30s for any message, report what we see."""
import asyncio, os, sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

token = os.environ.get("DISCORD_BOT_TOKEN", "")
if not token:
    sys.exit("ERROR: DISCORD_BOT_TOKEN not set")

import discord

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Connected as: {client.user}")
    print(f"message_content intent active: {client.intents.message_content}")
    print(f"Servers the bot is in: {[g.name for g in client.guilds]}")
    print(">> Send ANY message in Discord now. Waiting 30s...")
    await asyncio.sleep(30)
    print(">> 30s elapsed, no message received — closing.")
    await client.close()

@client.event
async def on_message(message):
    print(f"MESSAGE RECEIVED from {message.author} in #{message.channel}:")
    print(f"  content = {repr(message.content)}")
    print(f"  guild   = {message.guild}")
    await client.close()

client.run(token)
