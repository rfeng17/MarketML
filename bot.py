#!/usr/bin/env python3
# bot.py

import os
import discord
from dotenv import load_dotenv
import asyncio

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')

# Set up intents (required for message content access)
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)
            
async def notify_trades():
    
    channel_id = 530890688510951456 # Replace with your Discord channel ID
    
    channel = client.get_channel(channel_id)
    if channel is None:
        print("Channel not found! Please check the channel ID.")
    else:
        print(f"Channel found: {channel.name}")

    while True:
        try:
            # Read and clear the notification file
            with open("trade_notifications.txt", "r") as file:
                lines = file.readlines()
            open("trade_notifications.txt", "w").close()  # Clear the file after reading

            # Send each line as a message
            for line in lines:
                if channel and line.strip():
                    print(f"Sending message: {line.strip()}")
                    await channel.send(line.strip())
        except Exception as e:
            print(f"Error reading trade notifications: {e}")

        await asyncio.sleep(10)  # Check every 10 seconds

@client.event
async def on_ready():
    print(f'{client.user.name} is ready to make dough')
    open("trade_notifications.txt", "w").close()  # Makes sure file is empty for trade notifs
    client.loop.create_task(notify_trades())

@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Welcome trader {member.name}!'
    )
    
client.run(TOKEN)
