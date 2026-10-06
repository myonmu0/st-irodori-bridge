
# st-irodori-bridge
This bridge is intend to be used with [SillyTavern-IrodoriTTS](https://github.com/myonmu0/SillyTavern-IrodoriTTS).

## How to run
**Tested in a container running Ubuntu:24.04*

```
# 1. Install Irodori-TTS and make sure are working

# 2. Install st-irodori-bridge
git clone https://github.com/myonmu0/st-irodori-bridge

# 3. Run st-irodori-bridge
cd /path/to/your/Irodori-TTS
. .venv/bin/activate
cd /path/to/your/st-irodori-bridge
python3 ./st-irodori-bridge.py -v --irodori-dir /path/to/your/Irodori-TTS --host 127.0.0.1 --port 9040
```


## Security note
- This bridge don't have authentication, so DON'T expose this bridge to external network which anyone can access. If you want to run in Vastai/Runpod don't open the port, instead use SSH Tunnel so only you can access.

