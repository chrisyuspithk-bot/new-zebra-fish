from kaggle.api.kaggle_api_extended import KaggleApi
import time, sys, json, os

API = KaggleApi()
API.authenticate()
KERNEL = 'chrisyu2021/biohub-ct-improved'
COMP = 'biohub-cell-tracking-during-development'
VERSION = 15
STATE_FILE = os.path.expanduser('~/.poll_submit_state.json')
MSG = 'v15: fix CSV format, DoG+ILP+gap-repair+divaug'

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_state(s):
    with open(STATE_FILE, 'w') as f:
        json.dump(s, f)

state = load_state()
if state.get('submitted'):
    print(f'Already submitted: ref={state.get("ref")}')
    sys.exit(0)

last_status = state.get('last_status')
attempts = state.get('attempts', 0)
while True:
    try:
        s = str(API.kernels_status(KERNEL).status)
    except Exception as e:
        print(f'[{int(time.time())}] status check failed: {e}')
        time.sleep(60)
        continue

    if s != last_status:
        print(f'[{int(time.time())}] {s}')
        sys.stdout.flush()
        last_status = s
        save_state({'last_status': s, 'attempts': attempts})

    if 'COMPLETE' in s:
        try:
            result = API.competition_submit_code(
                file_name='submission.csv',
                message=MSG,
                competition=COMP,
                kernel=KERNEL,
                kernel_version=VERSION,
            )
            print(f'SUBMITTED: ref={result.ref} url={result.url}')
            sys.stdout.flush()
            save_state({'submitted': True, 'ref': result.ref, 'last_status': s})
            sys.exit(0)
        except Exception as e:
            msg = str(e)
            is_limit = '400' in msg
            attempts += 1
            delay = 3600 if is_limit else 30  # 1h wait for daily limit, 30s for other
            print(f'[{int(time.time())}] submit #{attempts} failed ({e}) — retry in {delay}s')
            sys.stdout.flush()
            save_state({'last_status': s, 'attempts': attempts})
            time.sleep(delay)
        continue

    if 'ERROR' in s:
        fm = API.kernels_status(KERNEL).failureMessage
        print(f'KERNEL ERROR: {fm}')
        sys.stdout.flush()
        sys.exit(1)

    time.sleep(15)
