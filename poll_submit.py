from kaggle.api.kaggle_api_extended import KaggleApi
import time, sys

api = KaggleApi()
api.authenticate()
done = False
last_status = None
while not done:
    try:
        status = api.kernels_status('chrisyu2021/biohub-ct-improved')
        s = str(status.status)
        if s != last_status:
            print(f'[{int(time.time())}] {s}')
            sys.stdout.flush()
            last_status = s
        if 'COMPLETE' in s:
            result = api.competition_submit_code(
                file_name='submission.csv',
                message='v15: fix CSV format, DoG+ILP+gap-repair+divaug',
                competition='biohub-cell-tracking-during-development',
                kernel='chrisyu2021/biohub-ct-improved',
                kernel_version=15,
            )
            print(f'SUBMITTED: {result}')
            sys.stdout.flush()
            done = True
        elif 'ERROR' in s:
            print(f'ERROR: {status.failureMessage}')
            sys.stdout.flush()
            done = True
        else:
            time.sleep(15)
    except Exception as e:
        print(f'EXCEPTION: {e}')
        sys.stdout.flush()
        time.sleep(30)
