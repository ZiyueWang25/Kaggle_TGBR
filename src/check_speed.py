# https://zenn.dev/currypurin/scraps/47d5f84a0ca89d
from kaggle.api.kaggle_api_extended import KaggleApi
import datetime
from datetime import timezone
import time
if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    COMPETITION =  'tensorflow-great-barrier-reef'
    list_of_submission = api.competition_submissions(COMPETITION)
    show_num = 10
    for i,result in enumerate(list_of_submission):
        status = result.status
        #print(dir(result)) # check its attributes
        name = result.description
        submit_time = result.date
        now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        elapsed_time = int((now - submit_time).seconds / 60) + 1
        if status == 'complete':
            print('\r', f'{name} start {submit_time} run-time: {elapsed_time} min, LB: {result.publicScore}, private Score: {result.privateScore}')
        else:
            print('\r', f'elapsed time: {elapsed_time} min', end='')
        if i > show_num:
            break