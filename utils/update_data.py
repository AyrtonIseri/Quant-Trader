from scrapper.data_scrapper import DataScrapper
from datetime import date, datetime
from glob import glob

class Updater:
    
    def update_data(self, data_name: str = None):
        '''
        Run an entire cycle to delete old dat and replace it with the new one.

        :args:

        data_name: file to be monitored and, subsequently, updated

        '''

        data_scrapper = DataScrapper()
        
        if data_name is None:
            data_name = 'bitcoin_historical_data_until*'

        current_data = glob(data_name)

        if len(current_data) != 0:
            current_data = current_data[0]
            latest_date = current_data[-10:]
            latest_date = datetime.strptime(latest_date, '%d_%m_%Y').date()

            if latest_date == date.today():
                print('Model data is already up to date! \nProceding to the next step...\n')

            else:
                data_scrapper.delete_data()
                data_scrapper.scrapp_data()

        else:
            data_scrapper.delete_data()
            data_scrapper.scrapp_data()
