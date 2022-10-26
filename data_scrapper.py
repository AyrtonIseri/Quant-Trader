from datetime import date
import subprocess

class DataScrapper:

    def scrapp_data(self, url: str = None, name: str = None):

        bash_command = 'curl -C - -o '
        
        if name is None:
            name = 'bitcoin_historical_data_until_' + date.today().strftime('%d_%m_%Y') + ' '
        
        if url is None:
            url = 'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1635203032&period2=1666739032&interval=1d&events=history&includeAdjustedClose=true'

        #generates the complete bash command
        bash_command = bash_command + name + url

        #executes the command
        output = subprocess.Popen(bash_command.split(),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)
        
        out, err = output.communicate()
        
        print(out)
