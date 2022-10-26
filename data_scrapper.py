from datetime import date
from glob import glob
import os
import subprocess

def execute_command(cmd: str):
    '''
    Executes a simples bash command line command and outputs the result

    :args:
    
    cmd: command line string to be executed
    '''

    output = subprocess.Popen(cmd.split(),
                              stdout=subprocess.PIPE,
                              universal_newlines=True)

    out, err = output.communicate()

    print(out)


class DataScrapper:

    def scrapp_data(self, url: str = None, name: str = None):
        '''
        Curl's new data from the url end point to local directory and output the command line results

        :args:

        url: URL to be Curl'ed. Default will be yahoo finance bitcoin data

        name: Scrapped file name to be saved in local directory

        '''

        bash_command = 'curl -C - -o '

        if name is None:
            name = 'bitcoin_historical_data_until_' + date.today().strftime('%d_%m_%Y') + ' '

        if url is None:
            url = 'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1635203032&period2=1666739032&interval=1d&events=history&includeAdjustedClose=true'

        #generates the complete bash command
        bash_command = bash_command + name + url

        execute_command(bash_command)


    def delete_data(self, arch:str = None):
        '''
        Utilize to delete old data. Should be imediatly followed by scrapp_data in order to achieve max. efficiency

        :args:

        arch: Archive to be delete by OS commanding. Default will be bitcoin_historical_data_until*
        '''

        if arch is None:
            arch = "bitcoin_historical_data_until*"

        arch = glob(arch)

        for archives in arch:
            os.remove(archives)